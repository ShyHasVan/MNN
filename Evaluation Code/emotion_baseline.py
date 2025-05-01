# -*- coding: utf-8 -*-
"""
Single-Task RoBERTa for Emotion Recognition.

This script trains and evaluates a RoBERTa-based model specifically for
identifying the primary emotion conveyed in text (multi-class, based on
GoEmotions simplified). It is derived from a multitask script and maintains
consistent hyperparameters for comparison.

Key Features:
- Uses RoBERTa as the base transformer model.
- Handles the GoEmotions dataset (Hugging Face Hub and potentially local CSVs).
- Applies class weighting to address imbalance in the emotion dataset.
- Includes evaluation on standard multi-class classification metrics (Accuracy, F1, Precision, Recall).
- Uses hyperparameters consistent with the original multitask script.
"""
# !pip install transformers datasets scikit-learn torch pandas numpy
# (Uncomment the !pip install line if running in an environment like Google Colab)

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import warnings
import math 
# Suppress specific warnings for cleaner output (optional)
warnings.filterwarnings("ignore", category=UserWarning, message=".*DataFrameGroupBy.apply.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*is_sparse is deprecated.*")


# === Configuration (Consistent with Multitask Script) ===
# --- Hyperparameters ---
MODEL_NAME = "roberta-base" # Base model identifier from Hugging Face Hub
MAX_LENGTH = 128           # Max sequence length for tokenizer
BATCH_SIZE = 32            # Batch size
EPOCHS = 10                 # Number of training epochs
LEARNING_RATE = 1e-5       # Optimizer learning rate
DROPOUT_RATE = 0.3         # Dropout rate for regularization in classification head
WEIGHT_DECAY = 0.01        # Weight decay for AdamW optimizer
SCHEDULER_PATIENCE = 3     # Patience for ReduceLROnPlateau scheduler
# --- Other Configurations ---
DATA_DIR = "."             # Directory to load local CSV datasets from
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use GPU if available
print(f"Using device: {DEVICE}")

# === Tokenizer ===
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

# === Emotion Labels (Consistent with Multitask Script) ===
emotion_label_list = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]
NUM_EMOTION_CLASSES = len(emotion_label_list)
emotion_label_map = {label: i for i, label in enumerate(emotion_label_list)}
emotion_id_map = {i: label for i, label in enumerate(emotion_label_list)}


# === Data Preprocessing Function (Emotion Only) ===

def preprocess_goemotions(df, dataset_name):
    """
    Preprocesses the GoEmotions DataFrame for emotion recognition.
    Assigns a single primary label, encodes it, and calculates class weights.
    """
    print(f"Preprocessing GoEmotions dataset: {dataset_name}")
    df = df.copy() # Work on a copy

    # Determine the primary emotion label (using the first label if multiple exist in simplified)
    def get_primary(labels):
        if isinstance(labels, list) or isinstance(labels, np.ndarray):
            return labels[0] if len(labels) > 0 else -1
        elif isinstance(labels, (int, np.integer)):
            return labels
        return -1 # Default invalid label

    df['primary_label_id'] = df['labels'].apply(get_primary)
    initial_count = len(df)
    df = df[df['primary_label_id'] != -1].copy() # Filter out rows with no valid primary label
    filtered_count = len(df)
    if initial_count != filtered_count:
        print(f"INFO: GoEmotions - Filtered out {initial_count - filtered_count} rows with no primary label.")
    if filtered_count == 0:
        print(f"WARNING: GoEmotions - No valid data remaining after filtering for {dataset_name}.")
        return None, None # Return None if no data left

    # Use the primary_label_id directly as the label (it's already 0-27)
    df['label'] = df['primary_label_id']

    # --- Calculate Class Weights for CrossEntropyLoss ---
    class_weights = None
    df['label'] = df['label'].astype(int)
    unique_labels = sorted(df['label'].unique())
    unique_labels_np = np.array(unique_labels)

    if len(unique_labels_np) > 1:
        try:
            weights_array = compute_class_weight(
                class_weight='balanced',
                classes=unique_labels_np,
                y=df['label'].values
            )
            class_weights_full = torch.ones(NUM_EMOTION_CLASSES, dtype=torch.float)
            for label_id, weight in zip(unique_labels_np, weights_array):
                if 0 <= label_id < NUM_EMOTION_CLASSES:
                    class_weights_full[label_id] = weight
                else:
                    print(f"Warning: Label ID {label_id} out of expected range [0, {NUM_EMOTION_CLASSES-1}] during weight calculation.")

            class_weights = class_weights_full
            if torch.isinf(class_weights).any() or torch.isnan(class_weights).any() or (class_weights <= 0).any():
                 print(f"WARNING: Invalid weights calculated for GoEmotions {dataset_name}. Using default weights (1.0).")
                 class_weights = torch.ones(NUM_EMOTION_CLASSES, dtype=torch.float)
            else:
                 print(f"INFO: Calculated balanced class weights for {dataset_name} (Emotion). Weights tensor shape: {class_weights.shape}")

        except ValueError as e:
             print(f"WARNING: Could not compute class weights for GoEmotions {dataset_name}: {e}. Using default weights (1.0).")
             class_weights = torch.ones(NUM_EMOTION_CLASSES, dtype=torch.float)
    else:
        num_unique = len(unique_labels_np)
        label_info = unique_labels_np[0] if num_unique == 1 else 'None'
        print(f"WARNING: Only {num_unique} unique class ({label_info}) found in GoEmotions {dataset_name}. Using default weights (1.0).")
        class_weights = torch.ones(NUM_EMOTION_CLASSES, dtype=torch.float)

    # Select necessary columns and add dataset identifier
    df_final = df[['text', 'label']].copy()
    df_final['dataset_name'] = dataset_name # Keep dataset name for weight mapping
    # Ensure labels are integers
    df_final['label'] = df_final['label'].astype(int)
    return df_final, class_weights # Return preprocessed df and class weights tensor


# === Data Loading Function (Emotion Only) ===
def load_and_process_emotion_data(data_dir="."):
    """
    Loads emotion datasets (GoEmotions) from Hugging Face Hub and local CSV files.
    Preprocesses each dataset using the emotion function.
    Collects preprocessed dataframes and class_weights tensors.
    """
    all_dataframes = {}
    # Stores class_weights (tensor of size NUM_EMOTION_CLASSES) per dataset
    dataset_weights_map = {}

    # --- Load Hugging Face Datasets ---
    print("\n--- Loading Hugging Face emotion datasets ---")
    # GoEmotions (Emotion)
    try:
        print("Loading GoEmotions...")
        goemotions_ds = load_dataset("go_emotions", "simplified", trust_remote_code=True)["train"].to_pandas()
        processed_df, weights_tensor = preprocess_goemotions(goemotions_ds, "goemotions")
        if processed_df is not None:
            all_dataframes["goemotions"] = processed_df
            dataset_weights_map["goemotions"] = weights_tensor # Store class weights tensor
    except Exception as e: print(f"ERROR loading/processing GoEmotions: {e}")

    # --- Load Local CSV Datasets ---
    print(f"\n--- Loading local CSV emotion datasets from directory: {data_dir} ---")
    loaded_csv_count = 0
    for filename in os.listdir(data_dir):
        if filename.endswith(".csv"):
            dataset_name = filename[:-4].lower().replace(" ", "_")
            # Simple heuristic: only load if 'emotion' is in the name
            if 'emotion' not in dataset_name:
                print(f"Skipping CSV '{filename}', does not appear to be an emotion dataset.")
                continue
            if dataset_name in all_dataframes:
                print(f"Skipping CSV '{filename}', dataset '{dataset_name}' already processed.")
                continue

            filepath = os.path.join(data_dir, filename)
            print(f"Attempting to load CSV: {filename} as '{dataset_name}' (assuming emotion)")
            try:
                df = pd.read_csv(filepath)
                processed_df, weight = preprocess_goemotions(df, dataset_name)

                if processed_df is not None:
                    all_dataframes[dataset_name] = processed_df
                    dataset_weights_map[dataset_name] = weight # Store class weights tensor
                    loaded_csv_count += 1
                else: print(f"Skipped CSV '{filename}' due to preprocessing issues.")
            except Exception as e: print(f"Error loading or processing CSV '{filename}': {e}")
    print(f"Loaded {loaded_csv_count} local CSV datasets successfully.")

    # --- Final Check and Summary ---
    print("\n--- Emotion Datasets Loaded Summary ---")
    final_datasets = {}
    final_weights = {} # Store weights only for the successfully loaded datasets
    for name, df in all_dataframes.items():
        if df is not None and not df.empty:
             required_cols = ['text', 'label', 'dataset_name']
             if all(col in df.columns for col in required_cols):
                 final_datasets[name] = df
                 weight_info = "N/A"
                 if name in dataset_weights_map and dataset_weights_map[name] is not None:
                     if isinstance(dataset_weights_map[name], torch.Tensor):
                         weight_info = f"Class Weights Tensor (shape: {dataset_weights_map[name].shape})"
                     else:
                         weight_info = f"Weight Info: {type(dataset_weights_map[name])} (Expected Tensor)"
                     final_weights[name] = dataset_weights_map[name]
                 else: weight_info = "No weight calculated/stored"

                 # Calculate label distribution safely
                 label_dist_str = "N/A"
                 if 'label' in df.columns:
                     try:
                         counts = df['label'].value_counts().to_dict()
                         label_dist_str = str({int(k): int(v) for k, v in counts.items()})
                     except Exception as e:
                         label_dist_str = f"Error calculating distribution: {e}"

                 print(f"- {name}: {len(df)} samples, Label dist: {label_dist_str}, Weight Info: {weight_info}")
             else: print(f"WARNING: Skipping dataset '{name}' due to missing required columns after processing.")
        else: print(f"WARNING: Dataset '{name}' is None or empty after processing.")

    print("\n--- Final Class Weights Summary ---")
    if not final_weights: print("No weights stored for any datasets.")
    else:
        for name, weight in final_weights.items():
             if isinstance(weight, torch.Tensor):
                print(f"- {name}: class_weights tensor shape = {weight.shape}")
             else:
                print(f"- {name}: Weight type = {type(weight)} (Expected Tensor)")

    if not final_datasets:
        print("\nFATAL: No emotion datasets were loaded successfully. Exiting.")
        exit()

    return final_datasets, final_weights

# === Custom Dataset Class (Single Task) ===
class SingleTaskDataset(Dataset):
    """
    PyTorch Dataset class for a single task.
    """
    def __init__(self, dataframe, tokenizer, max_length):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        required_cols = ['text', 'label', 'dataset_name']
        if not all(col in dataframe.columns for col in required_cols):
             missing = [col for col in required_cols if col not in dataframe.columns]
             raise ValueError(f"Input DataFrame for SingleTaskDataset must contain required columns. Missing: {missing}")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        try:
            item = self.dataframe.iloc[idx]
            text = str(item['text'])
            label = item['label']
            dataset_name = item['dataset_name']
        except IndexError:
            raise IndexError(f"Index {idx} out of bounds for DataFrame of length {len(self.dataframe)}")
        except KeyError as e:
             raise KeyError(f"Missing expected column '{e}' in DataFrame row at index {idx}")

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        # Convert label to tensor (long for CrossEntropyLoss)
        try:
            label_tensor = torch.tensor(label, dtype=torch.long)
        except Exception as e:
             raise ValueError(f"Error converting label '{label}' at index {idx}: {e}")

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label_tensor,
            'dataset_name': dataset_name # Pass dataset name for weight lookup
        }

# === Model Definition (Single Task - Emotion) ===
class SingleTaskEmotionModel(nn.Module):
    """
    Single-task RoBERTa model with a head for emotion recognition.
    """
    def __init__(self, model_name=MODEL_NAME, num_emotion_classes=NUM_EMOTION_CLASSES, dropout_rate=DROPOUT_RATE):
        super(SingleTaskEmotionModel, self).__init__()
        print(f"Initializing single-task emotion model with base: {model_name}")
        self.roberta = RobertaModel.from_pretrained(model_name, output_attentions=True)
        hidden_size = self.roberta.config.hidden_size
        print(f"RoBERTa hidden size: {hidden_size}")

        # Emotion head structure consistent with the multitask version
        self.emotion_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate), # Use configured dropout rate
            nn.Linear(512, num_emotion_classes) # Output logits for each emotion class
        )
        print(f"Initialized emotion head (output: {num_emotion_classes} logits, dropout: {dropout_rate})")

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model.
        """
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.emotion_head(pooled_output)
        return logits, outputs.attentions

# === Training and Evaluation Functions (Adapted for Single Task) ===

def train_epoch(model, dataloader, optimizer, device, dataset_weights_map):
    """Trains the emotion model for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    skipped_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        try:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            dataset_names = batch['dataset_name']
        except Exception as e:
            print(f"Error moving batch {batch_idx} to device. Skipping. Error: {e}")
            skipped_batches += 1
            continue

        if not dataset_names:
             print(f"Warning: Batch {batch_idx} has empty dataset_names. Skipping.")
             skipped_batches += 1
             continue
        current_dataset = dataset_names[0]
        if not all(d == current_dataset for d in dataset_names):
            print(f"Warning: Batch {batch_idx} mixed dataset ({dataset_names}). Using weight for first: {current_dataset}.")

        optimizer.zero_grad()

        try:
            logits, _ = model(input_ids, attention_mask)
        except Exception as e:
            print(f"Error in forward pass batch {batch_idx}. Skipping. Error: {e}")
            skipped_batches += 1
            continue

        try:
            # Fetch class_weights (tensor of size NUM_EMOTION_CLASSES)
            class_weights = dataset_weights_map.get(current_dataset)
            loss_fn = None # Define based on weights availability
            if class_weights is None:
                print(f"Warning: No class_weights tensor found for dataset '{current_dataset}' in batch {batch_idx}. Using unweighted loss.")
                loss_fn = nn.CrossEntropyLoss() # Unweighted
            else:
                # Ensure it's a tensor and on the correct device
                if not isinstance(class_weights, torch.Tensor):
                     try:
                         class_weights = torch.tensor(class_weights, dtype=torch.float, device=device)
                     except Exception as conv_e:
                         print(f"ERROR: Could not convert class_weights for {current_dataset} to tensor: {conv_e}. Using unweighted loss.")
                         class_weights = None # Force unweighted loss
                else:
                    class_weights = class_weights.to(device)

                # Define loss function with the class weights tensor if available
                if class_weights is not None:
                     loss_fn = nn.CrossEntropyLoss(weight=class_weights)
                else:
                     loss_fn = nn.CrossEntropyLoss() # Fallback to unweighted

            loss = loss_fn(logits, labels.long())

            if torch.isnan(loss):
                print(f"Warning: NaN loss detected batch {batch_idx} (Dataset: {current_dataset}). Skipping backward pass.")
                skipped_batches += 1
                continue

        except Exception as e:
            print(f"Error during loss calculation batch {batch_idx}. Skipping. Error: {e}")
            skipped_batches += 1
            continue

        try:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
        except Exception as e:
            print(f"Error during backward/step batch {batch_idx}. Skipping step. Error: {e}")
            skipped_batches += 1
            optimizer.zero_grad()

    if skipped_batches > 0:
        print(f"Skipped {skipped_batches} batches during training epoch.")
    return total_loss / num_batches if num_batches > 0 else 0.0

def evaluate(model, dataloader, device, dataset_weights_map):
    """Evaluates the emotion model."""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0
    skipped_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            try:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                dataset_names = batch['dataset_name']
            except Exception as e:
                print(f"Error processing eval batch {batch_idx}. Skipping. Error: {e}")
                skipped_batches += 1
                continue

            if not dataset_names:
                print(f"Warning: Eval Batch {batch_idx} has empty dataset_names. Skipping.")
                skipped_batches += 1
                continue
            current_dataset = dataset_names[0]
            if not all(d == current_dataset for d in dataset_names):
                print(f"Warning: Eval Batch {batch_idx} mixed datasets ({dataset_names}). Using weight for first: {current_dataset}.")

            try:
                logits, _ = model(input_ids, attention_mask)
            except Exception as e:
                print(f"Error during eval forward pass batch {batch_idx}. Skipping. Error: {e}")
                skipped_batches += 1
                continue

            try:
                # Fetch class_weights
                class_weights = dataset_weights_map.get(current_dataset)
                loss_fn = None
                if class_weights is None:
                    loss_fn = nn.CrossEntropyLoss() # Unweighted
                else:
                     # Ensure tensor and device
                     if not isinstance(class_weights, torch.Tensor):
                         try:
                             class_weights = torch.tensor(class_weights, dtype=torch.float, device=device)
                         except Exception:
                             print(f"Eval Error: Could not convert emotion weights for {current_dataset}. Using unweighted.")
                             class_weights = None
                     else:
                        class_weights = class_weights.to(device)

                     if class_weights is not None:
                        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
                     else:
                        loss_fn = nn.CrossEntropyLoss() # Unweighted

                loss = loss_fn(logits, labels.long())
                preds = torch.argmax(logits, dim=1)

                if not torch.isnan(loss):
                    total_loss += loss.item()
                    num_batches += 1
                else:
                     print(f"Warning: NaN loss detected during evaluation batch {batch_idx}.")

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            except Exception as e:
                print(f"Error during eval loss/pred calculation batch {batch_idx}. Skipping. Error: {e}")
                skipped_batches += 1

    if skipped_batches > 0:
        print(f"Skipped {skipped_batches} batches during evaluation.")

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    # --- Calculate Metrics (Multi-class) ---
    if not all_labels or not all_preds or len(all_labels) != len(all_preds):
         print("Warning: No valid labels/predictions collected during evaluation. Cannot calculate metrics.")
         return avg_loss, 0.0, 0.0, 0.0, 0.0 # Return zero for metrics

    try:
        # Use 'weighted' average strategy for multi-class emotion task
        average_strategy = 'weighted'
        print(f"Using '{average_strategy}' averaging for metrics.")

        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average=average_strategy, zero_division=0)
        precision = precision_score(all_labels, all_preds, average=average_strategy, zero_division=0)
        recall = recall_score(all_labels, all_preds, average=average_strategy, zero_division=0)
    except Exception as e:
        print(f"Error calculating evaluation metrics: {e}")
        accuracy, f1, precision, recall = 0.0, 0.0, 0.0, 0.0 # Default to zero on error

    return avg_loss, accuracy, f1, precision, recall

# ===========================================
# === Main Script Execution ===
# ===========================================
if __name__ == "__main__":

    # 1. Load and Preprocess Emotion Data
    all_datasets_dict, dataset_weights_map = load_and_process_emotion_data(data_dir=DATA_DIR)

    # 2. Create Datasets and DataLoaders
    train_dataloaders = [] # List for train loaders
    val_dataloaders = {}   # Dict for validation loaders
    test_dataloaders = {}  # Dict for test loaders

    print("\n--- Creating PyTorch Datasets and DataLoaders for Emotion Task ---")
    for name, df in all_datasets_dict.items():
        print(f"Processing dataset for DataLoader: {name} ({len(df)} samples)")
        if df.empty:
            print(f"Skipping empty DataFrame for dataset: {name}")
            continue

        # Use 'label' for stratification (multi-class emotion)
        stratify_key = None
        if 'label' in df.columns and df['label'].nunique() > 1:
             stratify_key = df['label']
        else:
             print(f"Warning: Cannot stratify for {name}, 'label' column missing, empty, or has only one unique value.")

        is_test_set = name.startswith("test_") or "test" in name.lower()

        if is_test_set:
            print(f"Creating TEST dataloader for: {name}")
            try:
                test_dataset = SingleTaskDataset(df, tokenizer, MAX_LENGTH)
                test_dataloaders[name] = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
            except Exception as e:
                print(f"ERROR creating TEST dataloader for {name}: {e}")
        else: # Process as training/validation data
            try:
                train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=stratify_key)
            except ValueError as e:
                print(f"Warning: Could not stratify split for {name}. Using regular split. Error: {e}")
                train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

            # Create training DataLoader
            if not train_df.empty:
                 print(f"Creating TRAIN dataloader for: {name} ({len(train_df)} samples)")
                 try:
                     train_dataset = SingleTaskDataset(train_df, tokenizer, MAX_LENGTH)
                     train_dataloaders.append(DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True))
                 except Exception as e:
                     print(f"ERROR creating TRAIN dataloader for {name}: {e}")
            else: print(f"Skipping TRAIN dataloader creation for {name} - DataFrame empty after split.")

            # Create validation DataLoader
            if not val_df.empty:
                 print(f"Creating VALIDATION dataloader for: {name} ({len(val_df)} samples)")
                 try:
                     val_dataset = SingleTaskDataset(val_df, tokenizer, MAX_LENGTH)
                     val_dataloaders[name] = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
                 except Exception as e:
                     print(f"ERROR creating VALIDATION dataloader for {name}: {e}")
            else: print(f"Skipping VALIDATION dataloader creation for {name} - DataFrame empty after split.")

    print("\n--- Training Setup (Emotion Task) ---")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Weight Decay: {WEIGHT_DECAY}")
    print(f"Dropout Rate: {DROPOUT_RATE}")
    print(f"Scheduler Patience: {SCHEDULER_PATIENCE}")
    print(f"Total training dataset loaders: {len(train_dataloaders)}")
    print(f"Validation datasets: {list(val_dataloaders.keys())}")
    print(f"Test datasets: {list(test_dataloaders.keys())}")

    if not train_dataloaders:
        print("\nFATAL: No training dataloaders created. Cannot train. Exiting.")
        exit()

    # 3. Initialize Model, Optimizer, Scheduler
    model = SingleTaskEmotionModel(
        num_emotion_classes=NUM_EMOTION_CLASSES,
        dropout_rate=DROPOUT_RATE
        ).to(DEVICE)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
        )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=SCHEDULER_PATIENCE,
        verbose=True
        )

    # 4. Training Loop
    print("\n--- Starting Training (Emotion Task) ---")
    best_val_loss = float('inf')
    model_save_path = "emotion_model_best.pth" # Specific save path
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")
        epoch_train_loss = 0.0
        num_valid_train_loaders = 0
        print(f"Training on {len(train_dataloaders)} dataset loaders sequentially...")

        for i, train_loader in enumerate(train_dataloaders):
            loader_dataset_name = "Unknown"
            try:
                if len(train_loader.dataset) > 0:
                    loader_dataset_name = train_loader.dataset.dataframe['dataset_name'].iloc[0]
                else:
                    print(f"Skipping empty train loader {i+1}.")
                    continue
            except Exception as e:
                 loader_dataset_name = f"Loader_{i+1}"
                 print(f"Could not get dataset name for loader {i+1}, using generic name. Error: {e}")

            print(f"\nTraining on loader {i+1}/{len(train_dataloaders)} (Dataset: {loader_dataset_name})...")
            loader_avg_loss = train_epoch(model, train_loader, optimizer, DEVICE, dataset_weights_map)
            print(f"Avg Loss for loader {i+1} ({loader_dataset_name}): {loader_avg_loss:.4f}")
            if not math.isnan(loader_avg_loss) and not math.isinf(loader_avg_loss):
                epoch_train_loss += loader_avg_loss
                num_valid_train_loaders += 1
            else:
                print(f"Warning: Invalid training loss ({loader_avg_loss}) for loader {i+1}. Excluding from epoch average.")

        avg_epoch_train_loss = epoch_train_loss / num_valid_train_loaders if num_valid_train_loaders > 0 else 0.0
        print(f"\nEpoch {epoch + 1} Average Training Loss (across {num_valid_train_loaders} valid loaders): {avg_epoch_train_loss:.4f}")

        # --- Validation Step ---
        epoch_val_loss = 0.0
        num_valid_val_loaders = 0
        print("\n--- Validation ---")
        if not val_dataloaders:
            print("No validation datasets found. Skipping validation step.")
        else:
            for name, val_loader in val_dataloaders.items():
                if len(val_loader.dataset) == 0:
                     print(f"Skipping empty validation loader: {name}")
                     continue

                print(f"Evaluating on validation set: {name}")
                val_loss, accuracy, f1, precision, recall = evaluate(model, val_loader, DEVICE, dataset_weights_map)
                print(f"  - Val Loss: {val_loss:.4f}, Acc: {accuracy:.4f}, F1: {f1:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}")
                if not math.isnan(val_loss) and not math.isinf(val_loss):
                    epoch_val_loss += val_loss
                    num_valid_val_loaders += 1
                else:
                    print(f"  - Warning: Invalid validation loss ({val_loss}) for {name}. Excluding from average.")

            avg_epoch_val_loss = epoch_val_loss / num_valid_val_loaders if num_valid_val_loaders > 0 else float('inf')
            print(f"\nEpoch {epoch + 1} Average Validation Loss (across {num_valid_val_loaders} valid loaders): {avg_epoch_val_loss:.4f}")

            if avg_epoch_val_loss != float('inf'):
                if avg_epoch_val_loss < best_val_loss:
                    print(f"Validation loss improved ({best_val_loss:.4f} --> {avg_epoch_val_loss:.4f}). Saving model to {model_save_path}...")
                    try:
                        torch.save(model.state_dict(), model_save_path)
                        best_val_loss = avg_epoch_val_loss
                        epochs_no_improve = 0
                    except Exception as e:
                        print(f"ERROR saving model: {e}")
                else:
                    epochs_no_improve += 1
                    print(f"Validation loss ({avg_epoch_val_loss:.4f}) did not improve from best ({best_val_loss:.4f}). Epochs without improvement: {epochs_no_improve}")

                scheduler.step(avg_epoch_val_loss)

            else:
                print("Average validation loss is invalid or no valid validation loaders. Skipping scheduler step and model saving check.")

    print("\n--- Training Finished ---")

    # 5. Load Best Model state for final evaluation
    print(f"\nLoading best emotion model state from {model_save_path}...")
    final_model = SingleTaskEmotionModel(
        num_emotion_classes=NUM_EMOTION_CLASSES,
        dropout_rate=DROPOUT_RATE
        ).to(DEVICE)
    if os.path.exists(model_save_path):
        try:
            final_model.load_state_dict(torch.load(model_save_path, map_location=DEVICE))
            print("Best emotion model loaded successfully.")
        except Exception as e:
            print(f"Error loading best model state from {model_save_path}: {e}. Using model from end of training instead.")
            final_model = model
    else:
        print(f"Best model file ({model_save_path}) not found. Using model from end of training.")
        final_model = model

    final_model.eval()

    # 6. Final Test Set Evaluation
    print("\n--- Final Test Set Evaluation (Emotion Task) ---")
    if not test_dataloaders:
        print("No test dataloaders found. Skipping final test evaluation.")
    else:
        for name, test_loader in test_dataloaders.items():
            if len(test_loader.dataset) == 0:
                 print(f"Skipping empty test loader: {name}")
                 continue
            print(f"Evaluating on Test Set: {name}")
            test_loss, accuracy, f1, precision, recall = evaluate(final_model, test_loader, DEVICE, dataset_weights_map)
            print(f"  - Test Loss: {test_loss:.4f}, Acc: {accuracy:.4f}, F1: {f1:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}")

    print("\n--- Emotion Script Finished ---")