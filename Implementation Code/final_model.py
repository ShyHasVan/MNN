# -*- coding: utf-8 -*-
"""
Multitask RoBERTa for Disagreement Detection and Emotion Recognition.

This script trains and evaluates a RoBERTa-based model capable of performing
two tasks simultaneously:
1.  **Disagreement Detection:** Classifying text as expressing disagreement (binary).
2.  **Emotion Recognition:** Identifying the primary emotion conveyed in text

Key Features:
- Uses RoBERTa as the base transformer model.
- Handles multiple datasets (Hugging Face Hub and local CSVs).
- Implements specific preprocessing logic for datasets like FNC, BD, Politifact.
- Applies class weighting to address imbalance in disagreement and emotion datasets.
- Includes evaluation on standard metrics (Accuracy, F1, Precision, Recall).
- Tunable hyperparameters.
"""
# !pip install transformers datasets nltk scikit-learn torch pandas numpy
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


# === Configuration ===
# --- Hyperparameters for Tuning ---
# Consider using tools like Optuna or Ray Tune for systematic hyperparameter search.
MODEL_NAME = "roberta-base" # Base model identifier from Hugging Face Hub
MAX_LENGTH = 128           # Max sequence length for tokenizer
BATCH_SIZE = 32            # Batch size (Adjust based on GPU memory)
EPOCHS = 10                # Number of training epochs (Increased from 5) <<< TUNED
LEARNING_RATE = 1e-5       # Optimizer learning rate (Standard for transformers)
DROPOUT_RATE = 0.3         # Dropout rate for regularization in classification heads
WEIGHT_DECAY = 0.01        # Weight decay for AdamW optimizer (helps prevent overfitting)
SCHEDULER_PATIENCE = 3     # Patience for ReduceLROnPlateau scheduler (Increased from 2) <<< TUNED
# --- Other Configurations ---
DATA_DIR = "."             # Directory to load local CSV datasets from
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use GPU if available
print(f"Using device: {DEVICE}")

# === Tokenizer ===
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

# === Emotion Labels ===
# Based on the simplified GoEmotions dataset label set
emotion_label_list = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]
NUM_EMOTION_CLASSES = len(emotion_label_list)
# Map labels to integers for internal use
emotion_label_map = {label: i for i, label in enumerate(emotion_label_list)}
emotion_id_map = {i: label for i, label in enumerate(emotion_label_list)}


# === Data Preprocessing Functions ===

def preprocess_goemotions(df, dataset_name):
    """
    Preprocesses the GoEmotions DataFrame for emotion recognition.
    Assigns a single primary label, encodes it, and calculates class weights.
    """
    print(f"Preprocessing GoEmotions dataset: {dataset_name}")
    df = df.copy() # Work on a copy

    # Determine the primary emotion label (using the first label if multiple exist in simplified)
    # GoEmotions simplified usually has only one label per example, but handle lists just in case
    def get_primary(labels):
        if isinstance(labels, list) or isinstance(labels, np.ndarray):
            # Ensure list is not empty before accessing index 0
            return labels[0] if len(labels) > 0 else -1
        elif isinstance(labels, (int, np.integer)): # If already integer encoded (incl. numpy int types)
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
    # Ensure labels are integers and get unique ones
    df['label'] = df['label'].astype(int)
    unique_labels = sorted(df['label'].unique())
    # Convert unique_labels to a NumPy array
    unique_labels_np = np.array(unique_labels)

    if len(unique_labels_np) > 1: # Need at least 2 classes to compute weights
        try:
            # Use scikit-learn's utility to compute 'balanced' weights
            # Pass the NumPy array to the 'classes' parameter
            weights_array = compute_class_weight(
                class_weight='balanced',
                classes=unique_labels_np, # Use the numpy array
                y=df['label'].values
            )
            # Create a tensor mapping class index to its weight
            # even if some are missing in this specific subset. Default weight is 1.0.
            class_weights_full = torch.ones(NUM_EMOTION_CLASSES, dtype=torch.float)
            # Use the numpy array for zipping
            for label_id, weight in zip(unique_labels_np, weights_array):
                if 0 <= label_id < NUM_EMOTION_CLASSES:
                    class_weights_full[label_id] = weight
                else:
                    print(f"Warning: Label ID {label_id} out of expected range [0, {NUM_EMOTION_CLASSES-1}] during weight calculation.")

            class_weights = class_weights_full
            # Ensure no inf/nan/non-positive weights
            if torch.isinf(class_weights).any() or torch.isnan(class_weights).any() or (class_weights <= 0).any():
                 print(f"WARNING: Invalid weights calculated for GoEmotions {dataset_name} (inf/nan/non-positive). Using default weights (1.0).")
                 class_weights = torch.ones(NUM_EMOTION_CLASSES, dtype=torch.float)
            else:
                 print(f"INFO: Calculated balanced class weights for {dataset_name} (Emotion). Weights tensor shape: {class_weights.shape}")
            

        except ValueError as e:
             print(f"WARNING: Could not compute class weights for GoEmotions {dataset_name} (likely due to insufficient classes/samples: {e}). Using default weights (1.0).")
             class_weights = torch.ones(NUM_EMOTION_CLASSES, dtype=torch.float)
    else:
        # Handle case with 0 or 1 unique label
        num_unique = len(unique_labels_np)
        label_info = unique_labels_np[0] if num_unique == 1 else 'None'
        print(f"WARNING: Only {num_unique} unique class ({label_info}) found in GoEmotions {dataset_name}. Cannot compute meaningful class weights. Using default weights (1.0).")
        class_weights = torch.ones(NUM_EMOTION_CLASSES, dtype=torch.float)


    # Select necessary columns and add task/dataset identifiers
    df_final = df[['text', 'label']].copy()
    df_final['task_type'] = 'emotion'
    df_final['dataset_name'] = dataset_name
    # Ensure labels are integers
    df_final['label'] = df_final['label'].astype(int)
    return df_final, class_weights # Return preprocessed df and class weights tensor

def preprocess_disagreement(df, dataset_name):
    """
    Preprocesses various disagreement/stance/fact-checking datasets.
    - Identifies text and label columns.
    - Maps diverse original labels to binary disagreement (0: No/Agree, 1: Yes/Disagree/False).
    - Calculates class weight (`pos_weight`) for BCEWithLogitsLoss.
    """
    print(f"Preprocessing disagreement dataset: {dataset_name}")
    df = df.copy() # Work on a copy to avoid modifying original DataFrame
    original_label_col = None
    pos_weight = None # Class weight for the positive class (disagreement)

    # Identify the text column
    potential_text_cols = ['text', 'statement', 'headline', 'title', 'content', 'title2_en', 'Body']
    found_text_col = None
    if 'text' not in df.columns:
        for col in potential_text_cols:
            if col in df.columns:
                found_text_col = col
                break
        if found_text_col:
            print(f"INFO: Renaming text column '{found_text_col}' to 'text' for {dataset_name}.")
            df.rename(columns={found_text_col: 'text'}, inplace=True)
        else:
            print(f"FATAL: Could not find a suitable text column for {dataset_name} using options {potential_text_cols}. Skipping.")
            return None, None
    df['text'] = df['text'].astype(str) # Ensure text column is string type

    # Identify the label column based on dataset specifics
    if dataset_name == "politifact":
        if 'orig_label' in df.columns: # Handle case where we already renamed it
            original_label_col = 'orig_label'
        elif 'label' in df.columns:
             original_label_col = 'label'
        else:
             potential_label_cols_pf = ["verdict", "claim_label"]
             for col in potential_label_cols_pf:
                 if col in df.columns:
                     original_label_col = col
                     break
    else:
        potential_label_cols_other = ["label", "Stance", "bd_label", "hyperpartisan", "stance"]
        for col in potential_label_cols_other:
            if col in df.columns:
                original_label_col = col
                break

    if original_label_col is None:
        print(f"Warning: Could not find suitable label column for {dataset_name}. Skipping.")
        return None, None

    # Map original labels to binary disagreement labels (0 or 1)
    df[original_label_col] = df[original_label_col].astype(str)
    if dataset_name == "politifact":
        try:
            numeric_labels = pd.to_numeric(df[original_label_col], errors='coerce')
            numeric_labels.fillna(-1, inplace=True)
            # Liar labels: 0:false, 1:half-true, 2:mostly-true, 3:true, 4:barely-true, 5:pants-fire
            # Map {0, 4, 5, -1} -> 1 (disagree/false-ish), {1, 2, 3} -> 0 (agree/true-ish) - MAPPING FOR DISAGREEMENT = 1
            false_ish_codes = [0, 4, 5, -1]
            df['label'] = numeric_labels.apply(lambda x: 1 if x in false_ish_codes else 0)
            print(f"INFO: Applied Politifact numeric mapping (codes {false_ish_codes} -> 1 (Disagree), others -> 0) for {dataset_name}.")
        except Exception as e:
             print(f"ERROR applying Politifact numeric mapping for {dataset_name}. Defaulting all to 0. Error: {e}")
             df['label'] = 0
    elif dataset_name == "hyperpartisan":
        # Map 'true' (hyperpartisan) -> 1 (Disagree), 'false' -> 0
        df['label'] = df[original_label_col].apply(lambda x: 1 if str(x).lower() == 'true' else 0)
        print(f"INFO: Applied Hyperpartisan mapping ('true' -> 1 (Disagree), 'false' -> 0) for {dataset_name}.")
    elif "fnc" in dataset_name.lower():
        # FNC mapping: 'disagree', 'discuss', 'unrelated' -> 1 (Disagree), 'agree' -> 0
        negative_labels_fnc = ['agree']
        df['label'] = df[original_label_col].str.lower().apply(lambda x: 0 if x in negative_labels_fnc else 1)
        print(f"INFO: Applied FNC mapping ('agree' -> 0, others -> 1 (Disagree)) for {dataset_name}.")
    elif "bd" in dataset_name.lower():
         # BD mapping: 'disagreed' -> 1 (Disagree), 'agreed' -> 0
         negative_labels_bd = ['agreed']
         df['label'] = df[original_label_col].str.lower().apply(lambda x: 0 if x in negative_labels_bd else 1)
         print(f"INFO: Applied BD mapping ('agreed' -> 0, 'disagreed' -> 1) for {dataset_name}.")
    else: # Generic fallback
        print(f"WARNING: Applying GENERIC FALLBACK mapping for {dataset_name}. Verify this logic is appropriate.")
        try:
            temp_labels = pd.to_numeric(df[original_label_col], errors='coerce')
            if temp_labels.notna().all():
                 # Assume 0 means agreement, non-zero means disagreement (adjust if needed)
                 df['label'] = temp_labels.apply(lambda x: 0 if x == 0 else 1)
                 print(f"INFO: Applied numeric fallback mapping (0 -> 0, non-zero -> 1 (Disagree)) for {dataset_name}.")
            else:
                 str_labels = df[original_label_col].astype(str).str.lower()
                 # Define indicators of agreement/truth -> map these to 0, others to 1 (Disagree)
                 negative_indicators = ['true', 'agree', '1', 'yes', 'support', 'agreed']
                 df['label'] = str_labels.apply(lambda x: 0 if any(indicator in x for indicator in negative_indicators) else 1)
                 print(f"INFO: Applied string indicator fallback mapping (agreement words -> 0, others -> 1 (Disagree)) for {dataset_name}.")
        except Exception as e:
            print(f"Error applying generic mapping for {dataset_name}. Defaulting all to 0. Error: {e}")
            df['label'] = 0

    # Calculate Class Weights for BCEWithLogitsLoss (Weight for the POSITIVE class = 1)
    if 'label' in df.columns:
        label_counts = df['label'].value_counts()
        # Check if both classes exist and positive count is > 0
        if 0 in label_counts and 1 in label_counts and label_counts.get(1, 0) > 0:
            neg_count = label_counts.get(0, 0)
            pos_count = label_counts.get(1, 0)
            weight_val = neg_count / pos_count # weight = neg/pos
            # Sanity check
            if math.isinf(weight_val) or math.isnan(weight_val) or weight_val <= 0:
                 print(f"WARNING: Invalid weight calculated ({weight_val}) for {dataset_name}. Using default 1.0.")
                 pos_weight = torch.tensor(1.0, dtype=torch.float)
            else:
                 pos_weight = torch.tensor(weight_val, dtype=torch.float)
                 print(f"INFO: Calculated pos_weight for {dataset_name}: {pos_weight.item():.4f} ({neg_count} neg / {pos_count} pos)")
        else: # Handle cases with only one class or zero positive samples
            pos_weight = torch.tensor(1.0, dtype=torch.float)
            print(f"WARNING: Could not calculate valid pos_weight for {dataset_name} (counts: {label_counts.to_dict()}). Using default weight 1.0.")
    else:
        print(f"WARNING: 'label' column not found before weight calculation for {dataset_name}. Using default weight 1.0.")
        pos_weight = torch.tensor(1.0, dtype=torch.float)

    # Finalize DataFrame
    if 'label' not in df.columns:
        print(f"FATAL: 'label' column missing after mapping for {dataset_name}. Skipping.")
        return None, None
    df_final = df[['text', 'label']].copy()
    df_final['task_type'] = 'disagreement'
    df_final['dataset_name'] = dataset_name
    if df_final['label'].isnull().any():
        nan_count = df_final['label'].isnull().sum()
        print(f"WARNING: Found {nan_count} NaN labels in final dataframe for {dataset_name}. Filling with 0 (assuming non-disagreement).")
        df_final['label'].fillna(0, inplace=True)
    df_final['label'] = df_final['label'].astype(int)
    # Return weight for the positive class (disagreement = 1)
    return df_final, pos_weight

# === Data Loading Function ===
def load_and_process_all_data(data_dir="."):
    """
    Loads datasets from Hugging Face Hub and local CSV files.
    Preprocesses each dataset using the appropriate function.
    Collects preprocessed dataframes and class weights.
    """
    all_dataframes = {}
    # Stores class weights:
    # - For 'disagreement': pos_weight (single tensor value)
    # - For 'emotion': class_weights (tensor of size NUM_EMOTION_CLASSES)
    dataset_weights_map = {}

    # --- Load Hugging Face Datasets ---
    print("\n--- Loading Hugging Face datasets ---")
    # Politifact (Disagreement)
    try:
        print("Loading Politifact (liar)...")
        politifact_ds = load_dataset("liar", trust_remote_code=True)["train"].to_pandas()
        politifact_ds.rename(columns={"statement": "text", "label": "orig_label"}, inplace=True)
        processed_df, weight = preprocess_disagreement(politifact_ds, "politifact")
        if processed_df is not None:
            all_dataframes["politifact"] = processed_df
            dataset_weights_map["politifact"] = weight # Store pos_weight
    except Exception as e: print(f"ERROR loading/processing Politifact: {e}")

    # Hyperpartisan News Detection (Disagreement)
    try:
        print("Loading Hyperpartisan News Detection...")
        hyperpartisan_ds = load_dataset("hyperpartisan_news_detection", "byarticle", trust_remote_code=True)["train"].to_pandas()
        processed_df, weight = preprocess_disagreement(hyperpartisan_ds, "hyperpartisan")
        if processed_df is not None:
            all_dataframes["hyperpartisan"] = processed_df
            dataset_weights_map["hyperpartisan"] = weight # Store pos_weight
    except Exception as e: print(f"ERROR loading/processing Hyperpartisan: {e}")

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
    print(f"\n--- Loading local CSV datasets from directory: {data_dir} ---")
    loaded_csv_count = 0
    for filename in os.listdir(data_dir):
        if filename.endswith(".csv"):
            dataset_name = filename[:-4].lower().replace(" ", "_")
            if dataset_name in all_dataframes:
                print(f"Skipping CSV '{filename}', dataset '{dataset_name}' already processed.")
                continue
            filepath = os.path.join(data_dir, filename)
            print(f"Attempting to load CSV: {filename} as '{dataset_name}'")
            try:
                df = pd.read_csv(filepath)
                # --- Determine Task Type based on filename or content (heuristic) ---
                # Simple heuristic: if 'emotion' in name, treat as emotion, else disagreement
                # This might need refinement based on your actual local file naming/content
                task_type = 'unknown'
                if 'emotion' in dataset_name:
                    task_type = 'emotion'
                    print(f"INFO: Assuming '{dataset_name}' is an EMOTION dataset based on name.")
                    processed_df, weight = preprocess_goemotions(df, dataset_name)
                else:
                    task_type = 'disagreement'
                    print(f"INFO: Assuming '{dataset_name}' is a DISAGREEMENT dataset.")
                    processed_df, weight = preprocess_disagreement(df, dataset_name)

                if processed_df is not None:
                    all_dataframes[dataset_name] = processed_df
                    dataset_weights_map[dataset_name] = weight # Store weight (pos_weight or tensor)
                    loaded_csv_count += 1
                else: print(f"Skipped CSV '{filename}' due to preprocessing issues.")
            except Exception as e: print(f"Error loading or processing CSV '{filename}': {e}")
    print(f"Loaded {loaded_csv_count} local CSV datasets successfully.")

    # --- Final Check and Summary ---
    print("\n--- Datasets Loaded Summary ---")
    final_datasets = {}
    final_weights = {} # Store weights only for the successfully loaded datasets
    for name, df in all_dataframes.items():
        if df is not None and not df.empty:
             required_cols = ['text', 'label', 'task_type', 'dataset_name']
             if all(col in df.columns for col in required_cols):
                 final_datasets[name] = df
                 task = df['task_type'].iloc[0]
                 weight_info = "N/A"
                 if name in dataset_weights_map and dataset_weights_map[name] is not None:
                     if task == 'disagreement':
                         # Ensure it's a tensor before calling .item()
                         if isinstance(dataset_weights_map[name], torch.Tensor):
                             weight_info = f"Pos Weight: {dataset_weights_map[name].item():.4f}"
                         else:
                             weight_info = f"Pos Weight: {dataset_weights_map[name]:.4f}" # Assume float if not tensor
                         final_weights[name] = dataset_weights_map[name]
                     elif task == 'emotion':
                         if isinstance(dataset_weights_map[name], torch.Tensor):
                             weight_info = f"Class Weights Tensor (shape: {dataset_weights_map[name].shape})"
                         else:
                             weight_info = f"Weight Info: {type(dataset_weights_map[name])} (Expected Tensor)"
                         final_weights[name] = dataset_weights_map[name]
                     else: weight_info = "Weight stored, unknown type"
                 else: weight_info = "No weight calculated/stored"

                 # Calculate label distribution safely
                 label_dist_str = "N/A"
                 if 'label' in df.columns:
                     try:
                         # Convert numpy int types to standard Python ints for printing if needed
                         counts = df['label'].value_counts().to_dict()
                         label_dist_str = str({int(k): int(v) for k, v in counts.items()})
                     except Exception as e:
                         label_dist_str = f"Error calculating distribution: {e}"

                 print(f"- {name}: {len(df)} samples, Task: {task}, Label dist: {label_dist_str}, Weight Info: {weight_info}")
             else: print(f"WARNING: Skipping dataset '{name}' due to missing required columns after processing.")
        else: print(f"WARNING: Dataset '{name}' is None or empty after processing.")

    print("\n--- Final Class Weights Summary ---")
    if not final_weights: print("No weights stored for any datasets.")
    else:
        for name, weight in final_weights.items():
             task = "unknown"
             if name in final_datasets and 'task_type' in final_datasets[name].columns:
                 # Ensure task type is available and valid before accessing
                 if not final_datasets[name]['task_type'].empty:
                     task = final_datasets[name]['task_type'].iloc[0]

             if task == 'disagreement':
                 weight_val = weight.item() if isinstance(weight, torch.Tensor) else weight
                 print(f"- {name} (Disagreement): pos_weight = {weight_val:.4f}")
             elif task == 'emotion':
                 if isinstance(weight, torch.Tensor):
                    print(f"- {name} (Emotion): class_weights tensor shape = {weight.shape}")
                 else:
                    print(f"- {name} (Emotion): Weight type = {type(weight)} (Expected Tensor)")
             else:
                 print(f"- {name} (Task: {task}): Weight type: {type(weight)}")


    if not final_datasets:
        print("\nFATAL: No datasets were loaded successfully. Exiting.")
        exit()

    return final_datasets, final_weights

# === Custom Dataset Class ===
class MultiTaskDataset(Dataset):
    """
    PyTorch Dataset class to handle samples from different tasks.
    """
    def __init__(self, dataframe, tokenizer, max_length):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        required_cols = ['text', 'label', 'task_type', 'dataset_name']
        if not all(col in dataframe.columns for col in required_cols):
             missing = [col for col in required_cols if col not in dataframe.columns]
             raise ValueError(f"Input DataFrame for MultiTaskDataset must contain required columns. Missing: {missing}")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        try:
            item = self.dataframe.iloc[idx]
            text = str(item['text'])
            label = item['label']
            task_type = item['task_type']
            dataset_name = item['dataset_name']
        except IndexError:
            # Handle potential index errors more gracefully
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

        # Convert label to tensor based on task type
        try:
            if task_type == 'disagreement':
                label_tensor = torch.tensor(label, dtype=torch.float)
            elif task_type == 'emotion':
                label_tensor = torch.tensor(label, dtype=torch.long)
            else:
                raise ValueError(f"Unknown task_type '{task_type}' encountered in dataset.")
        except Exception as e:
             # Add more context to the error
             raise ValueError(f"Error converting label '{label}' for task '{task_type}' at index {idx}: {e}")


        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label_tensor,
            'task_type': task_type,
            'dataset_name': dataset_name
        }

# === Model Definition ===
class MultiTaskModel(nn.Module):
    """
    Multitask RoBERTa model with separate heads for disagreement and emotion tasks.
    """
    def __init__(self, model_name=MODEL_NAME, num_emotion_classes=NUM_EMOTION_CLASSES, dropout_rate=DROPOUT_RATE):
        super(MultiTaskModel, self).__init__()
        print(f"Initializing multitask model with base: {model_name}")
        self.roberta = RobertaModel.from_pretrained(model_name, output_attentions=True)
        hidden_size = self.roberta.config.hidden_size
        print(f"RoBERTa hidden size: {hidden_size}")

        self.disagreement_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate), # Use configured dropout rate
            nn.Linear(512, 1)
        )
        print(f"Initialized disagreement head (output: 1 logit, dropout: {dropout_rate})")

        self.emotion_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate), # Use configured dropout rate
            nn.Linear(512, num_emotion_classes)
        )
        print(f"Initialized emotion head (output: {num_emotion_classes} logits, dropout: {dropout_rate})")

    def forward(self, input_ids, attention_mask, task_type):
        """
        Forward pass of the model. Routes input to the appropriate head based on task_type.
        """
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output

        # Ensure task_type is not empty and access the first element
        if not task_type:
             raise ValueError("task_type list cannot be empty during forward pass.")
        batch_task_type = task_type[0] # Assumes batch homogeneity

        if batch_task_type == 'disagreement':
            logits = self.disagreement_head(pooled_output)
        elif batch_task_type == 'emotion':
            logits = self.emotion_head(pooled_output)
        else:
            raise ValueError(f"Unknown task_type '{batch_task_type}' encountered during forward pass.")

        return logits, outputs.attentions

# === Training and Evaluation Functions ===

def train_epoch(model, dataloader, optimizer, device, dataset_weights_map):
    """Trains the model for one epoch on the provided dataloader."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    skipped_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        try:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            task_type = batch['task_type']
            dataset_names = batch['dataset_name']
        except Exception as e:
            print(f"Error moving batch {batch_idx} to device. Skipping. Error: {e}")
            skipped_batches += 1
            continue

        # Basic check for batch homogeneity
        if not task_type or not dataset_names:
             print(f"Warning: Batch {batch_idx} has empty task_type or dataset_names. Skipping.")
             skipped_batches += 1
             continue
        current_task = task_type[0]
        current_dataset = dataset_names[0]
        if not all(t == current_task for t in task_type) or not all(d == current_dataset for d in dataset_names):
            print(f"Warning: Batch {batch_idx} mixed task/dataset ({task_type} / {dataset_names}). Skipping.")
            skipped_batches += 1
            continue

        optimizer.zero_grad()

        try:
            logits, _ = model(input_ids, attention_mask, task_type)
        except Exception as e:
            print(f"Error in forward pass batch {batch_idx}. Skipping. Error: {e}")
            skipped_batches += 1
            continue

        try:
            loss = None # Initialize loss
            if current_task == 'disagreement':
                # Fetch pos_weight (single value tensor)
                pos_weight = dataset_weights_map.get(current_dataset)
                if pos_weight is None:
                    print(f"Warning: No pos_weight found for disagreement dataset '{current_dataset}' in batch {batch_idx}. Using weight 1.0.")
                    pos_weight = torch.tensor(1.0, device=device)
                else:
                     # Ensure it's a tensor and on the correct device
                     if not isinstance(pos_weight, torch.Tensor):
                         pos_weight = torch.tensor(pos_weight, dtype=torch.float, device=device)
                     else:
                         pos_weight = pos_weight.to(device)

                # Define loss function with the specific weight
                disagreement_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                loss = disagreement_loss_fn(logits.squeeze(-1), labels.float())

            elif current_task == 'emotion':
                # Fetch class_weights (tensor of size NUM_EMOTION_CLASSES)
                class_weights = dataset_weights_map.get(current_dataset)
                emotion_loss_fn = None # Define based on weights availability
                if class_weights is None:
                    print(f"Warning: No class_weights tensor found for emotion dataset '{current_dataset}' in batch {batch_idx}. Using unweighted loss.")
                    emotion_loss_fn = nn.CrossEntropyLoss() # Unweighted
                else:
                    # Ensure it's a tensor and on the correct device
                    if not isinstance(class_weights, torch.Tensor):
                         # Attempt conversion, assuming it's list/array like
                         try:
                             class_weights = torch.tensor(class_weights, dtype=torch.float, device=device)
                         except Exception as conv_e:
                             print(f"ERROR: Could not convert class_weights for {current_dataset} to tensor: {conv_e}. Using unweighted loss.")
                             class_weights = None # Force unweighted loss
                    else:
                        class_weights = class_weights.to(device)

                    # Define loss function with the class weights tensor if available
                    if class_weights is not None:
                         emotion_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
                    else:
                         emotion_loss_fn = nn.CrossEntropyLoss() # Fallback to unweighted

                loss = emotion_loss_fn(logits, labels.long())
            else:
                print(f"Warning: Unknown task '{current_task}' in loss calc batch {batch_idx}. Skipping loss.")
                skipped_batches += 1
                continue

            if loss is not None and torch.isnan(loss):
                print(f"Warning: NaN loss detected batch {batch_idx} (Task: {current_task}, Dataset: {current_dataset}). Skipping backward pass.")
                skipped_batches += 1
                continue

            if loss is None: # Should only happen if task was unknown and not handled above
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
            optimizer.zero_grad() # Clear gradients if step failed

    if skipped_batches > 0:
        print(f"Skipped {skipped_batches} batches during training epoch.")
    return total_loss / num_batches if num_batches > 0 else 0.0

def evaluate(model, dataloader, device, dataset_weights_map):
    """Evaluates the model on the provided dataloader."""
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
                task_type = batch['task_type']
                dataset_names = batch['dataset_name']
            except Exception as e:
                print(f"Error processing eval batch {batch_idx}. Skipping. Error: {e}")
                skipped_batches += 1
                continue

            # Check batch homogeneity
            if not task_type or not dataset_names:
                print(f"Warning: Eval Batch {batch_idx} has empty task_type or dataset_names. Skipping.")
                skipped_batches += 1
                continue
            current_task = task_type[0]
            current_dataset = dataset_names[0]
            if not all(t == current_task for t in task_type) or not all(d == current_dataset for d in dataset_names):
                print(f"Warning: Eval Batch {batch_idx} mixed types ({task_type} / {dataset_names}). Skipping.")
                skipped_batches += 1
                continue

            try:
                logits, _ = model(input_ids, attention_mask, task_type)
            except Exception as e:
                print(f"Error during eval forward pass batch {batch_idx}. Skipping. Error: {e}")
                skipped_batches += 1
                continue

            try:
                loss = None # Initialize loss
                preds = None # Initialize preds
                if current_task == 'disagreement':
                    pos_weight = dataset_weights_map.get(current_dataset)
                    if pos_weight is None: pos_weight = torch.tensor(1.0, device=device)
                    else:
                         # Ensure it's a tensor and on the correct device
                         if not isinstance(pos_weight, torch.Tensor):
                              pos_weight = torch.tensor(pos_weight, dtype=torch.float, device=device)
                         else:
                              pos_weight = pos_weight.to(device)
                    disagreement_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                    loss = disagreement_loss_fn(logits.squeeze(-1), labels.float())
                    preds = (torch.sigmoid(logits).squeeze(-1) > 0.5).int()

                elif current_task == 'emotion':
                    class_weights = dataset_weights_map.get(current_dataset)
                    emotion_loss_fn = None # Define based on weights availability
                    if class_weights is None:
                        emotion_loss_fn = nn.CrossEntropyLoss() # Unweighted
                    else:
                         # Ensure tensor and device
                         if not isinstance(class_weights, torch.Tensor):
                             try:
                                 class_weights = torch.tensor(class_weights, dtype=torch.float, device=device)
                             except Exception: # Fallback if conversion fails
                                 print(f"Eval Error: Could not convert emotion weights for {current_dataset}. Using unweighted.")
                                 class_weights = None # Force unweighted loss
                         else:
                            class_weights = class_weights.to(device)

                         if class_weights is not None:
                            emotion_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
                         else: # Handle case where conversion failed
                            emotion_loss_fn = nn.CrossEntropyLoss() # Unweighted

                    loss = emotion_loss_fn(logits, labels.long())
                    preds = torch.argmax(logits, dim=1)
                else:
                    print(f"Warning: Unknown task '{current_task}' in eval batch {batch_idx}. Skipping.")
                    skipped_batches += 1
                    continue

                if loss is not None and not torch.isnan(loss):
                    total_loss += loss.item()
                    num_batches += 1
                elif loss is not None and torch.isnan(loss):
                     print(f"Warning: NaN loss detected during evaluation batch {batch_idx}.")

                if preds is not None:
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            except Exception as e:
                print(f"Error during eval loss/pred calculation batch {batch_idx}. Skipping. Error: {e}")
                skipped_batches += 1

    if skipped_batches > 0:
        print(f"Skipped {skipped_batches} batches during evaluation.")

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    # --- Calculate Metrics ---
    if not all_labels or not all_preds or len(all_labels) != len(all_preds):
         print("Warning: No valid labels/predictions collected during evaluation. Cannot calculate metrics.")
         return avg_loss, 0.0, 0.0, 0.0, 0.0 # Return zero for metrics

    try:
        # Ensure labels are integers for max() check
        int_labels = [int(l) for l in all_labels]
        # Determine average strategy based on task type using max label value
        # This assumes labels are 0/1 for binary and >1 for multiclass
        average_strategy = 'binary' # Default for disagreement
        if max(int_labels) > 1:
             average_strategy = 'weighted' # Good for multiclass imbalance (emotion)
             print(f"Using '{average_strategy}' averaging for metrics (max label = {max(int_labels)}).")
        else:
            print(f"Using '{average_strategy}' averaging for metrics (max label = {max(int_labels)}).")


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

    # 1. Load and Preprocess Data
    all_datasets_dict, dataset_weights_map = load_and_process_all_data(data_dir=DATA_DIR)

    # 2. Create Datasets and DataLoaders
    train_dataloaders = []
    val_dataloaders = {}
    test_dataloaders = {}

    print("\n--- Creating PyTorch Datasets and DataLoaders ---")
    for name, df in all_datasets_dict.items():
        print(f"Processing dataset for DataLoader: {name} ({len(df)} samples)")
        if df.empty:
            print(f"Skipping empty DataFrame for dataset: {name}")
            continue

        # Use task_type for stratification key determination
        stratify_key = None
        # Check if 'label' column exists and has variance for stratification
        if 'label' in df.columns and df['label'].nunique() > 1:
             stratify_key = df['label']
        else:
             print(f"Warning: Cannot stratify for {name}, 'label' column missing, empty, or has only one unique value.")

        is_test_set = name.startswith("test_") or "test" in name.lower()

        if is_test_set:
            print(f"Creating TEST dataloader for: {name}")
            try:
                test_dataset = MultiTaskDataset(df, tokenizer, MAX_LENGTH)
                test_dataloaders[name] = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
            except Exception as e:
                print(f"ERROR creating TEST dataloader for {name}: {e}")
        else: # Process as training/validation data
            try:
                # Perform train/val split
                train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=stratify_key)
            except ValueError as e:
                # Fallback if stratification fails (e.g., too few samples in a class)
                print(f"Warning: Could not stratify split for {name}. Using regular split. Error: {e}")
                train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

            # Create training DataLoader
            if not train_df.empty:
                 print(f"Creating TRAIN dataloader for: {name} ({len(train_df)} samples)")
                 try:
                     train_dataset = MultiTaskDataset(train_df, tokenizer, MAX_LENGTH)
                     train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
                     train_dataloaders.append(train_loader)
                 except Exception as e:
                     print(f"ERROR creating TRAIN dataloader for {name}: {e}")
            else: print(f"Skipping TRAIN dataloader creation for {name} - DataFrame empty after split.")

            # Create validation DataLoader
            if not val_df.empty:
                 print(f"Creating VALIDATION dataloader for: {name} ({len(val_df)} samples)")
                 try:
                     val_dataset = MultiTaskDataset(val_df, tokenizer, MAX_LENGTH)
                     val_dataloaders[name] = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
                 except Exception as e:
                     print(f"ERROR creating VALIDATION dataloader for {name}: {e}")
            else: print(f"Skipping VALIDATION dataloader creation for {name} - DataFrame empty after split.")

    print("\n--- Training Setup ---")
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
    model = MultiTaskModel(
        num_emotion_classes=NUM_EMOTION_CLASSES,
        dropout_rate=DROPOUT_RATE # Pass configured dropout rate
        ).to(DEVICE)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY # Add weight decay
        )
    # Scheduler reduces LR if validation loss plateaus
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=SCHEDULER_PATIENCE, # Use configured patience <<< TUNED
        verbose=True
        )

    # 4. Training Loop
    print("\n--- Starting Training ---")
    best_val_loss = float('inf')
    model_save_path = "multitask_model_best.pth"
    epochs_no_improve = 0 # Counter for early stopping patience (related to scheduler)

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")
        epoch_train_loss = 0.0
        num_valid_train_loaders = 0 # Count loaders that actually run
        print(f"Training on {len(train_dataloaders)} dataset loaders sequentially...")

        for i, train_loader in enumerate(train_dataloaders):
            loader_dataset_name = "Unknown" # Default
            try:
                # Peek at the first batch to get dataset name (requires non-empty loader)
                if len(train_loader) > 0:
                    # Use iter and next to safely get the first batch
                    batch_iterator = iter(train_loader)
                    temp_batch = next(batch_iterator)
                    # Ensure dataset_name exists and is not empty
                    if temp_batch and 'dataset_name' in temp_batch and temp_batch['dataset_name']:
                        loader_dataset_name = temp_batch['dataset_name'][0]
                    else:
                         loader_dataset_name = f"Loader_{i+1}_Unnamed"
                else:
                    print(f"Skipping empty train loader {i+1}.")
                    continue # Skip if loader is empty
            except StopIteration: # Handles case where loader is technically not empty but yields no batches
                 print(f"Train loader {i+1} yielded no batches. Skipping.")
                 continue
            except Exception as e:
                 loader_dataset_name = f"Loader_{i+1}"
                 print(f"Could not get dataset name for loader {i+1}, using generic name. Error: {e}")


            print(f"\nTraining on loader {i+1}/{len(train_dataloaders)} (Dataset: {loader_dataset_name})...")
            # Train one epoch on this specific dataloader
            loader_avg_loss = train_epoch(model, train_loader, optimizer, DEVICE, dataset_weights_map)
            print(f"Avg Loss for loader {i+1} ({loader_dataset_name}): {loader_avg_loss:.4f}")
            # Only accumulate loss if it's valid
            if not math.isnan(loader_avg_loss) and not math.isinf(loader_avg_loss):
                epoch_train_loss += loader_avg_loss
                num_valid_train_loaders += 1
            else:
                print(f"Warning: Invalid training loss ({loader_avg_loss}) for loader {i+1}. Excluding from epoch average.")


        # Calculate average training loss across all *valid* training dataloaders for the epoch
        avg_epoch_train_loss = epoch_train_loss / num_valid_train_loaders if num_valid_train_loaders > 0 else 0.0
        print(f"\nEpoch {epoch + 1} Average Training Loss (across {num_valid_train_loaders} valid loaders): {avg_epoch_train_loss:.4f}")

        # --- Validation Step ---
        epoch_val_loss = 0.0
        num_valid_val_loaders = 0 # Count loaders with valid loss
        print("\n--- Validation ---")
        if not val_dataloaders:
            print("No validation datasets found. Skipping validation step.")
            # If no validation, consider saving model based on training loss or every epoch
        else:
            # Evaluate on each validation dataloader
            for name, val_loader in val_dataloaders.items():
                if len(val_loader) == 0:
                     print(f"Skipping empty validation loader: {name}")
                     continue

                print(f"Evaluating on validation set: {name}")
                val_loss, accuracy, f1, precision, recall = evaluate(model, val_loader, DEVICE, dataset_weights_map)
                print(f"  - Val Loss: {val_loss:.4f}, Acc: {accuracy:.4f}, F1: {f1:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}")
                # Ensure val_loss is valid before adding
                if not math.isnan(val_loss) and not math.isinf(val_loss):
                    epoch_val_loss += val_loss
                    num_valid_val_loaders += 1
                else:
                    print(f"  - Warning: Invalid validation loss ({val_loss}) for {name}. Excluding from average.")


            # Calculate average validation loss across all *valid* validation datasets
            avg_epoch_val_loss = epoch_val_loss / num_valid_val_loaders if num_valid_val_loaders > 0 else float('inf') # Avoid division by zero
            print(f"\nEpoch {epoch + 1} Average Validation Loss (across {num_valid_val_loaders} valid loaders): {avg_epoch_val_loss:.4f}")

            # Only step scheduler and save model if avg_epoch_val_loss is valid
            if avg_epoch_val_loss != float('inf'):
                # Check for improvement before stepping scheduler
                if avg_epoch_val_loss < best_val_loss:
                    print(f"Validation loss improved ({best_val_loss:.4f} --> {avg_epoch_val_loss:.4f}). Saving model to {model_save_path}...")
                    try:
                        torch.save(model.state_dict(), model_save_path)
                        best_val_loss = avg_epoch_val_loss
                        epochs_no_improve = 0 # Reset counter
                    except Exception as e:
                        print(f"ERROR saving model: {e}")
                else:
                    epochs_no_improve += 1
                    print(f"Validation loss ({avg_epoch_val_loss:.4f}) did not improve from best ({best_val_loss:.4f}). Epochs without improvement: {epochs_no_improve}")

                # Step the scheduler *after* saving the model (if improved) or incrementing counter
                scheduler.step(avg_epoch_val_loss)


            else:
                print("Average validation loss is invalid or no valid validation loaders. Skipping scheduler step and model saving check.")


    print("\n--- Training Finished ---")

    # 5. Load Best Model state for final evaluation
    print(f"\nLoading best model state from {model_save_path}...")
    # Initialize a new model instance for loading the state dict
    final_model = MultiTaskModel(
        num_emotion_classes=NUM_EMOTION_CLASSES,
        dropout_rate=DROPOUT_RATE # Ensure loaded model has same architecture
        ).to(DEVICE)
    if os.path.exists(model_save_path):
        try:
            final_model.load_state_dict(torch.load(model_save_path, map_location=DEVICE)) # Ensure map_location for device compatibility
            print("Best model loaded successfully.")
        except Exception as e:
            print(f"Error loading best model state from {model_save_path}: {e}. Using model from end of training instead.")
            # If loading fails, final_model will retain the weights from the last training epoch
            # Re-assign the trained model if loading failed
            final_model = model
    else:
        print(f"Best model file ({model_save_path}) not found. Using model from end of training.")
        # Re-assign the trained model if file not found
        final_model = model

    final_model.eval() # Ensure the loaded model is in evaluation mode

    # 6. Final Test Set Evaluation
    print("\n--- Final Test Set Evaluation ---")
    if not test_dataloaders:
        print("No test dataloaders found. Skipping final test evaluation.")
    else:
        for name, test_loader in test_dataloaders.items():
            if len(test_loader) == 0:
                 print(f"Skipping empty test loader: {name}")
                 continue
            print(f"Evaluating on Test Set: {name}")
            # Evaluate using the loaded best model (or last model if loading failed)
            test_loss, accuracy, f1, precision, recall = evaluate(final_model, test_loader, DEVICE, dataset_weights_map)
            print(f"  - Test Loss: {test_loss:.4f}, Acc: {accuracy:.4f}, F1: {f1:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}")

   
    print("\n--- Script Finished ---")