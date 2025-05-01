# -*- coding: utf-8 -*-
"""
Single-Task RoBERTa for Disagreement Detection.

This script trains and evaluates a RoBERTa-based model specifically for
classifying text as expressing disagreement (binary). It is derived from
a multitask script and maintains consistent hyperparameters for comparison.

Key Features:
- Uses RoBERTa as the base transformer model.
- Handles multiple disagreement datasets (Hugging Face Hub and local CSVs).
- Implements specific preprocessing logic for datasets like FNC, BD, Politifact.
- Applies class weighting (pos_weight) to address imbalance in disagreement datasets.
- Includes evaluation on standard binary classification metrics (Accuracy, F1, Precision, Recall).
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

# === Data Preprocessing Function (Disagreement Only) ===

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
    df_final['dataset_name'] = dataset_name # Keep dataset name for weight mapping
    if df_final['label'].isnull().any():
        nan_count = df_final['label'].isnull().sum()
        print(f"WARNING: Found {nan_count} NaN labels in final dataframe for {dataset_name}. Filling with 0 (assuming non-disagreement).")
        df_final['label'].fillna(0, inplace=True)
    df_final['label'] = df_final['label'].astype(int)
    # Return weight for the positive class (disagreement = 1)
    return df_final, pos_weight

# === Data Loading Function (Disagreement Only) ===
def load_and_process_disagreement_data(data_dir="."):
    """
    Loads disagreement datasets from Hugging Face Hub and local CSV files.
    Preprocesses each dataset using the disagreement function.
    Collects preprocessed dataframes and pos_weights.
    """
    all_dataframes = {}
    # Stores pos_weight (single tensor value) per dataset
    dataset_weights_map = {}

    # --- Load Hugging Face Datasets ---
    print("\n--- Loading Hugging Face disagreement datasets ---")
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

    # --- Load Local CSV Datasets ---
    print(f"\n--- Loading local CSV disagreement datasets from directory: {data_dir} ---")
    loaded_csv_count = 0
    for filename in os.listdir(data_dir):
        if filename.endswith(".csv"):
            dataset_name = filename[:-4].lower().replace(" ", "_")
            # Simple heuristic: skip if 'emotion' is in the name
            if 'emotion' in dataset_name:
                print(f"Skipping CSV '{filename}', likely an emotion dataset.")
                continue
            if dataset_name in all_dataframes:
                print(f"Skipping CSV '{filename}', dataset '{dataset_name}' already processed.")
                continue

            filepath = os.path.join(data_dir, filename)
            print(f"Attempting to load CSV: {filename} as '{dataset_name}' (assuming disagreement)")
            try:
                df = pd.read_csv(filepath)
                processed_df, weight = preprocess_disagreement(df, dataset_name)

                if processed_df is not None:
                    all_dataframes[dataset_name] = processed_df
                    dataset_weights_map[dataset_name] = weight # Store pos_weight
                    loaded_csv_count += 1
                else: print(f"Skipped CSV '{filename}' due to preprocessing issues.")
            except Exception as e: print(f"Error loading or processing CSV '{filename}': {e}")
    print(f"Loaded {loaded_csv_count} local CSV datasets successfully.")

    # --- Final Check and Summary ---
    print("\n--- Disagreement Datasets Loaded Summary ---")
    final_datasets = {}
    final_weights = {} # Store weights only for the successfully loaded datasets
    for name, df in all_dataframes.items():
        if df is not None and not df.empty:
             required_cols = ['text', 'label', 'dataset_name']
             if all(col in df.columns for col in required_cols):
                 final_datasets[name] = df
                 weight_info = "N/A"
                 if name in dataset_weights_map and dataset_weights_map[name] is not None:
                     # Ensure it's a tensor before calling .item()
                     if isinstance(dataset_weights_map[name], torch.Tensor):
                         weight_info = f"Pos Weight: {dataset_weights_map[name].item():.4f}"
                     else:
                         weight_info = f"Pos Weight: {dataset_weights_map[name]:.4f}" # Assume float if not tensor
                     final_weights[name] = dataset_weights_map[name]
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

                 print(f"- {name}: {len(df)} samples, Label dist: {label_dist_str}, Weight Info: {weight_info}")
             else: print(f"WARNING: Skipping dataset '{name}' due to missing required columns after processing.")
        else: print(f"WARNING: Dataset '{name}' is None or empty after processing.")

    print("\n--- Final Pos_Weights Summary ---")
    if not final_weights: print("No weights stored for any datasets.")
    else:
        for name, weight in final_weights.items():
             weight_val = weight.item() if isinstance(weight, torch.Tensor) else weight
             print(f"- {name}: pos_weight = {weight_val:.4f}")

    if not final_datasets:
        print("\nFATAL: No disagreement datasets were loaded successfully. Exiting.")
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
            # Use iloc for integer-location based indexing
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

        # Convert label to tensor (float for BCEWithLogitsLoss)
        try:
            label_tensor = torch.tensor(label, dtype=torch.float)
        except Exception as e:
             raise ValueError(f"Error converting label '{label}' at index {idx}: {e}")

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label_tensor,
            'dataset_name': dataset_name # Pass dataset name for weight lookup
        }

# === Model Definition (Single Task - Disagreement) ===
class SingleTaskDisagreementModel(nn.Module):
    """
    Single-task RoBERTa model with a head for disagreement detection.
    """
    def __init__(self, model_name=MODEL_NAME, dropout_rate=DROPOUT_RATE):
        super(SingleTaskDisagreementModel, self).__init__()
        print(f"Initializing single-task disagreement model with base: {model_name}")
        self.roberta = RobertaModel.from_pretrained(model_name, output_attentions=True)
        hidden_size = self.roberta.config.hidden_size
        print(f"RoBERTa hidden size: {hidden_size}")

        # Disagreement head structure consistent with the multitask version
        self.disagreement_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate), # Use configured dropout rate
            nn.Linear(512, 1) # Output a single logit for binary classification
        )
        print(f"Initialized disagreement head (output: 1 logit, dropout: {dropout_rate})")

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model.
        """
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        # Use pooler output which represents the [CLS] token's features
        pooled_output = outputs.pooler_output
        # Pass through the disagreement head
        logits = self.disagreement_head(pooled_output)
        return logits, outputs.attentions

# === Training and Evaluation Functions (Adapted for Single Task) ===

def train_epoch(model, dataloader, optimizer, device, dataset_weights_map):
    """Trains the disagreement model for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    skipped_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        try:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            dataset_names = batch['dataset_name'] # Get dataset names for weight lookup
        except Exception as e:
            print(f"Error moving batch {batch_idx} to device. Skipping. Error: {e}")
            skipped_batches += 1
            continue

        # Basic check for batch homogeneity (should be less of an issue here, but good practice)
        if not dataset_names:
             print(f"Warning: Batch {batch_idx} has empty dataset_names. Skipping.")
             skipped_batches += 1
             continue
        current_dataset = dataset_names[0]
        if not all(d == current_dataset for d in dataset_names):
            print(f"Warning: Batch {batch_idx} mixed dataset ({dataset_names}). Using weight for first: {current_dataset}.")
            # Decide how to handle mixed batches if they occur (e.g., skip, use first weight)

        optimizer.zero_grad()

        try:
            logits, _ = model(input_ids, attention_mask)
        except Exception as e:
            print(f"Error in forward pass batch {batch_idx}. Skipping. Error: {e}")
            skipped_batches += 1
            continue

        try:
            # Fetch pos_weight (single value tensor)
            pos_weight = dataset_weights_map.get(current_dataset)
            if pos_weight is None:
                print(f"Warning: No pos_weight found for dataset '{current_dataset}' in batch {batch_idx}. Using weight 1.0.")
                pos_weight = torch.tensor(1.0, device=device)
            else:
                 # Ensure it's a tensor and on the correct device
                 if not isinstance(pos_weight, torch.Tensor):
                     pos_weight = torch.tensor(pos_weight, dtype=torch.float, device=device)
                 else:
                     pos_weight = pos_weight.to(device)

            # Define loss function with the specific weight
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            loss = loss_fn(logits.squeeze(-1), labels.float())

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
            optimizer.zero_grad() # Clear gradients if step failed

    if skipped_batches > 0:
        print(f"Skipped {skipped_batches} batches during training epoch.")
    return total_loss / num_batches if num_batches > 0 else 0.0

def evaluate(model, dataloader, device, dataset_weights_map):
    """Evaluates the disagreement model."""
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

            # Check dataset homogeneity for weight lookup
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
                # Fetch pos_weight
                pos_weight = dataset_weights_map.get(current_dataset)
                if pos_weight is None: pos_weight = torch.tensor(1.0, device=device)
                else:
                     # Ensure tensor and device
                     if not isinstance(pos_weight, torch.Tensor):
                          pos_weight = torch.tensor(pos_weight, dtype=torch.float, device=device)
                     else:
                          pos_weight = pos_weight.to(device)

                loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                loss = loss_fn(logits.squeeze(-1), labels.float())
                preds = (torch.sigmoid(logits).squeeze(-1) > 0.5).int()

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

    # --- Calculate Metrics (Binary) ---
    if not all_labels or not all_preds or len(all_labels) != len(all_preds):
         print("Warning: No valid labels/predictions collected during evaluation. Cannot calculate metrics.")
         return avg_loss, 0.0, 0.0, 0.0, 0.0 # Return zero for metrics

    try:
        # Use 'binary' average strategy for disagreement task
        average_strategy = 'binary'
        print(f"Using '{average_strategy}' averaging for metrics.")

        accuracy = accuracy_score(all_labels, all_preds)
        # Specify pos_label=1 if needed, though often inferred for binary
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

    # 1. Load and Preprocess Disagreement Data
    all_datasets_dict, dataset_weights_map = load_and_process_disagreement_data(data_dir=DATA_DIR)

    # 2. Create Datasets and DataLoaders
    train_dataloaders = [] # List to hold train loaders from different datasets
    val_dataloaders = {}   # Dict to hold validation loaders keyed by dataset name
    test_dataloaders = {}  # Dict to hold test loaders keyed by dataset name

    print("\n--- Creating PyTorch Datasets and DataLoaders for Disagreement Task ---")
    for name, df in all_datasets_dict.items():
        print(f"Processing dataset for DataLoader: {name} ({len(df)} samples)")
        if df.empty:
            print(f"Skipping empty DataFrame for dataset: {name}")
            continue

        # Use 'label' for stratification (binary disagreement)
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
                # Perform train/val split
                train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=stratify_key)
            except ValueError as e:
                print(f"Warning: Could not stratify split for {name}. Using regular split. Error: {e}")
                train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

            # Create training DataLoader
            if not train_df.empty:
                 print(f"Creating TRAIN dataloader for: {name} ({len(train_df)} samples)")
                 try:
                     train_dataset = SingleTaskDataset(train_df, tokenizer, MAX_LENGTH)
                     # Add this loader to the list of training loaders
                     train_dataloaders.append(DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True))
                 except Exception as e:
                     print(f"ERROR creating TRAIN dataloader for {name}: {e}")
            else: print(f"Skipping TRAIN dataloader creation for {name} - DataFrame empty after split.")

            # Create validation DataLoader
            if not val_df.empty:
                 print(f"Creating VALIDATION dataloader for: {name} ({len(val_df)} samples)")
                 try:
                     val_dataset = SingleTaskDataset(val_df, tokenizer, MAX_LENGTH)
                     # Store this loader in the dictionary
                     val_dataloaders[name] = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
                 except Exception as e:
                     print(f"ERROR creating VALIDATION dataloader for {name}: {e}")
            else: print(f"Skipping VALIDATION dataloader creation for {name} - DataFrame empty after split.")

    print("\n--- Training Setup (Disagreement Task) ---")
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
    model = SingleTaskDisagreementModel(
        dropout_rate=DROPOUT_RATE
        ).to(DEVICE)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
        )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min', # Monitor validation loss
        factor=0.1,
        patience=SCHEDULER_PATIENCE,
        verbose=True
        )

    # 4. Training Loop
    print("\n--- Starting Training (Disagreement Task) ---")
    best_val_loss = float('inf')
    model_save_path = "disagreement_model_best.pth" # Specific save path
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")
        epoch_train_loss = 0.0
        num_valid_train_loaders = 0
        print(f"Training on {len(train_dataloaders)} dataset loaders sequentially...")

        # Iterate through each training dataloader (one per original dataset)
        for i, train_loader in enumerate(train_dataloaders):
            loader_dataset_name = "Unknown"
            try:
                if len(train_loader.dataset) > 0: # Check dataset size directly
                    # Access dataset_name from the first item in the dataset
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
                if len(val_loader.dataset) == 0: # Check dataset size
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
    print(f"\nLoading best disagreement model state from {model_save_path}...")
    final_model = SingleTaskDisagreementModel(dropout_rate=DROPOUT_RATE).to(DEVICE)
    if os.path.exists(model_save_path):
        try:
            final_model.load_state_dict(torch.load(model_save_path, map_location=DEVICE))
            print("Best disagreement model loaded successfully.")
        except Exception as e:
            print(f"Error loading best model state from {model_save_path}: {e}. Using model from end of training instead.")
            final_model = model # Use the model from the end of training if loading fails
    else:
        print(f"Best model file ({model_save_path}) not found. Using model from end of training.")
        final_model = model # Use the model from the end of training if file not found

    final_model.eval()

    # 6. Final Test Set Evaluation
    print("\n--- Final Test Set Evaluation (Disagreement Task) ---")
    if not test_dataloaders:
        print("No test dataloaders found. Skipping final test evaluation.")
    else:
        for name, test_loader in test_dataloaders.items():
            if len(test_loader.dataset) == 0: # Check dataset size
                 print(f"Skipping empty test loader: {name}")
                 continue
            print(f"Evaluating on Test Set: {name}")
            test_loss, accuracy, f1, precision, recall = evaluate(final_model, test_loader, DEVICE, dataset_weights_map)
            print(f"  - Test Loss: {test_loss:.4f}, Acc: {accuracy:.4f}, F1: {f1:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}")

    print("\n--- Disagreement Script Finished ---")