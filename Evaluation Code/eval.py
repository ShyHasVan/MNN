# -*- coding: utf-8 -*-
"""
Multitask RoBERTa for Disagreement Detection and Emotion Recognition.

This script trains and evaluates a RoBERTa-based model capable of performing
two tasks simultaneously:
1.  **Disagreement Detection:** Classifying text as expressing disagreement (binary).
2.  **Emotion Recognition:** Identifying the primary emotion conveyed in text
    (multi-class, based on GoEmotions simplified).

Key Features:
- Uses RoBERTa as the base transformer model.
- Handles multiple datasets (Hugging Face Hub and local CSVs).
- Implements specific preprocessing logic for datasets like FNC, BD, Politifact.
- Applies class weighting to address imbalance in disagreement and emotion datasets.
- Includes evaluation on standard metrics (Accuracy, F1, Precision, Recall).
- Generates performance plots (confusion matrix, ROC curve, per-class metrics) during final evaluation.
- Tunable hyperparameters.
"""
# !pip install transformers datasets scikit-learn torch pandas numpy matplotlib seaborn
# (Uncomment the !pip install line if running in an environment like Google Colab)

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel
from datasets import load_dataset
# === Metrics and Plotting Imports ===
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
# ===================================
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import warnings
import math # For checking potential inf/nan weights/loss

# Suppress specific warnings for cleaner output (optional)
warnings.filterwarnings("ignore", category=UserWarning, message=".*DataFrameGroupBy.apply.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*is_sparse is deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, message="Glyph.*missing from current font")


# === Configuration ===
# --- Hyperparameters for Tuning ---
MODEL_NAME = "roberta-base"
MAX_LENGTH = 128
BATCH_SIZE = 32
EPOCHS = 10 # Tuned from 5
LEARNING_RATE = 1e-5
DROPOUT_RATE = 0.3
WEIGHT_DECAY = 0.01
SCHEDULER_PATIENCE = 3 # Tuned from 2
# --- Other Configurations ---
DATA_DIR = "."
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "multitask_model_best.pth" # Path to save the best model
print(f"Using device: {DEVICE}")

# === Tokenizer ===
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

# === Emotion Labels ===
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
disagreement_label_list = ['True/Agree', 'False/Disagree']

# === Data Preprocessing Functions ===

def preprocess_goemotions(df, dataset_name):
    """Preprocesses GoEmotions: determines primary label, calculates class weights."""
    print(f"Preprocessing GoEmotions dataset: {dataset_name}")
    df = df.copy()
    def get_primary(labels):
        if isinstance(labels, (list, np.ndarray)): return labels[0] if len(labels) > 0 else -1
        elif isinstance(labels, (int, np.integer)): return labels
        return -1
    df['primary_label_id'] = df['labels'].apply(get_primary)
    df = df[df['primary_label_id'] != -1].copy()
    if df.empty: print(f"WARNING: GoEmotions - No valid data remaining for {dataset_name}."); return None, None
    df['label'] = df['primary_label_id'].astype(int)
    class_weights = torch.ones(NUM_EMOTION_CLASSES, dtype=torch.float)
    unique_labels_np = np.array(sorted(df['label'].unique()))
    if len(unique_labels_np) > 1:
        try:
            weights_array = compute_class_weight('balanced', classes=unique_labels_np, y=df['label'].values)
            temp_weights = torch.ones(NUM_EMOTION_CLASSES, dtype=torch.float)
            for label_id, weight in zip(unique_labels_np, weights_array):
                if 0 <= label_id < NUM_EMOTION_CLASSES: temp_weights[label_id] = weight
            if not(torch.isinf(temp_weights).any() or torch.isnan(temp_weights).any() or (temp_weights <= 0).any()):
                class_weights = temp_weights
                print(f"INFO: Calculated balanced class weights for {dataset_name} (Emotion).")
            else: print(f"WARNING: Invalid weights calculated for GoEmotions {dataset_name}. Using default.")
        except Exception as e: print(f"WARNING: Could not compute emotion class weights for {dataset_name}: {e}. Using default.")
    else: print(f"WARNING: Only {len(unique_labels_np)} unique emotion class found. Using default weights.")
    df_final = df[['text', 'label']].copy()
    df_final['task_type'] = 'emotion'; df_final['dataset_name'] = dataset_name
    return df_final, class_weights

def preprocess_disagreement(df, dataset_name):
    """Preprocesses disagreement datasets: maps labels to binary, calculates pos_weight."""
    print(f"Preprocessing disagreement dataset: {dataset_name}")
    df = df.copy()
    pos_weight = torch.tensor(1.0, dtype=torch.float) # Default weight

    # Identify text column
    if 'text' not in df.columns:
        potential_text_cols = ['statement', 'headline', 'title', 'content', 'title2_en', 'Body']
        found_text_col = next((col for col in potential_text_cols if col in df.columns), None)
        if found_text_col: df.rename(columns={found_text_col: 'text'}, inplace=True)
        else: print(f"FATAL: No text column found for {dataset_name}."); return None, None
    df['text'] = df['text'].astype(str)

    # Identify label column
    potential_label_cols = ["label", "Stance", "bd_label", "hyperpartisan", "stance", "verdict", "claim_label", "orig_label"]
    original_label_col = next((col for col in potential_label_cols if col in df.columns), None)
    if original_label_col is None: print(f"FATAL: No label column found for {dataset_name}."); return None, None

    df[original_label_col] = df[original_label_col].astype(str)
    true_indicators = ['agree', 'true', '0', 'agreed'] # Words indicating the '0' class
    df['label'] = df[original_label_col].str.lower().apply(lambda x: 0 if x in true_indicators else 1)
    print(f"INFO: Applied heuristic mapping ({true_indicators} -> 0, others -> 1) for {dataset_name}.")

    if 'label' in df.columns:
        label_counts = df['label'].value_counts()
        # Check if both classes exist and positive count (label '1') is > 0
        if 0 in label_counts and 1 in label_counts and label_counts.get(1, 0) > 0:
            neg_count = label_counts.get(0, 0) # Count of label '0'
            pos_count = label_counts.get(1, 0) # Count of label '1'
            weight_val = neg_count / pos_count # weight = neg/pos
            if not(math.isinf(weight_val) or math.isnan(weight_val) or weight_val <= 0):
                 pos_weight = torch.tensor(weight_val, dtype=torch.float)
                 print(f"INFO: Calculated pos_weight for {dataset_name} (label 1): {pos_weight.item():.4f}")
            else: print(f"WARNING: Invalid pos_weight calculated ({weight_val}) for {dataset_name}. Using default 1.0.")
        else: print(f"WARNING: Could not calculate valid pos_weight for {dataset_name} (counts: {label_counts.to_dict()}). Using default 1.0.")

    df_final = df[['text', 'label']].copy()
    df_final['task_type'] = 'disagreement'; df_final['dataset_name'] = dataset_name
    df_final['label'].fillna(1, inplace=True) # Fill NaN with False/Disagree (label '1')
    df_final['label'] = df_final['label'].astype(int)
    return df_final, pos_weight


# === Data Loading Function ===
def load_and_process_all_data(data_dir="."):
    """Loads and preprocesses datasets from Hugging Face and local CSVs."""
    all_dataframes = {}
    dataset_weights_map = {}
    print("\n--- Loading Hugging Face datasets ---")
    # Simplified loading (add try-except blocks as needed from original)
    try:
        politifact_ds = load_dataset("liar", trust_remote_code=True)["train"].to_pandas()
        politifact_ds.rename(columns={"statement": "text", "label": "orig_label"}, inplace=True)
        processed_df, weight = preprocess_disagreement(politifact_ds, "politifact")
        if processed_df is not None: all_dataframes["politifact"] = processed_df; dataset_weights_map["politifact"] = weight
    except Exception as e: print(f"ERROR loading/processing Politifact: {e}")
    try:
        hyperpartisan_ds = load_dataset("hyperpartisan_news_detection", "byarticle", trust_remote_code=True)["train"].to_pandas()
        processed_df, weight = preprocess_disagreement(hyperpartisan_ds, "hyperpartisan")
        if processed_df is not None: all_dataframes["hyperpartisan"] = processed_df; dataset_weights_map["hyperpartisan"] = weight
    except Exception as e: print(f"ERROR loading/processing Hyperpartisan: {e}")
    try:
        goemotions_ds = load_dataset("go_emotions", "simplified", trust_remote_code=True)["train"].to_pandas()
        processed_df, weights_tensor = preprocess_goemotions(goemotions_ds, "goemotions")
        if processed_df is not None: all_dataframes["goemotions"] = processed_df; dataset_weights_map["goemotions"] = weights_tensor
    except Exception as e: print(f"ERROR loading/processing GoEmotions: {e}")

    print(f"\n--- Loading local CSV datasets from directory: {data_dir} ---")
    loaded_csv_count = 0
    for filename in os.listdir(data_dir):
        if filename.endswith(".csv"):
            dataset_name = filename[:-4].lower().replace(" ", "_").replace("-","_") # Sanitize name
            if dataset_name in all_dataframes: continue # Skip if already loaded
            filepath = os.path.join(data_dir, filename)
            print(f"Attempting to load CSV: {filename} as '{dataset_name}'")
            try:
                df = pd.read_csv(filepath)
                task_type = 'unknown'; processed_df, weight = None, None
                # Basic heuristic for task type
                if 'emotion' in dataset_name or ('labels' in df.columns and df['labels'].dtype == 'object'):
                    task_type = 'emotion'
                    print(f"INFO: Assuming '{dataset_name}' is EMOTION.")
                    processed_df, weight = preprocess_goemotions(df, dataset_name)
                else:
                    task_type = 'disagreement'
                    print(f"INFO: Assuming '{dataset_name}' is DISAGREEMENT.")
                    processed_df, weight = preprocess_disagreement(df, dataset_name)

                if processed_df is not None:
                    all_dataframes[dataset_name] = processed_df
                    dataset_weights_map[dataset_name] = weight
                    loaded_csv_count += 1
            except Exception as e: print(f"Error loading/processing CSV '{filename}': {e}")
    print(f"Loaded {loaded_csv_count} local CSV datasets successfully.")

    # Final Check and Summary
    print("\n--- Datasets Loaded Summary ---")
    final_datasets, final_weights = {}, {}
    for name, df in all_dataframes.items():
        if df is not None and not df.empty and all(col in df.columns for col in ['text', 'label', 'task_type', 'dataset_name']):
             final_datasets[name] = df
             if name in dataset_weights_map: final_weights[name] = dataset_weights_map[name]
             task = df['task_type'].iloc[0]
             label_dist_str = str(df['label'].value_counts().to_dict())
             print(f"- {name}: {len(df)} samples, Task: {task}, Label dist: {label_dist_str}")
        else: print(f"WARNING: Skipping dataset '{name}' due to processing issues.")
    if not final_datasets: print("\nFATAL: No datasets loaded successfully."); exit()
    return final_datasets, final_weights

# === Custom Dataset Class ===
class MultiTaskDataset(Dataset):
    """PyTorch Dataset class to handle samples from different tasks."""
    def __init__(self, dataframe, tokenizer, max_length):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Add original index tracking if needed for disorder type analysis later
        self.dataframe = self.dataframe.reset_index(drop=True).reset_index().rename(columns={'index': 'original_index'})


    def __len__(self): return len(self.dataframe)
    def __getitem__(self, idx):
        item = self.dataframe.iloc[idx]
        text, label, task_type = str(item['text']), item['label'], item['task_type']
        encoding = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        input_ids, attention_mask = encoding['input_ids'].squeeze(0), encoding['attention_mask'].squeeze(0)
        label_tensor = torch.tensor(label, dtype=torch.float if task_type == 'disagreement' else torch.long)
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'label': label_tensor,
                'task_type': task_type, 'dataset_name': item['dataset_name'],
                'original_index': item['original_index']} # Return original index

# === Model Definition ===
class MultiTaskModel(nn.Module):
    """Multitask RoBERTa model with separate heads for disagreement and emotion tasks."""
    def __init__(self, model_name=MODEL_NAME, num_emotion_classes=NUM_EMOTION_CLASSES, dropout_rate=DROPOUT_RATE):
        super(MultiTaskModel, self).__init__()
        print(f"Initializing multitask model with base: {model_name}")
        self.roberta = RobertaModel.from_pretrained(model_name, output_attentions=False) # Attentions off for efficiency
        hidden_size = self.roberta.config.hidden_size
        print(f"RoBERTa hidden size: {hidden_size}")
        self.disagreement_head = nn.Sequential(nn.Linear(hidden_size, 512), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(512, 1))
        print(f"Initialized disagreement head (output: 1 logit, dropout: {dropout_rate})")
        self.emotion_head = nn.Sequential(nn.Linear(hidden_size, 512), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(512, num_emotion_classes))
        print(f"Initialized emotion head (output: {num_emotion_classes} logits, dropout: {dropout_rate})")

    def forward(self, input_ids, attention_mask, task_type):
        """Forward pass, routes input to the appropriate head."""
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        batch_task_type = task_type[0] # Assumes batch homogeneity
        if batch_task_type == 'disagreement': logits = self.disagreement_head(pooled_output)
        elif batch_task_type == 'emotion': logits = self.emotion_head(pooled_output)
        else: raise ValueError(f"Unknown task_type '{batch_task_type}'.")
        return logits # Return only logits

# === Training Function ===
def train_epoch(model, dataloader, optimizer, device, dataset_weights_map):
    """Trains the model for one epoch."""
    model.train()
    total_loss = 0.0; num_batches = 0; skipped_batches = 0
    for batch_idx, batch in enumerate(dataloader):
        try:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            task_type = batch['task_type']; dataset_names = batch['dataset_name']
        except Exception as e: print(f"Error moving train batch {batch_idx} to device: {e}"); skipped_batches += 1; continue
        if not task_type or not dataset_names: skipped_batches += 1; continue
        current_task, current_dataset = task_type[0], dataset_names[0]
        if not all(t == current_task for t in task_type) or not all(d == current_dataset for d in dataset_names):
             print(f"Warning: Train Batch {batch_idx} mixed types. Skipping."); skipped_batches += 1; continue

        optimizer.zero_grad()
        try: logits = model(input_ids, attention_mask, task_type)
        except Exception as e: print(f"Error train forward pass batch {batch_idx}: {e}"); skipped_batches += 1; continue

        loss = None
        try:
            if current_task == 'disagreement':
                pos_weight = dataset_weights_map.get(current_dataset)
                if pos_weight is None: pos_weight = torch.tensor(1.0, device=device)
                else:
                    if not isinstance(pos_weight, torch.Tensor): pos_weight = torch.tensor(pos_weight, dtype=torch.float, device=device)
                    else: pos_weight = pos_weight.to(device)
                loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                loss = loss_fn(logits.squeeze(-1), labels.float())
            elif current_task == 'emotion':
                class_weights = dataset_weights_map.get(current_dataset)
                loss_fn = None
                if class_weights is None: loss_fn = nn.CrossEntropyLoss()
                else:
                    if not isinstance(class_weights, torch.Tensor):
                        try: class_weights = torch.tensor(class_weights, dtype=torch.float, device=device)
                        except Exception: class_weights = None
                    else: class_weights = class_weights.to(device)
                    loss_fn = nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None else nn.CrossEntropyLoss()
                loss = loss_fn(logits, labels.long())
            else: skipped_batches += 1; continue # Unknown task

            if loss is not None and torch.isnan(loss):
                print(f"Warning: NaN loss detected train batch {batch_idx}. Skipping backward."); skipped_batches += 1; continue
            if loss is None: continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item(); num_batches += 1
        except Exception as e: print(f"Error train loss/backward/step batch {batch_idx}: {e}"); skipped_batches += 1; optimizer.zero_grad()

    if skipped_batches > 0: print(f"Skipped {skipped_batches} batches during training epoch.")
    return total_loss / num_batches if num_batches > 0 else 0.0


# === Evaluation Function (MODIFIED for plotting) ===
def evaluate(model, dataloader, device, dataset_weights_map, calculate_loss=True):
    """Evaluates the model and returns metrics AND raw predictions/labels/logits."""
    model.eval()
    all_preds = []
    all_labels = []
    all_disagreement_logits = []
    task_for_metrics = None
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            try:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels_cpu = batch['label'] # Keep on CPU
                labels_gpu = labels_cpu.to(device) # Move to GPU for loss calc if needed
                task_type = batch['task_type']
            except Exception as e: print(f"Error processing eval batch {batch_idx}: {e}"); continue

            if not task_type: continue
            current_task = task_type[0]
            if task_for_metrics is None: task_for_metrics = current_task
            if not all(t == current_task for t in task_type): continue

            try: logits = model(input_ids, attention_mask, task_type)
            except Exception as e: print(f"Error eval forward pass batch {batch_idx}: {e}"); continue

            all_labels.extend(labels_cpu.numpy()) # Store CPU labels

            loss = None
            if calculate_loss: # Only calculate loss if needed (e.g., during validation)
                try:
                    current_dataset = batch['dataset_name'][0] # Assumes batch homogeneity
                    if current_task == 'disagreement':
                        pos_weight = dataset_weights_map.get(current_dataset, torch.tensor(1.0, device=device))
                        if not isinstance(pos_weight, torch.Tensor): pos_weight = torch.tensor(pos_weight, dtype=torch.float, device=device)
                        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
                        loss = loss_fn(logits.squeeze(-1), labels_gpu.float())
                    elif current_task == 'emotion':
                        class_weights = dataset_weights_map.get(current_dataset)
                        loss_fn = None
                        if class_weights is None: loss_fn = nn.CrossEntropyLoss()
                        else:
                           if not isinstance(class_weights, torch.Tensor):
                               try: class_weights = torch.tensor(class_weights, dtype=torch.float, device=device)
                               except Exception: class_weights = None
                           loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device)) if class_weights is not None else nn.CrossEntropyLoss()
                        loss = loss_fn(logits, labels_gpu.long())

                    if loss is not None and not torch.isnan(loss):
                        total_loss += loss.item(); num_batches += 1
                    elif loss is not None: print(f"Warning: NaN loss eval batch {batch_idx}.")
                except Exception as e: print(f"Error calculating eval loss batch {batch_idx}: {e}")

            # Process predictions
            if current_task == 'disagreement':
                preds = (torch.sigmoid(logits).squeeze(-1) > 0.5).int().cpu().numpy()
                all_preds.extend(preds)
                all_disagreement_logits.extend(logits.squeeze(-1).cpu().numpy())
            elif current_task == 'emotion':
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    if not all_labels or not all_preds or len(all_labels) != len(all_preds):
        print("Warning: No valid labels/preds collected."); return avg_loss, 0.0, 0.0, 0.0, 0.0, [], [], [], task_for_metrics

    all_labels_np = np.array(all_labels); all_preds_np = np.array(all_preds)
    all_disagreement_logits_np = np.array(all_disagreement_logits) if all_disagreement_logits else None
    accuracy, f1, precision, recall = 0.0, 0.0, 0.0, 0.0
    try:
        average_strategy = 'weighted' if task_for_metrics == 'emotion' else 'binary'
        accuracy = accuracy_score(all_labels_np, all_preds_np)
        if task_for_metrics == 'disagreement' and len(np.unique(all_labels_np)) < 2:
             print("Warning: Only one class in labels. Using pos_label=1 for metrics.")
             f1 = f1_score(all_labels_np, all_preds_np, average=average_strategy, zero_division=0, pos_label=1)
             precision = precision_score(all_labels_np, all_preds_np, average=average_strategy, zero_division=0, pos_label=1)
             recall = recall_score(all_labels_np, all_preds_np, average=average_strategy, zero_division=0, pos_label=1)
        else:
             f1 = f1_score(all_labels_np, all_preds_np, average=average_strategy, zero_division=0)
             precision = precision_score(all_labels_np, all_preds_np, average=average_strategy, zero_division=0)
             recall = recall_score(all_labels_np, all_preds_np, average=average_strategy, zero_division=0)
    except Exception as e: print(f"Error calculating overall metrics: {e}")

    # Return loss, metrics, AND raw data
    return avg_loss, accuracy, f1, precision, recall, all_labels_np, all_preds_np, all_disagreement_logits_np, task_for_metrics


# === Plotting Functions (Copied from evaluate_model.py) ===

# 1. Per-Class Metrics for Emotion
def plot_emotion_metrics(labels, preds, class_names, dataset_name):
    """Calculates and plots per-class Precision, Recall, F1 for emotion."""
    # Check if there are any predictions to report on
    if len(labels) == 0 or len(preds) == 0:
        print(f"Skipping emotion metrics plot for {dataset_name}: No labels or predictions.")
        return
    # Ensure class_names covers all unique labels present
    unique_labels = np.unique(np.concatenate((labels, preds)))
    relevant_class_names = [class_names[i] for i in unique_labels if i < len(class_names)]
    if not relevant_class_names:
        print(f"Skipping emotion metrics plot for {dataset_name}: No relevant classes found.")
        return

    try:
        report = classification_report(labels, preds, target_names=relevant_class_names, labels=unique_labels, output_dict=True, zero_division=0)
        df = pd.DataFrame(report).transpose()
        # Keep only class rows, remove avg rows if present
        df = df.loc[[name for name in relevant_class_names if name in df.index]]

        if df.empty:
            print(f"Skipping emotion metrics plot for {dataset_name}: No valid class data in report.")
            return

        fig, axes = plt.subplots(1, 3, figsize=(18, max(6, len(relevant_class_names)*0.3)), sharey=True) # Adjust height dynamically
        fig.suptitle(f'Per-Class Emotion Metrics: {dataset_name}', fontsize=16)

        df['precision'].plot(kind='bar', ax=axes[0], title='Precision')
        df['recall'].plot(kind='bar', ax=axes[1], title='Recall')
        df['f1-score'].plot(kind='bar', ax=axes[2], title='F1-Score')

        axes[0].set_ylabel('Score')
        for ax in axes:
            ax.tick_params(axis='x', rotation=90)
            ax.grid(axis='y', linestyle='--')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = f"emotion_metrics_{dataset_name}.png"
        plt.savefig(save_path)
        print(f"Saved emotion metrics plot to {save_path}")
        plt.close(fig)
    except Exception as e:
        print(f"Error generating emotion metrics plot for {dataset_name}: {e}")
        # Attempt to close plot if error occurred after creation
        try: plt.close(fig)
        except: pass


# 2. Confusion Matrix
def plot_confusion_matrix(labels, preds, class_names, dataset_name, task_name):
    """Plots a confusion matrix heatmap."""
    if len(labels) == 0 or len(preds) == 0:
        print(f"Skipping confusion matrix for {dataset_name} ({task_name}): No labels or predictions.")
        return
    try:
        # Determine the set of labels actually present
        present_labels = np.unique(np.concatenate((labels, preds)))
        # Filter class_names to only those present
        relevant_class_names = [class_names[i] for i in present_labels if i < len(class_names)]
        # Calculate matrix using only present labels to avoid errors if some classes are missing
        cm = confusion_matrix(labels, preds, labels=present_labels)

        # Adjust figure size based on number of classes
        figsize = (10, 8) if len(relevant_class_names) > 5 else (6, 5)
        plt.figure(figsize=figsize)

        sns.heatmap(cm, annot=True, fmt=".0f", cmap="Blues",
                    xticklabels=relevant_class_names, yticklabels=relevant_class_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix - {task_name.capitalize()}: {dataset_name}')
        plt.xticks(rotation=90 if len(relevant_class_names) > 5 else 0)
        plt.yticks(rotation=0)
        plt.tight_layout()
        save_path = f"confusion_matrix_{task_name}_{dataset_name}.png"
        plt.savefig(save_path)
        print(f"Saved confusion matrix plot to {save_path}")
        plt.close()
    except Exception as e:
        print(f"Error generating confusion matrix for {dataset_name} ({task_name}): {e}")
        try: plt.close()
        except: pass

# 3. ROC Curve and AUC (for Disagreement Task)
def plot_roc_curve(labels, logits, dataset_name, pos_label=1):
    """Plots the ROC curve and shows AUC for binary disagreement."""
    if logits is None or len(logits) == 0 or len(labels) == 0:
        print(f"Skipping ROC curve for {dataset_name}: No disagreement logits/labels available.")
        return
    if len(np.unique(labels)) < 2:
         print(f"Skipping ROC curve for {dataset_name}: Only one class present in labels.")
         return
    try:
        # Ensure pos_label exists in labels
        if pos_label not in np.unique(labels):
             print(f"Skipping ROC curve for {dataset_name}: Positive label '{pos_label}' not found in true labels.")
             return

        fpr, tpr, thresholds = roc_curve(labels, logits, pos_label=pos_label)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Diagonal line
        plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - Disagreement: {dataset_name}')
        plt.legend(loc="lower right"); plt.grid(True); plt.tight_layout()
        save_path = f"roc_curve_disagreement_{dataset_name}.png"
        plt.savefig(save_path)
        print(f"Saved ROC curve plot to {save_path}")
        plt.close()
    except ValueError as ve: print(f"Error generating ROC curve for {dataset_name}: {ve}.")
    except Exception as e: print(f"Error generating ROC curve for {dataset_name}: {e}"); plt.close()


# 4. Performance Across Information Disorder Types (Placeholder)
def plot_disorder_type_performance(results_dict, dataset_name):
    """Plots performance metrics for different information disorder types."""
    # --- Requires modifications to the evaluation loop and labelled data ---
    if not results_dict:
        print(f"Skipping disorder type plot for {dataset_name}: No results provided.")
        print("NOTE: Requires data labelled by disorder type and modified evaluation logic.")
        return
    try:
        df = pd.DataFrame(results_dict).T
        metric_to_plot = 'f1' if 'f1' in df.columns else 'accuracy'
        if metric_to_plot not in df.columns: print(f"Skipping disorder plot: Metric missing."); return

        df[metric_to_plot].plot(kind='bar', figsize=(10, 6))
        plt.title(f'Performance ({metric_to_plot}) by Information Disorder Type: {dataset_name}')
        plt.xlabel('Disorder Type'); plt.ylabel(metric_to_plot.capitalize())
        plt.xticks(rotation=45, ha='right'); plt.grid(axis='y', linestyle='--'); plt.tight_layout()
        save_path = f"disorder_types_{metric_to_plot}_{dataset_name}.png"
        plt.savefig(save_path); print(f"Saved disorder type plot to {save_path}"); plt.close()
    except Exception as e: print(f"Error generating disorder type plot for {dataset_name}: {e}"); plt.close()


# ===========================================
# === Main Script Execution ===
# ===========================================
if __name__ == "__main__":

    # 1. Load and Preprocess Data
    all_datasets_dict, dataset_weights_map = load_and_process_all_data(data_dir=DATA_DIR)

    # 2. Create Datasets and DataLoaders
    train_dataloaders = []; val_dataloaders = {}; test_dataloaders = {}
    # Keep track of original test dataframes for potential disorder analysis later
    original_test_dfs = {}

    print("\n--- Creating PyTorch Datasets and DataLoaders ---")
    for name, df in all_datasets_dict.items():
        print(f"Processing dataset for DataLoader: {name} ({len(df)} samples)")
        if df.empty: continue

        stratify_key = df['label'] if 'label' in df.columns and df['label'].nunique() > 1 else None
        # Simple check if it's intended as a test set (adjust if needed)
        # Example: assumes files containing 'test' in name are test sets
        is_test_set = "test" in name.lower() or name in ["fnc_test", "bd_test"] # Add specific test set names

        if is_test_set:
            print(f"Creating TEST dataloader for: {name}")
            try:
                original_test_dfs[name] = df # Store original df
                test_dataset = MultiTaskDataset(df, tokenizer, MAX_LENGTH)
                test_dataloaders[name] = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
            except Exception as e: print(f"ERROR creating TEST dataloader for {name}: {e}")
        else: # Process as training/validation data
            try: train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=stratify_key)
            except ValueError as e: print(f"Warning: Stratify failed for {name}. Using regular split. Error: {e}"); train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

            if not train_df.empty:
                 try: train_dataset = MultiTaskDataset(train_df, tokenizer, MAX_LENGTH); train_dataloaders.append(DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True))
                 except Exception as e: print(f"ERROR creating TRAIN dataloader for {name}: {e}")
            if not val_df.empty:
                 try: val_dataset = MultiTaskDataset(val_df, tokenizer, MAX_LENGTH); val_dataloaders[name] = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
                 except Exception as e: print(f"ERROR creating VALIDATION dataloader for {name}: {e}")

    print("\n--- Training Setup ---"); print(f"Batch Size: {BATCH_SIZE}", f"Epochs: {EPOCHS}", f"LR: {LEARNING_RATE}", f"WD: {WEIGHT_DECAY}", f"Dropout: {DROPOUT_RATE}", f"Scheduler Patience: {SCHEDULER_PATIENCE}", sep="\n")
    print(f"Total training dataset loaders: {len(train_dataloaders)}"); print(f"Validation datasets: {list(val_dataloaders.keys())}"); print(f"Test datasets: {list(test_dataloaders.keys())}")
    if not train_dataloaders: print("\nFATAL: No training dataloaders created."); exit()

    # 3. Initialize Model, Optimizer, Scheduler
    model = MultiTaskModel(num_emotion_classes=NUM_EMOTION_CLASSES, dropout_rate=DROPOUT_RATE).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=SCHEDULER_PATIENCE, verbose=True)

    # 4. Training Loop
    print("\n--- Starting Training ---")
    best_val_loss = float('inf'); epochs_no_improve = 0
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")
        epoch_train_loss = 0.0; num_valid_train_loaders = 0
        print(f"Training on {len(train_dataloaders)} dataset loaders sequentially...")
        for i, train_loader in enumerate(train_dataloaders):
            loader_name = f"Loader_{i+1}"; temp_batch = next(iter(train_loader), None)
            if temp_batch and 'dataset_name' in temp_batch and temp_batch['dataset_name']: loader_name = temp_batch['dataset_name'][0]
            print(f"\nTraining on {loader_name}...")
            loader_avg_loss = train_epoch(model, train_loader, optimizer, DEVICE, dataset_weights_map)
            print(f"Avg Loss for {loader_name}: {loader_avg_loss:.4f}")
            if not math.isnan(loader_avg_loss) and not math.isinf(loader_avg_loss): epoch_train_loss += loader_avg_loss; num_valid_train_loaders += 1
            else: print(f"Warning: Invalid training loss for {loader_name}. Excluding from epoch average.")
        avg_epoch_train_loss = epoch_train_loss / num_valid_train_loaders if num_valid_train_loaders > 0 else 0.0
        print(f"\nEpoch {epoch + 1} Average Training Loss: {avg_epoch_train_loss:.4f}")

        # --- Validation Step ---
        epoch_val_loss = 0.0; num_valid_val_loaders = 0
        print("\n--- Validation ---")
        if not val_dataloaders: print("No validation datasets. Skipping.")
        else:
            for name, val_loader in val_dataloaders.items():
                if len(val_loader) == 0: continue
                print(f"Evaluating on validation set: {name}")
                # Only calculate loss for validation step scheduler, metrics optional
                val_loss, v_acc, v_f1, v_prec, v_rec, _, _, _, _ = evaluate(model, val_loader, DEVICE, dataset_weights_map, calculate_loss=True)
                print(f"  - Val Loss: {val_loss:.4f}, Acc: {v_acc:.4f}, F1: {v_f1:.4f}") # Print main metrics
                if not math.isnan(val_loss) and not math.isinf(val_loss): epoch_val_loss += val_loss; num_valid_val_loaders += 1
                else: print(f"  - Warning: Invalid validation loss for {name}.")
            avg_epoch_val_loss = epoch_val_loss / num_valid_val_loaders if num_valid_val_loaders > 0 else float('inf')
            print(f"\nEpoch {epoch + 1} Average Validation Loss: {avg_epoch_val_loss:.4f}")

            if avg_epoch_val_loss != float('inf'):
                if avg_epoch_val_loss < best_val_loss:
                    print(f"Val loss improved ({best_val_loss:.4f} --> {avg_epoch_val_loss:.4f}). Saving model to {MODEL_SAVE_PATH}...")
                    try: torch.save(model.state_dict(), MODEL_SAVE_PATH); best_val_loss = avg_epoch_val_loss; epochs_no_improve = 0
                    except Exception as e: print(f"ERROR saving model: {e}")
                else: epochs_no_improve += 1; print(f"Val loss did not improve. Epochs w/o improve: {epochs_no_improve}")
                scheduler.step(avg_epoch_val_loss)
                # Optional Early Stopping Check:
                # if epochs_no_improve >= SCHEDULER_PATIENCE * 2: # Example: Stop after 2x scheduler patience
                #     print("Early stopping triggered.")
                #     break
            else: print("Avg val loss invalid. Skipping scheduler step.")
        # End of Epoch Loop

    print("\n--- Training Finished ---")

    # 5. Load Best Model state for final evaluation
    print(f"\nLoading best model state from {MODEL_SAVE_PATH}...")
    # Re-initialize model structure before loading state_dict
    final_model = MultiTaskModel(num_emotion_classes=NUM_EMOTION_CLASSES, dropout_rate=DROPOUT_RATE).to(DEVICE)
    if os.path.exists(MODEL_SAVE_PATH):
        try: final_model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE)); print("Best model loaded successfully.")
        except Exception as e: print(f"Error loading best model state: {e}. Using model from end of training."); final_model = model # Fallback
    else: print(f"Best model file not found. Using model from end of training."); final_model = model # Fallback

    final_model.eval() # Ensure model is in eval mode

    # 6. Final Test Set Evaluation & Plotting
    print("\n--- Final Test Set Evaluation & Plotting ---")
    # Placeholder for disorder type results (needs population based on labelled data)
    disorder_results = {}

    if not test_dataloaders: print("No test dataloaders found. Skipping final test evaluation.")
    else:
        for name, test_loader in test_dataloaders.items():
            print(f"\n--- Evaluating Test Set: {name} ---")
            if len(test_loader) == 0: print("Skipping empty test loader."); continue

            # Run evaluation - returns metrics AND raw data, don't recalculate loss here
            test_loss, accuracy, f1, precision, recall, labels, preds, disagreement_logits, task_type = evaluate(final_model, test_loader, DEVICE, dataset_weights_map, calculate_loss=False)

            print(f"\nOverall Metrics for Test Set: {name} (Task: {task_type})")
            print(f"  - Accuracy:  {accuracy:.4f}")
            print(f"  - F1 Score:  {f1:.4f} ({'weighted' if task_type == 'emotion' else 'binary'} avg)")
            print(f"  - Precision: {precision:.4f} ({'weighted' if task_type == 'emotion' else 'binary'} avg)")
            print(f"  - Recall:    {recall:.4f} ({'weighted' if task_type == 'emotion' else 'binary'} avg)")

            # --- Generate Plots based on Task Type ---
            if not labels.size == 0:
                if task_type == 'emotion':
                    # 1. Plot Per-Class Emotion Metrics
                    plot_emotion_metrics(labels, preds, emotion_label_list, name)
                    # 2. Plot Confusion Matrix
                    plot_confusion_matrix(labels, preds, emotion_label_list, name, task_type)

                elif task_type == 'disagreement':
                     # 2. Plot Confusion Matrix (Binary)
                     # Pass the correct class names based on your preprocessing
                     plot_confusion_matrix(labels, preds, disagreement_label_list, name, task_type)
                     # 3. Plot ROC Curve
                     positive_label_for_roc = 1
                     plot_roc_curve(labels, disagreement_logits, name, pos_label=positive_label_for_roc)

                # 4. Plot Disorder Type Performance (Requires data and logic)
                # --- Placeholder Call ---
                # TODO: Populate disorder_results dictionary here based on analysis
                # of 'preds' mapped back to 'original_test_dfs[name]' which needs disorder labels.
                plot_disorder_type_performance(disorder_results.get(name, {}), name)
            else:
                print(f"Skipping plot generation for {name} due to lack of evaluation results.")

    print("\n--- Script Finished ---")