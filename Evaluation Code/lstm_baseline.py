# -*- coding: utf-8 -*-
"""
Baseline Multitask LSTM for Disagreement Detection and Emotion Recognition.

This script trains and evaluates an LSTM-based model as a baseline comparison
for the RoBERTa model in final.py. It performs the same two tasks:
1.  **Disagreement Detection:** Classifying text as expressing disagreement (binary).
2.  **Emotion Recognition:** Identifying the primary emotion conveyed in text
    (multi-class, based on GoEmotions simplified).

Key Features:
- Uses an LSTM layer instead of a transformer.
- Requires building a vocabulary from the training data.
- Uses nn.Embedding layer for word representations.
- Handles multiple datasets (Hugging Face Hub and local CSVs) similarly to final.py.
- Applies class weighting for loss functions.
- Includes evaluation on standard metrics (Accuracy, F1, Precision, Recall).
"""
# !pip install datasets scikit-learn torch pandas numpy nltk # Added nltk for basic tokenization
# (Uncomment the !pip install line if running in an environment like Google Colab)

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import nltk # Using nltk for simple word tokenization
from collections import Counter, defaultdict
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import warnings
import math
import re # For basic text cleaning

# Suppress specific warnings for cleaner output (optional)
warnings.filterwarnings("ignore", category=UserWarning, message=".*DataFrameGroupBy.apply.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*is_sparse is deprecated.*")


# === Configuration ===
# --- LSTM Specific Hyperparameters ---
VOCAB_SIZE = 10000         # Maximum size of the vocabulary
EMBEDDING_DIM = 100        # Dimension of word embeddings
HIDDEN_DIM = 128           # Dimension of LSTM hidden state
NUM_LSTM_LAYERS = 1        # Number of LSTM layers
BIDIRECTIONAL_LSTM = True  # Whether to use a bidirectional LSTM
# --- General Hyperparameters (align with RoBERTa script where applicable) ---
MAX_LENGTH = 128           # Max sequence length (in words/tokens for LSTM)
BATCH_SIZE = 64            # Batch size (might need adjustment vs RoBERTa)
EPOCHS = 10                # Number of training epochs
LEARNING_RATE = 1e-3       # Learning rate (often higher for LSTMs than transformers)
DROPOUT_RATE = 0.4         # Dropout rate for regularization
WEIGHT_DECAY = 1e-4        # Weight decay for AdamW optimizer
SCHEDULER_PATIENCE = 3     # Patience for ReduceLROnPlateau scheduler
# --- Other Configurations ---
DATA_DIR = "."             # Directory to load local CSV datasets from
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use GPU if available
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
print(f"Using device: {DEVICE}")

# === NLTK Setup ===
# Ensure necessary NLTK resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
    print("NLTK 'punkt' resource found.")
except LookupError:
    print("NLTK 'punkt' resource not found. Downloading...")
    nltk.download('punkt', quiet=True)

# === Emotion Labels (Same as RoBERTa script) ===
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

# === Text Cleaning and Tokenization ===
def clean_text(text):
    """Basic text cleaning."""
    text = str(text).lower() # Lowercase
    text = re.sub(r'[^a-z0-9\s]', '', text) # Remove punctuation except spaces
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra whitespace
    return text

def tokenize(text):
    """Tokenizes text using NLTK word_tokenize."""
    return nltk.word_tokenize(text)

# === Vocabulary Class ===
class Vocabulary:
    """Manages the mapping between words and indices."""
    def __init__(self, freq_threshold=2):
        # Initialize with padding and unknown tokens
        self.itos = {0: PAD_TOKEN, 1: UNK_TOKEN} # index to string
        self.stoi = {PAD_TOKEN: 0, UNK_TOKEN: 1} # string to index
        self.freq_threshold = freq_threshold
        self.word_freq = Counter()

    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, sentence_list, max_size=VOCAB_SIZE):
        """Builds the vocabulary from a list of tokenized sentences."""
        print("Building vocabulary...")
        # Count word frequencies
        for sentence_tokens in sentence_list:
            self.word_freq.update(sentence_tokens)

        # Sort words by frequency (most common first) and filter by threshold
        most_common_words = [
            word for word, freq in self.word_freq.most_common()
            if freq >= self.freq_threshold
        ]

        # Limit vocabulary size, keeping most frequent words
        limited_words = most_common_words[:max_size - len(self.itos)] # Adjust for existing PAD/UNK

        # Add words to stoi and itos
        idx = len(self.itos) # Start index after PAD and UNK
        for word in limited_words:
            if word not in self.stoi:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1
        print(f"Vocabulary built with {len(self.itos)} unique tokens (frequency >= {self.freq_threshold}, max size {max_size}).")

    def numericalize(self, text_tokens):
        """Converts a list of tokens into a list of corresponding indices."""
        return [self.stoi.get(token, self.stoi[UNK_TOKEN]) for token in text_tokens]

# === Data Preprocessing Functions (Adapted from final.py) ===
# These functions are largely the same, focusing on label mapping and weight calculation.
# The main difference is that they now operate on the assumption that the 'text'
# column will be tokenized later by the Dataset class.

def preprocess_goemotions(df, dataset_name):
    """
    Preprocesses the GoEmotions DataFrame for emotion recognition.
    Assigns a single primary label, encodes it, and calculates class weights.
    (Largely identical to final.py version, just ensures 'text' column exists)
    """
    print(f"Preprocessing GoEmotions dataset: {dataset_name}")
    df = df.copy() # Work on a copy

    # Ensure 'text' column exists and is string
    if 'text' not in df.columns:
        print(f"FATAL: 'text' column not found in GoEmotions {dataset_name}. Skipping.")
        return None, None
    df['text'] = df['text'].astype(str)

    # Determine the primary emotion label (using the first label if multiple exist in simplified)
    def get_primary(labels):
        if isinstance(labels, list) or isinstance(labels, np.ndarray):
            return labels[0] if len(labels) > 0 else -1
        elif isinstance(labels, (int, np.integer)):
            return labels
        return -1

    df['primary_label_id'] = df['labels'].apply(get_primary)
    initial_count = len(df)
    df = df[df['primary_label_id'] != -1].copy()
    filtered_count = len(df)
    if initial_count != filtered_count:
        print(f"INFO: GoEmotions - Filtered out {initial_count - filtered_count} rows with no primary label.")
    if filtered_count == 0:
        print(f"WARNING: GoEmotions - No valid data remaining after filtering for {dataset_name}.")
        return None, None

    df['label'] = df['primary_label_id']

    # Calculate Class Weights for CrossEntropyLoss
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
                    print(f"Warning: Label ID {label_id} out of range during weight calc.")

            class_weights = class_weights_full
            if torch.isinf(class_weights).any() or torch.isnan(class_weights).any() or (class_weights <= 0).any():
                 print(f"WARNING: Invalid weights calculated for GoEmotions {dataset_name}. Using default weights (1.0).")
                 class_weights = torch.ones(NUM_EMOTION_CLASSES, dtype=torch.float)
            else:
                 print(f"INFO: Calculated balanced class weights for {dataset_name} (Emotion). Shape: {class_weights.shape}")
        except ValueError as e:
             print(f"WARNING: Could not compute class weights for GoEmotions {dataset_name}: {e}. Using default weights (1.0).")
             class_weights = torch.ones(NUM_EMOTION_CLASSES, dtype=torch.float)
    else:
        num_unique = len(unique_labels_np)
        label_info = unique_labels_np[0] if num_unique == 1 else 'None'
        print(f"WARNING: Only {num_unique} unique class ({label_info}) found in GoEmotions {dataset_name}. Using default weights (1.0).")
        class_weights = torch.ones(NUM_EMOTION_CLASSES, dtype=torch.float)

    # Select necessary columns and add task/dataset identifiers
    df_final = df[['text', 'label']].copy()
    df_final['task_type'] = 'emotion'
    df_final['dataset_name'] = dataset_name
    df_final['label'] = df_final['label'].astype(int)
    return df_final, class_weights

def preprocess_disagreement(df, dataset_name):
    """
    Preprocesses various disagreement/stance/fact-checking datasets.
    - Identifies text and label columns.
    - Maps diverse original labels to binary disagreement (0: No/Agree, 1: Yes/Disagree/False).
    - Calculates class weight (`pos_weight`) for BCEWithLogitsLoss.
    (Largely identical to final.py version, just ensures 'text' column exists)
    """
    print(f"Preprocessing disagreement dataset: {dataset_name}")
    df = df.copy()
    original_label_col = None
    pos_weight = None

    # Identify and ensure text column exists and is string
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
            print(f"FATAL: Could not find a suitable text column for {dataset_name}. Skipping.")
            return None, None
    df['text'] = df['text'].astype(str)

    # Identify label column (same logic as final.py)
    if dataset_name == "politifact":
        if 'orig_label' in df.columns: original_label_col = 'orig_label'
        elif 'label' in df.columns: original_label_col = 'label'
        else:
             potential_label_cols_pf = ["verdict", "claim_label"]
             for col in potential_label_cols_pf:
                 if col in df.columns: original_label_col = col; break
    else:
        potential_label_cols_other = ["label", "Stance", "bd_label", "hyperpartisan", "stance"]
        for col in potential_label_cols_other:
            if col in df.columns: original_label_col = col; break

    if original_label_col is None:
        print(f"Warning: Could not find suitable label column for {dataset_name}. Skipping.")
        return None, None

    # Map original labels to binary disagreement (same logic as final.py)
    df[original_label_col] = df[original_label_col].astype(str)
    if dataset_name == "politifact":
        try:
            numeric_labels = pd.to_numeric(df[original_label_col], errors='coerce').fillna(-1)
            false_ish_codes = [0, 4, 5, -1] # Liar codes for false/pants-fire/barely-true
            df['label'] = numeric_labels.apply(lambda x: 1 if x in false_ish_codes else 0)
            print(f"INFO: Applied Politifact numeric mapping (codes {false_ish_codes} -> 1) for {dataset_name}.")
        except Exception as e:
             print(f"ERROR applying Politifact mapping for {dataset_name}. Defaulting to 0. Error: {e}")
             df['label'] = 0
    elif dataset_name == "hyperpartisan":
        df['label'] = df[original_label_col].apply(lambda x: 1 if str(x).lower() == 'true' else 0)
        print(f"INFO: Applied Hyperpartisan mapping ('true' -> 1) for {dataset_name}.")
    elif "fnc" in dataset_name.lower():
        negative_labels_fnc = ['agree']
        df['label'] = df[original_label_col].str.lower().apply(lambda x: 0 if x in negative_labels_fnc else 1)
        print(f"INFO: Applied FNC mapping ('agree' -> 0, others -> 1) for {dataset_name}.")
    elif "bd" in dataset_name.lower():
         negative_labels_bd = ['agreed']
         df['label'] = df[original_label_col].str.lower().apply(lambda x: 0 if x in negative_labels_bd else 1)
         print(f"INFO: Applied BD mapping ('agreed' -> 0, 'disagreed' -> 1) for {dataset_name}.")
    else: # Generic fallback (same as final.py)
        print(f"WARNING: Applying GENERIC FALLBACK mapping for {dataset_name}.")
        try:
            temp_labels = pd.to_numeric(df[original_label_col], errors='coerce')
            if temp_labels.notna().all():
                 df['label'] = temp_labels.apply(lambda x: 0 if x == 0 else 1)
                 print(f"INFO: Applied numeric fallback mapping (0 -> 0, non-zero -> 1) for {dataset_name}.")
            else:
                 str_labels = df[original_label_col].astype(str).str.lower()
                 negative_indicators = ['true', 'agree', '1', 'yes', 'support', 'agreed']
                 df['label'] = str_labels.apply(lambda x: 0 if any(indicator in x for indicator in negative_indicators) else 1)
                 print(f"INFO: Applied string indicator fallback mapping (agreement words -> 0, others -> 1) for {dataset_name}.")
        except Exception as e:
            print(f"Error applying generic mapping for {dataset_name}. Defaulting to 0. Error: {e}")
            df['label'] = 0

    # Calculate Class Weights for BCEWithLogitsLoss (same logic as final.py)
    if 'label' in df.columns:
        label_counts = df['label'].value_counts()
        if 0 in label_counts and 1 in label_counts and label_counts.get(1, 0) > 0:
            neg_count = label_counts.get(0, 0)
            pos_count = label_counts.get(1, 0)
            weight_val = neg_count / pos_count
            if math.isinf(weight_val) or math.isnan(weight_val) or weight_val <= 0:
                 print(f"WARNING: Invalid weight calculated ({weight_val}) for {dataset_name}. Using default 1.0.")
                 pos_weight = torch.tensor(1.0, dtype=torch.float)
            else:
                 pos_weight = torch.tensor(weight_val, dtype=torch.float)
                 print(f"INFO: Calculated pos_weight for {dataset_name}: {pos_weight.item():.4f}")
        else:
            pos_weight = torch.tensor(1.0, dtype=torch.float)
            print(f"WARNING: Could not calculate valid pos_weight for {dataset_name}. Using default 1.0.")
    else:
        print(f"WARNING: 'label' column not found before weight calculation for {dataset_name}. Using default 1.0.")
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
        print(f"WARNING: Found {nan_count} NaN labels in final dataframe for {dataset_name}. Filling with 0.")
        df_final['label'].fillna(0, inplace=True)
    df_final['label'] = df_final['label'].astype(int)
    return df_final, pos_weight

# === Data Loading Function (Adapted from final.py) ===
def load_and_process_all_data(data_dir="."):
    """
    Loads datasets from Hugging Face Hub and local CSV files.
    Preprocesses each dataset using the appropriate function.
    Collects preprocessed dataframes and class weights.
    (Largely identical to final.py, ensures text cleaning happens before vocab build)
    """
    all_dataframes = {}
    dataset_weights_map = {}

    # --- Load Hugging Face Datasets ---
    print("\n--- Loading Hugging Face datasets ---")
    # Politifact
    try:
        print("Loading Politifact (liar)...")
        politifact_ds = load_dataset("liar", trust_remote_code=True)["train"].to_pandas()
        politifact_ds.rename(columns={"statement": "text", "label": "orig_label"}, inplace=True)
        processed_df, weight = preprocess_disagreement(politifact_ds, "politifact")
        if processed_df is not None:
            all_dataframes["politifact"] = processed_df
            dataset_weights_map["politifact"] = weight
    except Exception as e: print(f"ERROR loading/processing Politifact: {e}")

    # Hyperpartisan News Detection
    try:
        print("Loading Hyperpartisan News Detection...")
        hyperpartisan_ds = load_dataset("hyperpartisan_news_detection", "byarticle", trust_remote_code=True)["train"].to_pandas()
        processed_df, weight = preprocess_disagreement(hyperpartisan_ds, "hyperpartisan")
        if processed_df is not None:
            all_dataframes["hyperpartisan"] = processed_df
            dataset_weights_map["hyperpartisan"] = weight
    except Exception as e: print(f"ERROR loading/processing Hyperpartisan: {e}")

    # GoEmotions
    try:
        print("Loading GoEmotions...")
        goemotions_ds = load_dataset("go_emotions", "simplified", trust_remote_code=True)["train"].to_pandas()
        processed_df, weights_tensor = preprocess_goemotions(goemotions_ds, "goemotions")
        if processed_df is not None:
            all_dataframes["goemotions"] = processed_df
            dataset_weights_map["goemotions"] = weights_tensor
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
                task_type = 'unknown'
                if 'emotion' in dataset_name:
                    task_type = 'emotion'
                    print(f"INFO: Assuming '{dataset_name}' is an EMOTION dataset.")
                    processed_df, weight = preprocess_goemotions(df, dataset_name)
                else:
                    task_type = 'disagreement'
                    print(f"INFO: Assuming '{dataset_name}' is a DISAGREEMENT dataset.")
                    processed_df, weight = preprocess_disagreement(df, dataset_name)

                if processed_df is not None:
                    all_dataframes[dataset_name] = processed_df
                    dataset_weights_map[dataset_name] = weight
                    loaded_csv_count += 1
                else: print(f"Skipped CSV '{filename}' due to preprocessing issues.")
            except Exception as e: print(f"Error loading or processing CSV '{filename}': {e}")
    print(f"Loaded {loaded_csv_count} local CSV datasets successfully.")

    # --- Clean Text and Final Summary ---
    print("\n--- Cleaning Text and Final Summary ---")
    final_datasets = {}
    final_weights = {}
    all_training_texts = [] # Collect texts for vocabulary building

    for name, df in all_dataframes.items():
        if df is not None and not df.empty:
             required_cols = ['text', 'label', 'task_type', 'dataset_name']
             if all(col in df.columns for col in required_cols):
                 # Apply basic text cleaning BEFORE building vocabulary
                 print(f"Cleaning text for {name}...")
                 df['cleaned_text'] = df['text'].apply(clean_text)

                 # Collect cleaned text for vocab building (only from non-test sets)
                 is_test_set = name.startswith("test_") or "test" in name.lower()
                 if not is_test_set:
                     all_training_texts.extend(df['cleaned_text'].tolist())

                 final_datasets[name] = df
                 final_weights[name] = dataset_weights_map.get(name) # Get weight

                 task = df['task_type'].iloc[0]
                 label_dist_str = str(df['label'].value_counts().to_dict())
                 print(f"- {name}: {len(df)} samples, Task: {task}, Label dist: {label_dist_str}")
             else: print(f"WARNING: Skipping dataset '{name}' due to missing required columns.")
        else: print(f"WARNING: Dataset '{name}' is None or empty.")

    if not final_datasets:
        print("\nFATAL: No datasets were loaded successfully. Exiting.")
        exit()

    # --- Build Vocabulary ---
    # Tokenize all collected training texts
    print("Tokenizing training texts for vocabulary...")
    tokenized_training_texts = [tokenize(text) for text in all_training_texts]
    # Initialize and build vocabulary
    vocab = Vocabulary(freq_threshold=2)
    vocab.build_vocabulary(tokenized_training_texts, max_size=VOCAB_SIZE)

    return final_datasets, final_weights, vocab

# === Custom Dataset Class for LSTM ===
class LstmMultiTaskDataset(Dataset):
    """
    PyTorch Dataset class for the LSTM model.
    Handles tokenization, numericalization, and padding.
    """
    def __init__(self, dataframe, vocab, max_length=MAX_LENGTH):
        self.dataframe = dataframe
        self.vocab = vocab
        self.max_length = max_length
        required_cols = ['cleaned_text', 'label', 'task_type', 'dataset_name']
        if not all(col in dataframe.columns for col in required_cols):
             missing = [col for col in required_cols if col not in dataframe.columns]
             raise ValueError(f"Input DataFrame must contain required columns. Missing: {missing}")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        item = self.dataframe.iloc[idx]
        text = item['cleaned_text']
        label = item['label']
        task_type = item['task_type']
        dataset_name = item['dataset_name']

        # Tokenize and numericalize
        tokens = tokenize(text)
        numericalized_tokens = self.vocab.numericalize(tokens)

        # Truncate if necessary
        if len(numericalized_tokens) > self.max_length:
            numericalized_tokens = numericalized_tokens[:self.max_length]

        # Store original length before padding (useful for PackedSequence if needed, though not used here)
        original_length = len(numericalized_tokens)

        # Convert to tensor
        sequence_tensor = torch.tensor(numericalized_tokens, dtype=torch.long)

        # Convert label to tensor based on task type
        if task_type == 'disagreement':
            label_tensor = torch.tensor(label, dtype=torch.float)
        elif task_type == 'emotion':
            label_tensor = torch.tensor(label, dtype=torch.long)
        else:
            raise ValueError(f"Unknown task_type '{task_type}' encountered.")

        return {
            'sequence': sequence_tensor,
            'label': label_tensor,
            'task_type': task_type,
            'dataset_name': dataset_name,
            'length': original_length # Store original length
        }

# === Custom Collate Function for Padding ===
def collate_batch(batch):
    """Collates data samples into batches with padding."""
    sequences = [item['sequence'] for item in batch]
    labels = torch.stack([item['label'] for item in batch]) # Stack labels directly
    task_types = [item['task_type'] for item in batch]
    dataset_names = [item['dataset_name'] for item in batch]
    lengths = torch.tensor([item['length'] for item in batch], dtype=torch.long) # Get lengths

    # Pad sequences
    # batch_first=True means output shape is (batch_size, seq_len)
    padded_sequences = pad_sequence(
        sequences,
        batch_first=True,
        padding_value=vocab.stoi[PAD_TOKEN] # Use PAD token index for padding
    )

    # Ensure batch homogeneity (same task and dataset within a batch)
    if len(set(task_types)) > 1 or len(set(dataset_names)) > 1:
        # This should ideally not happen if dataloaders are created per dataset
        # but good to have a check. Handle or raise error as needed.
        print(f"Warning: Mixed task/dataset in collate_batch. Tasks: {set(task_types)}, Datasets: {set(dataset_names)}")

    return {
        'input_ids': padded_sequences, # Renamed to 'input_ids' for consistency
        'attention_mask': (padded_sequences != vocab.stoi[PAD_TOKEN]).long(), # Create mask based on padding
        'label': labels,
        'task_type': task_types, # Keep as list for checking in train/eval
        'dataset_name': dataset_names, # Keep as list
        'lengths': lengths # Include lengths if needed for PackedSequence later
    }


# === LSTM Model Definition ===
class LstmMultiTaskModel(nn.Module):
    """
    Multitask LSTM model with separate heads for disagreement and emotion tasks.
    """
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 hidden_dim,
                 num_emotion_classes,
                 num_layers=NUM_LSTM_LAYERS,
                 bidirectional=BIDIRECTIONAL_LSTM,
                 dropout=DROPOUT_RATE,
                 pad_idx=0): # Assuming PAD index is 0
        super(LstmMultiTaskModel, self).__init__()
        print(f"Initializing LSTM multitask model:")
        print(f"  Vocab Size: {vocab_size}, Embedding Dim: {embedding_dim}")
        print(f"  LSTM Hidden Dim: {hidden_dim}, Layers: {num_layers}, Bidirectional: {bidirectional}")
        print(f"  Dropout: {dropout}, Emotion Classes: {num_emotion_classes}")

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=num_layers,
                            bidirectional=bidirectional,
                            dropout=dropout if num_layers > 1 else 0, # Dropout only between LSTM layers
                            batch_first=True)

        # Calculate the input size for the linear layers
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.disagreement_head = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim // 2), # Intermediate layer
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1) # Output 1 logit
        )
        print(f"  Initialized disagreement head (Input: {lstm_output_dim}, Output: 1 logit)")

        self.emotion_head = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim // 2), # Intermediate layer
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_emotion_classes) # Output N logits
        )
        print(f"  Initialized emotion head (Input: {lstm_output_dim}, Output: {num_emotion_classes} logits)")

        self.dropout = nn.Dropout(dropout) # Apply dropout after embedding/LSTM

    def forward(self, input_ids, attention_mask, task_type, lengths=None):
        """
        Forward pass of the LSTM model.
        Args:
            input_ids (Tensor): Batch of sequences (batch_size, seq_len).
            attention_mask (Tensor): Mask indicating non-padding tokens (unused by basic LSTM).
            task_type (list[str]): List containing the task type for the batch.
            lengths (Tensor, optional): Original sequence lengths before padding.
                                        Needed for PackedSequence, but not used in this simple version.
        Returns:
            Tensor: Logits for the specified task.
        """
        # 1. Embedding
        # input_ids shape: (batch_size, seq_len)
        embedded = self.dropout(self.embedding(input_ids))
        # embedded shape: (batch_size, seq_len, embedding_dim)

        # 2. LSTM
        # Note: Using PackedSequence can improve efficiency by skipping padding,
        # but requires lengths and careful handling. For simplicity, we process padded sequences.
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # 3. Select LSTM Output for Classification
        # Using the final hidden state of the last layer:
        if self.lstm.bidirectional:
            # Concatenate the final forward and backward hidden states
            final_hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            # hidden shape is (num_layers, batch, hidden_dim)
            final_hidden = hidden[-1, :, :]

        pooled_output = self.dropout(final_hidden) # Apply dropout before heads

        # 4. Task-Specific Heads
        if not task_type:
             raise ValueError("task_type list cannot be empty during forward pass.")
        batch_task_type = task_type[0] # Assumes batch homogeneity

        if batch_task_type == 'disagreement':
            logits = self.disagreement_head(pooled_output)
        elif batch_task_type == 'emotion':
            logits = self.emotion_head(pooled_output)
        else:
            raise ValueError(f"Unknown task_type '{batch_task_type}' encountered.")

        return logits, None # Return None for attention compatibility


# === Training and Evaluation Functions (Adapted from final.py) ===
# These functions are very similar, just using the LSTM model and loss functions.

def train_epoch(model, dataloader, optimizer, device, dataset_weights_map):
    """Trains the LSTM model for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    skipped_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        try:
            # Note: 'input_ids' now refers to numericalized sequences
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device) # Mask might not be used by LSTM itself
            labels = batch['label'].to(device)
            task_type = batch['task_type']
            dataset_names = batch['dataset_name']
            # lengths = batch['lengths'].to(device) # If using PackedSequence
        except Exception as e:
            print(f"Error moving batch {batch_idx} to device. Skipping. Error: {e}")
            skipped_batches += 1
            continue

        if not task_type or not dataset_names:
             print(f"Warning: Batch {batch_idx} has empty task/dataset names. Skipping.")
             skipped_batches += 1
             continue
        current_task = task_type[0]
        current_dataset = dataset_names[0]
        if not all(t == current_task for t in task_type) or not all(d == current_dataset for d in dataset_names):
            print(f"Warning: Batch {batch_idx} mixed task/dataset. Skipping.")
            skipped_batches += 1
            continue

        optimizer.zero_grad()

        try:
            # Pass necessary inputs to the LSTM model's forward method
            logits, _ = model(input_ids, attention_mask, task_type) # Pass mask, lengths if needed
        except Exception as e:
            print(f"Error in forward pass batch {batch_idx}. Skipping. Error: {e}")
            skipped_batches += 1
            continue

        # Loss calculation (identical logic to final.py)
        try:
            loss = None
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
                if class_weights is None:
                    loss_fn = nn.CrossEntropyLoss()
                else:
                    if not isinstance(class_weights, torch.Tensor):
                        try: class_weights = torch.tensor(class_weights, dtype=torch.float, device=device)
                        except Exception: class_weights = None
                    else: class_weights = class_weights.to(device)
                    loss_fn = nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None else nn.CrossEntropyLoss()
                loss = loss_fn(logits, labels.long())
            else:
                print(f"Warning: Unknown task '{current_task}' in loss calc batch {batch_idx}. Skipping loss.")
                skipped_batches += 1
                continue

            if loss is not None and torch.isnan(loss):
                print(f"Warning: NaN loss detected batch {batch_idx}. Skipping backward pass.")
                skipped_batches += 1
                continue
            if loss is None: continue

        except Exception as e:
            print(f"Error during loss calculation batch {batch_idx}. Skipping. Error: {e}")
            skipped_batches += 1
            continue

        try:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping
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
    """Evaluates the LSTM model."""
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
                # lengths = batch['lengths'].to(device) # If needed
            except Exception as e:
                print(f"Error processing eval batch {batch_idx}. Skipping. Error: {e}")
                skipped_batches += 1
                continue

            if not task_type or not dataset_names:
                print(f"Warning: Eval Batch {batch_idx} empty task/dataset. Skipping.")
                skipped_batches += 1
                continue
            current_task = task_type[0]
            current_dataset = dataset_names[0]
            if not all(t == current_task for t in task_type) or not all(d == current_dataset for d in dataset_names):
                print(f"Warning: Eval Batch {batch_idx} mixed types. Skipping.")
                skipped_batches += 1
                continue

            try:
                logits, _ = model(input_ids, attention_mask, task_type) # Pass mask, lengths if needed
            except Exception as e:
                print(f"Error during eval forward pass batch {batch_idx}. Skipping. Error: {e}")
                skipped_batches += 1
                continue

            # Loss and prediction calculation (identical logic to final.py)
            try:
                loss = None
                preds = None
                if current_task == 'disagreement':
                    pos_weight = dataset_weights_map.get(current_dataset)
                    if pos_weight is None: pos_weight = torch.tensor(1.0, device=device)
                    else:
                        if not isinstance(pos_weight, torch.Tensor): pos_weight = torch.tensor(pos_weight, dtype=torch.float, device=device)
                        else: pos_weight = pos_weight.to(device)
                    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                    loss = loss_fn(logits.squeeze(-1), labels.float())
                    preds = (torch.sigmoid(logits).squeeze(-1) > 0.5).int()

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
                print(f"Error during eval loss/pred calc batch {batch_idx}. Skipping. Error: {e}")
                skipped_batches += 1

    if skipped_batches > 0:
        print(f"Skipped {skipped_batches} batches during evaluation.")

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    # Calculate Metrics (identical logic to final.py)
    if not all_labels or not all_preds or len(all_labels) != len(all_preds):
         print("Warning: No valid labels/predictions collected. Cannot calculate metrics.")
         return avg_loss, 0.0, 0.0, 0.0, 0.0

    try:
        int_labels = [int(l) for l in all_labels]
        average_strategy = 'binary' if max(int_labels) <= 1 else 'weighted'
        print(f"Using '{average_strategy}' averaging for metrics (max label = {max(int_labels)}).")

        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average=average_strategy, zero_division=0)
        precision = precision_score(all_labels, all_preds, average=average_strategy, zero_division=0)
        recall = recall_score(all_labels, all_preds, average=average_strategy, zero_division=0)
    except Exception as e:
        print(f"Error calculating evaluation metrics: {e}")
        accuracy, f1, precision, recall = 0.0, 0.0, 0.0, 0.0

    return avg_loss, accuracy, f1, precision, recall


# ===========================================
# === Main Script Execution ===
# ===========================================
if __name__ == "__main__":

    # 1. Load, Preprocess Data, and Build Vocabulary
    all_datasets_dict, dataset_weights_map, vocab = load_and_process_all_data(data_dir=DATA_DIR)

    # 2. Create Datasets and DataLoaders using LstmMultiTaskDataset and collate_batch
    train_dataloaders = []
    val_dataloaders = {}
    test_dataloaders = {}

    print("\n--- Creating PyTorch Datasets and DataLoaders for LSTM ---")
    for name, df in all_datasets_dict.items():
        print(f"Processing dataset for DataLoader: {name} ({len(df)} samples)")
        if df.empty or 'cleaned_text' not in df.columns: # Check for cleaned_text
            print(f"Skipping empty or improperly processed DataFrame: {name}")
            continue

        stratify_key = None
        if 'label' in df.columns and df['label'].nunique() > 1:
             stratify_key = df['label']
        else:
             print(f"Warning: Cannot stratify for {name}.")

        is_test_set = name.startswith("test_") or "test" in name.lower()

        if is_test_set:
            print(f"Creating TEST dataloader for: {name}")
            try:
                # Use LstmMultiTaskDataset
                test_dataset = LstmMultiTaskDataset(df, vocab, MAX_LENGTH)
                # Use custom collate_fn
                test_dataloaders[name] = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
            except Exception as e:
                print(f"ERROR creating TEST dataloader for {name}: {e}")
        else:
            try:
                train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=stratify_key)
            except ValueError as e:
                print(f"Warning: Stratify failed for {name}. Using regular split. Error: {e}")
                train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

            if not train_df.empty:
                 print(f"Creating TRAIN dataloader for: {name} ({len(train_df)} samples)")
                 try:
                     train_dataset = LstmMultiTaskDataset(train_df, vocab, MAX_LENGTH)
                     train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
                     train_dataloaders.append(train_loader)
                 except Exception as e:
                     print(f"ERROR creating TRAIN dataloader for {name}: {e}")

            if not val_df.empty:
                 print(f"Creating VALIDATION dataloader for: {name} ({len(val_df)} samples)")
                 try:
                     val_dataset = LstmMultiTaskDataset(val_df, vocab, MAX_LENGTH)
                     val_dataloaders[name] = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
                 except Exception as e:
                     print(f"ERROR creating VALIDATION dataloader for {name}: {e}")

    print("\n--- LSTM Training Setup ---")
    print(f"Vocab Size: {len(vocab)}")
    print(f"Embedding Dim: {EMBEDDING_DIM}, Hidden Dim: {HIDDEN_DIM}")
    print(f"Batch Size: {BATCH_SIZE}, Epochs: {EPOCHS}, LR: {LEARNING_RATE}")
    print(f"Weight Decay: {WEIGHT_DECAY}, Dropout: {DROPOUT_RATE}")
    print(f"Scheduler Patience: {SCHEDULER_PATIENCE}")
    print(f"Total training dataset loaders: {len(train_dataloaders)}")
    print(f"Validation datasets: {list(val_dataloaders.keys())}")
    print(f"Test datasets: {list(test_dataloaders.keys())}")

    if not train_dataloaders:
        print("\nFATAL: No training dataloaders created. Exiting.")
        exit()

    # 3. Initialize LSTM Model, Optimizer, Scheduler
    model = LstmMultiTaskModel(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_emotion_classes=NUM_EMOTION_CLASSES,
        num_layers=NUM_LSTM_LAYERS,
        bidirectional=BIDIRECTIONAL_LSTM,
        dropout=DROPOUT_RATE,
        pad_idx=vocab.stoi[PAD_TOKEN]
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=SCHEDULER_PATIENCE, verbose=True)

    # 4. Training Loop (Similar to final.py)
    print("\n--- Starting LSTM Training ---")
    best_val_loss = float('inf')
    model_save_path = "lstm_multitask_model_best.pth" # Different save path
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")
        epoch_train_loss = 0.0
        num_valid_train_loaders = 0
        print(f"Training on {len(train_dataloaders)} dataset loaders sequentially...")

        for i, train_loader in enumerate(train_dataloaders):
            loader_dataset_name = "Unknown"
            try: # Safely get dataset name if loader not empty
                 if len(train_loader) > 0:
                    batch_iterator = iter(train_loader)
                    temp_batch = next(batch_iterator)
                    if temp_batch and 'dataset_name' in temp_batch and temp_batch['dataset_name']:
                        loader_dataset_name = temp_batch['dataset_name'][0]
                    else: loader_dataset_name = f"Loader_{i+1}_Unnamed"
                 else: print(f"Skipping empty train loader {i+1}."); continue
            except StopIteration: print(f"Train loader {i+1} yielded no batches. Skipping."); continue
            except Exception as e: loader_dataset_name = f"Loader_{i+1}"; print(f"Could not get dataset name for loader {i+1}. Error: {e}")

            print(f"\nTraining on loader {i+1}/{len(train_dataloaders)} (Dataset: {loader_dataset_name})...")
            loader_avg_loss = train_epoch(model, train_loader, optimizer, DEVICE, dataset_weights_map)
            print(f"Avg Loss for loader {i+1} ({loader_dataset_name}): {loader_avg_loss:.4f}")
            if not math.isnan(loader_avg_loss) and not math.isinf(loader_avg_loss):
                epoch_train_loss += loader_avg_loss
                num_valid_train_loaders += 1
            else: print(f"Warning: Invalid training loss ({loader_avg_loss}) for loader {i+1}.")

        avg_epoch_train_loss = epoch_train_loss / num_valid_train_loaders if num_valid_train_loaders > 0 else 0.0
        print(f"\nEpoch {epoch + 1} Average Training Loss: {avg_epoch_train_loss:.4f}")

        # --- Validation Step ---
        epoch_val_loss = 0.0
        num_valid_val_loaders = 0
        print("\n--- Validation ---")
        if not val_dataloaders:
            print("No validation datasets found.")
        else:
            for name, val_loader in val_dataloaders.items():
                if len(val_loader) == 0: print(f"Skipping empty validation loader: {name}"); continue
                print(f"Evaluating on validation set: {name}")
                val_loss, accuracy, f1, precision, recall = evaluate(model, val_loader, DEVICE, dataset_weights_map)
                print(f"  - Val Loss: {val_loss:.4f}, Acc: {accuracy:.4f}, F1: {f1:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}")
                if not math.isnan(val_loss) and not math.isinf(val_loss):
                    epoch_val_loss += val_loss
                    num_valid_val_loaders += 1
                else: print(f"  - Warning: Invalid validation loss ({val_loss}) for {name}.")

            avg_epoch_val_loss = epoch_val_loss / num_valid_val_loaders if num_valid_val_loaders > 0 else float('inf')
            print(f"\nEpoch {epoch + 1} Average Validation Loss: {avg_epoch_val_loss:.4f}")

            if avg_epoch_val_loss != float('inf'):
                if avg_epoch_val_loss < best_val_loss:
                    print(f"Validation loss improved ({best_val_loss:.4f} --> {avg_epoch_val_loss:.4f}). Saving model to {model_save_path}...")
                    try: torch.save(model.state_dict(), model_save_path); best_val_loss = avg_epoch_val_loss; epochs_no_improve = 0
                    except Exception as e: print(f"ERROR saving model: {e}")
                else:
                    epochs_no_improve += 1
                    print(f"Validation loss did not improve. Epochs without improvement: {epochs_no_improve}")
                scheduler.step(avg_epoch_val_loss)
                
            else: print("Avg validation loss invalid. Skipping scheduler/save.")

    print("\n--- LSTM Training Finished ---")

    # 5. Load Best LSTM Model state
    print(f"\nLoading best LSTM model state from {model_save_path}...")
    final_model = LstmMultiTaskModel(len(vocab), EMBEDDING_DIM, HIDDEN_DIM, NUM_EMOTION_CLASSES).to(DEVICE) # Re-initialize
    if os.path.exists(model_save_path):
        try:
            final_model.load_state_dict(torch.load(model_save_path, map_location=DEVICE))
            print("Best LSTM model loaded successfully.")
        except Exception as e:
            print(f"Error loading best LSTM model state: {e}. Using model from end of training.")
            final_model = model # Fallback to last trained model
    else:
        print(f"Best LSTM model file ({model_save_path}) not found. Using model from end of training.")
        final_model = model # Fallback to last trained model

    final_model.eval()

    # 6. Final Test Set Evaluation for LSTM
    print("\n--- Final LSTM Test Set Evaluation ---")
    if not test_dataloaders:
        print("No test dataloaders found.")
    else:
        for name, test_loader in test_dataloaders.items():
            if len(test_loader) == 0: print(f"Skipping empty test loader: {name}"); continue
            print(f"Evaluating LSTM on Test Set: {name}")
            test_loss, accuracy, f1, precision, recall = evaluate(final_model, test_loader, DEVICE, dataset_weights_map)
            print(f"  - Test Loss: {test_loss:.4f}, Acc: {accuracy:.4f}, F1: {f1:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}")

    print("\n--- LSTM Script Finished ---")