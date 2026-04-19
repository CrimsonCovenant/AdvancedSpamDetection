"""
utils/data_loader.py
--------------------
Data pipeline aligned with the project proposal:

  Split strategy (Section 4 of proposal):
    - Stratified 80/20 train-test split  (preserves ~13.4% spam ratio)
    - Internal validation partition drawn from the training set (10% of train)
      used exclusively for early stopping and model selection
    - Test set is never touched during training

  Three feature pipelines:
    1. TF-IDF unigrams + bigrams          → SVM  [6]
    2. Vocabulary index + GloVe-100d [4]  → TextCNN [5], Bi-GRU
    3. Subword tokenisation               → DistilBERT [1], RoBERTa [2]

  Class imbalance strategy:
    - SVM  : class_weight='balanced'  [6]
    - Deep : weighted cross-entropy   (inverse-frequency weights)

References
----------
[1] Sanh et al., DistilBERT, arXiv:1910.01108, 2019.
[2] Liu et al., RoBERTa, arXiv:1907.11692, 2019.
[4] Pennington et al., GloVe, EMNLP 2014.
[5] Kim, TextCNN, EMNLP 2014.
[6] Yang et al., Weighted SVM, IJPRAI 2007.
"""

import os
import re
import zipfile
import requests
import numpy as np
import pandas as pd
import nltk

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

# ── Constants ─────────────────────────────────────────────────────────────────
STOP_WORDS       = set(stopwords.words("english"))
DATA_DIR         = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
UCI_URL          = ("https://archive.ics.uci.edu/ml/machine-learning-databases"
                    "/00228/smsspamcollection.zip")
RAW_FILE         = os.path.join(DATA_DIR, "SMSSpamCollection")

RANDOM_SEED      = 42
TEST_SIZE        = 0.20          # 80/20 train-test split (proposal §4)
VAL_FRAC         = 0.10          # 10% of train → internal validation
MAX_VOCAB        = 20_000
MAX_SEQ_LEN      = 128
TRANSFORMER_LEN  = 128


# ── Download / load raw data ──────────────────────────────────────────────────
def _download_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(RAW_FILE):
        return
    print("Downloading UCI SMS Spam Collection ...")
    zip_path = os.path.join(DATA_DIR, "tmp.zip")
    r = requests.get(UCI_URL, timeout=30)
    r.raise_for_status()
    with open(zip_path, "wb") as f:
        f.write(r.content)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(DATA_DIR)
    os.remove(zip_path)
    print(f"  Saved to: {RAW_FILE}")


def load_raw_dataframe() -> pd.DataFrame:
    """Return DataFrame with columns ['label', 'text', 'label_int']."""
    _download_dataset()
    df = pd.read_csv(
        RAW_FILE, sep="\t", header=None,
        names=["label", "text"], encoding="latin-1",
    )
    df["label_int"] = (df["label"] == "spam").astype(int)
    spam = df["label_int"].sum()
    ham  = (df["label_int"] == 0).sum()
    print(f"Dataset: {len(df):,} messages  |  spam={spam} ({spam/len(df)*100:.1f}%)  "
          f"ham={ham} ({ham/len(df)*100:.1f}%)")
    return df


# ── Text cleaning ─────────────────────────────────────────────────────────────
def clean_text(text: str, remove_stopwords: bool = False) -> str:
    """
    Universal cleaning pipeline applied to all models.
    Steps:
      1. Lowercase
      2. URL tokens   → 'url'
      3. Email tokens → 'email'
      4. Phone numbers (10+ digits) → 'phone'
      5. Remove non-alphanumeric characters
      6. Collapse whitespace
      7. (SVM only) Remove stopwords
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\S+",  " url ",   text)
    text = re.sub(r"\S+@\S+",          " email ", text)
    text = re.sub(r"\b\d{10,}\b",      " phone ", text)
    text = re.sub(r"[^a-z0-9\s]",      " ",       text)
    text = re.sub(r"\s+",              " ",       text).strip()
    if remove_stopwords:
        text = " ".join(w for w in text.split() if w not in STOP_WORDS)
    return text


# ── Stratified splits ─────────────────────────────────────────────────────────
def split_data(df: pd.DataFrame):
    """
    Produce three disjoint, stratified partitions:

      train_df  : 72% of full dataset (80% × 90%)
      val_df    : 8%  of full dataset (80% × 10%)  ← internal only
      test_df   : 20% of full dataset

    The test set is held out and never used during training.
    val_df is used solely for early stopping and model selection.
    """
    # Step 1: 80/20 test split (proposal specification)
    train_val_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df["label_int"],
        random_state=RANDOM_SEED,
    )
    # Step 2: 10% of train → internal validation
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=VAL_FRAC,
        stratify=train_val_df["label_int"],
        random_state=RANDOM_SEED,
    )
    for name, part in [("Train", train_df), ("Val  ", val_df), ("Test ", test_df)]:
        spam = part["label_int"].sum()
        print(f"  {name}: {len(part):>5}  spam={spam:>3} ({spam/len(part)*100:.1f}%)  "
              f"ham={(part['label_int']==0).sum():>4}")
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


# ── SVM / TF-IDF pipeline ─────────────────────────────────────────────────────
def prepare_svm_data(train_df, val_df, test_df):
    """
    TF-IDF (1–2 gram, top 10,000, sublinear TF) feature matrices.
    Vectoriser fitted on training set only (no leakage).
    Returns: (X_train, y_train), (X_val, y_val), (X_test, y_test), vectoriser
    """
    def _clean(df_):
        return df_["text"].apply(lambda t: clean_text(t, remove_stopwords=True))

    vectoriser = TfidfVectorizer(
        max_features=10_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    X_train = vectoriser.fit_transform(_clean(train_df))
    X_val   = vectoriser.transform(_clean(val_df))
    X_test  = vectoriser.transform(_clean(test_df))

    y_train = train_df["label_int"].values
    y_val   = val_df["label_int"].values
    y_test  = test_df["label_int"].values

    print(f"  TF-IDF  train={X_train.shape}  val={X_val.shape}  test={X_test.shape}")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), vectoriser


# ── Vocabulary (TextCNN / Bi-GRU) ────────────────────────────────────────────
class Vocabulary:
    """
    Word-level vocabulary built from the training corpus only.
    Special tokens:
      index 0 → <PAD>  (padding; embedding zeroed out)
      index 1 → <UNK>  (out-of-vocabulary words at inference)
    """
    PAD = "<PAD>"
    UNK = "<UNK>"

    def __init__(self, max_size: int = MAX_VOCAB):
        self.max_size = max_size
        self.word2idx = {self.PAD: 0, self.UNK: 1}
        self.idx2word = {0: self.PAD, 1: self.UNK}

    def build(self, texts):
        from collections import Counter
        counts = Counter(w for t in texts for w in t.split())
        for word, _ in counts.most_common(self.max_size - 2):
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx]  = word
        print(f"  Vocabulary: {len(self.word2idx):,} tokens")
        return self

    def encode(self, text: str, max_len: int = MAX_SEQ_LEN):
        tokens = text.split()[:max_len]
        ids    = [self.word2idx.get(w, 1) for w in tokens]
        ids   += [0] * (max_len - len(ids))
        return ids

    def __len__(self):
        return len(self.word2idx)


def load_glove_embeddings(vocab: Vocabulary, glove_path: str, embed_dim: int = 100):
    """
    Load GloVe-100d vectors [4] for words in vocab.
    OOV words → small random vectors.  <PAD> → zero vector.
    Falls back to all-random if the file is absent.
    """
    emb = np.random.normal(scale=0.1, size=(len(vocab), embed_dim)).astype(np.float32)
    emb[0] = 0.0  # PAD

    if not os.path.exists(glove_path):
        print(f"  [WARNING] GloVe not found at {glove_path}. "
              "Using random initialisations.")
        return emb

    loaded = 0
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split()
            word  = parts[0]
            if word in vocab.word2idx:
                emb[vocab.word2idx[word]] = np.array(parts[1:], dtype=np.float32)
                loaded += 1
    print(f"  GloVe: {loaded}/{len(vocab):,} tokens "
          f"({loaded/len(vocab)*100:.1f}% coverage)")
    return emb


# ── PyTorch Datasets ──────────────────────────────────────────────────────────
class SMSDataset(Dataset):
    """Vocab-indexed dataset for TextCNN [5] and Bi-GRU."""

    def __init__(self, texts, labels, vocab: Vocabulary,
                 max_len: int = MAX_SEQ_LEN):
        self.inputs = [
            torch.tensor(vocab.encode(clean_text(t), max_len), dtype=torch.long)
            for t in texts
        ]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):           return len(self.labels)
    def __getitem__(self, idx):  return self.inputs[idx], self.labels[idx]


class TransformerSMSDataset(Dataset):
    """Subword-tokenised dataset for DistilBERT [1] and RoBERTa [2]."""

    def __init__(self, texts, labels, tokenizer,
                 max_len: int = TRANSFORMER_LEN):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):          return len(self.labels)
    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


# ── DataLoader factories ──────────────────────────────────────────────────────
def prepare_dl_data(train_df, val_df, test_df, batch_size: int = 64):
    """
    Build vocabulary, encode texts, return (train_loader, val_loader,
    test_loader, vocab) for TextCNN / Bi-GRU.
    """
    vocab = Vocabulary().build(
        [clean_text(t) for t in train_df["text"].tolist()]
    )

    def _loader(df_, shuffle):
        ds = SMSDataset(df_["text"].tolist(), df_["label_int"].tolist(), vocab)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=False)

    return (
        _loader(train_df, shuffle=True),
        _loader(val_df,   shuffle=False),
        _loader(test_df,  shuffle=False),
        vocab,
    )


def prepare_transformer_data(train_df, val_df, test_df,
                              model_name: str, batch_size: int = 16):
    """
    Tokenise with a HuggingFace tokeniser; return DataLoaders.
    model_name: 'distilbert-base-uncased' [1] or 'roberta-base' [2].
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _loader(df_, shuffle):
        ds = TransformerSMSDataset(
            df_["text"].tolist(), df_["label_int"].tolist(), tokenizer
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=False)

    return (
        _loader(train_df, shuffle=True),
        _loader(val_df,   shuffle=False),
        _loader(test_df,  shuffle=False),
        tokenizer,
    )


# ── Class-weight helper ───────────────────────────────────────────────────────
def compute_class_weights(labels, device="cpu"):
    """
    Inverse-frequency weights:  w_c = N / (K * n_c)
    Returns FloatTensor of shape (2,) placed on `device`.
    """
    counts  = np.bincount(labels)
    weights = len(labels) / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float).to(device)
