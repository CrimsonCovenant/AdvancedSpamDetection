# SMS Spam Detection — Comparative NLP Study

## Project Proposal Alignment

| Proposal Requirement | Implementation |
|----------------------|----------------|
| UCI SMS Spam dataset (5,572 messages, 747 spam) | Auto-downloaded in `utils/data_loader.py` |
| Stratified 80/20 train-test split | `split_data()` — `TEST_SIZE=0.20`, stratified |
| Internal validation partition | 10% drawn from training set for early stopping |
| SVM with class weighting [6] | `models/svm_model.py` — `class_weight='balanced'` |
| TextCNN for n-gram patterns [5] | `models/textcnn_model.py` — filter sizes {2,3,4,5} |
| Bi-GRU + GloVe embeddings [4] | `models/bigru_model.py` — GloVe-100d + attention |
| DistilBERT subword model [1] | `models/distilbert_model.py` |
| RoBERTa high-performance model [2] | `models/roberta_model.py` |
| Weighted cross-entropy (deep models) | `utils/trainer.py` — inverse-frequency weights |
| MCC, PR-AUC, F1 evaluation | `utils/evaluation.py` |

## Structure

```
sms_spam/
├── data/                        # Dataset + GloVe embeddings
├── models/
│   ├── svm_model.py             # [6] LinearSVC + TF-IDF
│   ├── textcnn_model.py         # [5] Parallel Conv1d filters
│   ├── bigru_model.py           # Bi-GRU + GloVe [4] + Attention
│   ├── distilbert_model.py      # [1] DistilBERT-base-uncased
│   └── roberta_model.py         # [2] RoBERTa-base
├── utils/
│   ├── data_loader.py           # Download, clean, split, encode
│   ├── trainer.py               # Shared PyTorch training loop
│   └── evaluation.py            # MCC / F1 / PR-AUC + plots
├── train.py                     # Main training entry point
├── evaluate.py                  # Plot generation
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt

# Optional: GloVe embeddings for Bi-GRU / TextCNN
# Download glove.6B.zip from https://nlp.stanford.edu/projects/glove/
# Extract glove.6B.100d.txt into data/
```

## Running

```bash
# Train all models
python train.py

# Train a specific model
python train.py --model svm
python train.py --model textcnn
python train.py --model bigru
python train.py --model distilbert
python train.py --model roberta

# Generate plots
python evaluate.py
```

## References

- [1] Sanh et al., DistilBERT, arXiv:1910.01108, 2019.
- [2] Liu et al., RoBERTa, arXiv:1907.11692, 2019.
- [3] Devlin et al., BERT, NAACL-HLT 2019.
- [4] Pennington et al., GloVe, EMNLP 2014.
- [5] Kim, TextCNN, EMNLP 2014.
- [6] Yang et al., Weighted SVM, IJPRAI 2007.
