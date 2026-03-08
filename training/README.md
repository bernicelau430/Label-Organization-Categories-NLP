# Model Training Scripts

This directory contains all machine learning model training, evaluation, and optimization scripts.

## KNN Model

### `model_training.py` 
Base KNN model implementation with tuned hyperparameters determined by `hyperparameter_tuning.py`

**Accuracy and F1 Scores of the Tuned KNN Model (k=3)**

- Business Classification:
   - Accuracy: 0.8990
   - Macro F1: 0.7800
- Group Classification:
   - Accuracy: 0.9522
   - Macro F1: 0.9321
- Industry Classification:
   - Accuracy: 0.9229
   - Macro F1: 0.8419

Overall Accuracy: 0.9247

**Accuracy and F1 Scores of the Base KNN Model (k=7)**

- Business Classification:
   - Accuracy: 0.8090
   - Macro F1: 0.6223
- Group Classification:
   - Accuracy: 0.8806
   - Macro F1: 0.8568
- Industry Classification:
   - Accuracy: 0.8320
   - Macro F1: 0.6957

Overall Accuracy: 0.8405

**Key Features:**
- Text preprocessing (legal entity standardization, abbreviation expansion)
- Domain-specific feature engineering (industry keywords, org types, geographic scope)
- Concept-based kNN classification with similarity-weighted voting
- Model save/load functionality
- Comprehensive evaluation metrics

**Usage:**
```bash
cd training
python model_training.py
```

**Output:**
- Trains model with optimized parameters
- Evaluates on test set
- Saves model to `../model/org_categorizer_model.pkl`


## Random Forest Model (Data Split by Concept)

### `random_forest_model.py` 
Base Random Forest model implementation with the train-test subsets split by concept.

**Accuracy and F1 Scores of the Random Forest Model**

- Business Classification:
   - Accuracy: 0.4717
   - Macro F1: 0.2461
- Group Classification:
   - Accuracy: 0.6770
   - Macro F1: 0.5484
- Industry Classification:
   - Accuracy: 0.5752
   - Macro F1: 0.3316

Overall Accuracy: 0.5746

**Usage:**
```bash
cd training
python random_forest_model.py
```

**Output:**
- Trains model with optimized parameters
- Evaluates on test set
- Saves model to `../model/org_categorizer_randomforest_model.pkl`


## KNN Model (Data Split by Concept)

### `knn_model_v2.py` 
Base KNN model implementation with the train-test subsets split by concept.

**Accuracy and F1 Scores of the KNN v2 Model**

- Business Classification:
   - Accuracy: 0.4310
   - Macro F1: 0.2640
- Group Classification:
   - Accuracy: 0.6425
   - Macro F1: 0.5147
- Industry Classification:
   - Accuracy: 0.5504
   - Macro F1: 0.3504

Overall Accuracy: 0.5413

**Usage:**
```bash
cd training
python knn_model_v2.py
```

**Output:**
- Trains model with optimized parameters
- Evaluates on test set
- Saves model to `../model/org_categorizer_knn_model_v2.pkl`


## KNN Model v3 (Handles Data Leakage and Generalizes to No-Concept Organizations)

### `knn_model_v3.py`
Enhanced KNN model designed to reduce train-test leakage and support organizations that do not have a concept ID.

**Key changes from KNN v2:**
- Leakage-aware split groups:
   - Uses `oc_id` when present
   - Falls back to normalized organization name when `oc_id` is missing
   - Splits are performed at the group level (not raw row level) to avoid near-duplicate leakage
- No-concept organization support:
   - Keeps organizations with missing `oc_id` in train/test instead of dropping them
   - Predicts `Business`, `Group`, and `Industry` directly using weighted neighbor voting
- Additional word-sensitive feature engineering:
   - Explicit handling for legal/entity and membership-style words (e.g., `inc`, `corp`, `group`, `association`, `chapter`, `union`)
   - Common-word analysis utility to show how broadly frequent words are distributed across industries

**Accuracy and F1 Scores of the KNN v3 Model**

- Business Classification:
   - Accuracy: 0.6327
   - Macro F1: 0.5405
- Group Classification:
   - Accuracy: 0.8031
   - Macro F1: 0.7887
- Industry Classification:
   - Accuracy: 0.7192
   - Macro F1: 0.6557

Overall Accuracy: 0.7183

**Usage:**
```bash
cd training
python knn_model_v3.py
```

**Output:**
- Trains leakage-aware KNN model
- Evaluates on leakage-resistant test set
- Prints common-word behavior analysis
- Saves model to `../model/org_categorizer_knn_model_v3.pkl`