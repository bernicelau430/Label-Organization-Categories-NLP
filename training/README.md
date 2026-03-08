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