# Model Training Scripts

This directory contains all machine learning model training, evaluation, and optimization scripts.

## Files

### `model_training.py` 
Base model implementation with tuned hyperparameters determined by `hyperparameter_tuning.py`

**Accuracy and F1 Scores of the Tuned Model (k=3)**

- Business Classification:
   - Accuracy: 
   - Macro F1: 
- Group Classification:
   - Accuracy: 
   - Macro F1: 
- Industry Classification:
   - Accuracy: 
   - Macro F1: 

Overall Accuracy: 

**Accuracy and F1 Scores of the Base Model (k=7)**

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