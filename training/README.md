# Model Training Scripts

This directory contains all machine learning model training, evaluation, and optimization scripts.

## Files

### `model_training.py` 
**Main Implementation**

Complete OrganizationCategorizer class with advanced NLP features.

**Accuracy and F1 Scores**

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