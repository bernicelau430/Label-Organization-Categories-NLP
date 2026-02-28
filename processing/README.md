# Data Processing Scripts

This directory contains scripts for cleaning and processing the raw TSV data files.

## Files

### `tsv_cleaner.py`
Utility script that reconstructs TSV rows broken by newline characters.

**Function:**
- `clean_tsv(input_path, output_path)` - Cleans a TSV file and writes to output

**Usage:**
```python
import tsv_cleaner
tsv_cleaner.clean_tsv("../data/raw/Organizations.tsv", "../data/clean/Organizations.tsv")
```

### `processing.py`
Main data processing pipeline that:
1. Cleans the raw TSV files using `tsv_cleaner`
2. Loads data into pandas DataFrames
3. Drops unnecessary columns
4. Pivots OrgCategories from long to wide format
5. Joins all tables (Organizations, OrgCategories, OrgConcept)
6. Outputs final processed dataset

**Input:**
- `../data/clean/Organizations.tsv`
- `../data/clean/OrgCategories.tsv`
- `../data/clean/OrgConcept.tsv`

**Output:**
- `../data/processed/OrganizationsFull.tsv` (18,626 rows × 9 columns)

**Usage:**
```bash
cd processing
python processing.py
```

## Running Data Processing

From the project root:
```bash
cd processing
python processing.py
```

This will:
1. Clean all three raw TSV files
2. Process and join them
3. Create the final dataset in `data/processed/OrganizationsFull.tsv`