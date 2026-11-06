# Tax Fraud Detection — Tunisia Ministry of Finance Data

Detect potential under-reporting of taxes using historical taxpayer data and supervised machine learning.

## Background

Tax fraud is the intentional misstatement of a tax return to lower tax liability. The most common form is **under-reporting**: declaring a smaller tax base than the real one. This reduces public revenue and weakens funding for public services.

Tax authorities cannot audit everyone, so the goal is to build a model that flags the taxpayers most likely to be committing fraud. That lets inspectors focus their limited time where it matters most.

## Dataset

| File | Rows | Columns | Description |
|------|------|---------|-------------|
| `Train.csv` | 21,295 | 121 | Historical taxpayer records with the known `target` |

### Column groups

The column names are abbreviations of the original French field names. They fall into families by prefix:

| Prefix | What it describes |
|--------|-------------------|
| `BCT_`, `CTR_`, `FJU_`, `ACT_` | Taxpayer profile: tax office, tax ID, category, legal form, activity code, and which tax obligations apply (`CTR_OBL*`) |
| `EXE_EXERCI`, `RES_ANNIMP` | Fiscal year / tax year |
| `TVA_` | VAT (TVA) declarations: turnover at different rates, deductions, credits, amounts paid |
| `ADB_`, `SND_` | Assessed amounts and audit/inspection-related amounts (suffixes `_I`, `_E`, `_A` are different categories) |
| `AX1_` … `AX5_` | Annex data: gross taxable amounts, fees, rents, capital income, margins |
| `FAC_` | Invoice data: amounts as supplier (`_F`) and as customer (`_C`) |
| `id` | Unique row identifier (not a feature) |
| `target` | What we want to predict |

> **Check first:** `target` is stored as a float. During exploration, confirm whether it is a 0/1 fraud label (classification) or a fraud amount (regression). This decides which models and metrics to use.

## Project structure

```
tax-fraud-detection/
├── data/
│   ├── raw/            # Original CSVs (not committed to git)
│   └── processed/      # Cleaned data / engineered features
├── notebooks/
│   ├── 01_eda.ipynb            # Explore the data
│   ├── 02_preprocessing.ipynb  # Clean and engineer features
│   └── 03_modeling.ipynb       # Train and evaluate models
├── src/                # Reusable Python code
├── models/             # Saved trained models (not committed)
├── submissions/        # Prediction files
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

1. **Install Python 3.10+.**
2. **Create a virtual environment** (keeps this project's packages separate):
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS / Linux
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Add the data:** put `Train.csv` in `data/raw/`.
5. **Start Jupyter:**
   ```bash
   jupyter lab
   ```

## Workflow (step by step)

### 1. Understand the problem
- Read the project overview and decide: classification or regression? (see note above)
- Decide on an evaluation metric:
  - Classification → ROC-AUC, F1, precision/recall (fraud is usually rare, so plain accuracy is misleading)
  - Regression → RMSE or MAE

### 2. Exploratory Data Analysis (EDA)
- Look at shape, data types, and `df.describe()`
- Plot the distribution of `target`; check how imbalanced it is
- Count missing values per column (many `TVA_`, `SND_`, `FAC_` columns are likely sparse)
- Find constant or near-constant columns
- Check how `target` relates to key features

### 3. Data cleaning and preprocessing
- Drop identifiers that carry no signal (`id`; consider `CTR_MATFIS`, the tax ID)
- Handle missing values (for financial amounts, missing often means 0 — verify before assuming)
- Encode categorical columns (`CTR_CATEGO_X`, `FJU_CODFJU`, `ACT_CODACT`, `BCT_CODBUR`)
- Remove columns with almost no variation

### 4. Feature engineering
Fraud often shows up as **inconsistencies** between related numbers. Ideas:
- Total declared turnover across all VAT rates
- Ratio of deductions to turnover
- Invoice amounts (`FAC_`) vs. declared turnover (`TVA_`) — big gaps can signal under-reporting
- Count of tax obligations a taxpayer has
- Year-over-year changes for the same taxpayer

### 5. Modeling
- Split data into training and validation sets (or use cross-validation)
- Start with a simple **baseline** (e.g. logistic/linear regression)
- Move to tree-based models, which usually work best on this kind of tabular data: Random Forest, LightGBM, XGBoost
- Handle class imbalance if needed (`class_weight`, `scale_pos_weight`)

### 6. Evaluation and interpretation
- Compare models on the validation metric
- Look at feature importance to see what drives predictions
- Sanity-check: do the top-ranked suspects make sense?

### 7. Predict and save
- Retrain the best model on all training data
- Save it to `models/` with `joblib`
- Generate predictions for new/test data into `submissions/`

## Tips for beginners
- **Get a baseline working end-to-end first**, then improve it.
- **Never tune on the data you evaluate on** — always keep a validation set.
- **Watch for data leakage**: some columns (e.g. audit outcomes) may only exist *because* fraud was found. If a feature looks too good to be true, investigate it.
- Use a fixed `random_state` so results are reproducible.

## Data confidentiality
This is government tax data. Keep raw data out of version control and do not share it publicly.
