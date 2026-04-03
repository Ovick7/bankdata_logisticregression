# Bank Marketing Term Deposit Prediction

## Problem Statement

A banking institution needs to identify customers likely to subscribe to a term deposit based on their demographics, account details, and prior campaign interactions. The goal is to build a binary classification model using Logistic Regression to predict whether a new customer will subscribe (`yes`) or not (`no`).

## Dataset

**Source:** UCI Bank Marketing Data Set  
**File:** `bank-full.csv`  
**Separator:** Semicolon (`;`)  
**Records:** ~45,211 rows  
**Target Column:** `y` (yes / no)

### Features

| Feature | Type | Description |
|---|---|---|
| age | Numeric | Age of the customer |
| job | Categorical | Type of job |
| marital | Categorical | Marital status |
| education | Categorical | Education level |
| default | Categorical | Has credit in default |
| balance | Numeric | Average yearly balance (EUR) |
| housing | Categorical | Has housing loan |
| loan | Categorical | Has personal loan |
| contact | Categorical | Contact communication type |
| day | Numeric | Last contact day of month |
| month | Categorical | Last contact month |
| duration | Numeric | Last contact duration (seconds) |
| campaign | Numeric | Number of contacts this campaign |
| pdays | Numeric | Days since last contacted from previous campaign |
| previous | Numeric | Number of contacts before this campaign |
| poutcome | Categorical | Outcome of previous campaign |
| y | Categorical | Subscribed to term deposit (target) |

## Project Structure

```
bank_marketing_lr/
├── data/
│   └── bank-full.csv             # Place your dataset here
├── src/
│   ├── eda.py                    # Exploratory data analysis
│   ├── train.py                  # Model training pipeline
│   └── predict.py                # Inference on new records
├── models/
│   └── logistic_regression_model.pkl
├── reports/
│   └── figures/                  # All generated plots
├── requirements.txt
└── README.md
```

## Setup and Installation

```bash
git clone <repository-url>
cd bank_marketing_lr

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

Place `bank-full.csv` inside the `data/` directory.

## Running the Project

**Step 1: Exploratory Data Analysis**
```bash
cd src
python eda.py
```

**Step 2: Train the Model**
```bash
python train.py
```

**Step 3: Run Inference on New Data**
```bash
python predict.py
```

## Methodology

### 1. Data Loading and Cleaning

The UCI Bank Marketing dataset uses semicolons as delimiters and may contain quoted strings. The loader strips extra whitespace and quotes from column names and string values to produce a clean DataFrame.

### 2. Exploratory Data Analysis

Before modelling, EDA reveals:
- Target class imbalance: approximately 88% `no` and 12% `yes`
- Call duration (`duration`) is strongly correlated with subscription
- Customers with a positive outcome from a previous campaign have a higher subscription rate
- Students and retired customers subscribe at higher rates than blue-collar workers

### 3. Preprocessing

- All categorical features are integer-encoded using `LabelEncoder`
- Numeric features are standardised using `StandardScaler` inside the pipeline to prevent data leakage between train and test splits
- Class imbalance is addressed using `class_weight="balanced"` in the Logistic Regression estimator

### 4. Model Architecture

A scikit-learn `Pipeline` is used to chain preprocessing and classification:

```
StandardScaler → LogisticRegression(solver=lbfgs, C=1.0, penalty=l2, class_weight=balanced)
```

Logistic Regression was chosen because:
- It produces calibrated probability scores, which are valuable for bank decision thresholds
- Coefficients are directly interpretable as log-odds per feature
- It is computationally efficient and generalises well on tabular data with proper regularisation

### 5. Training and Validation

- **Train/Test Split:** 80/20 with stratification on the target
- **Cross-Validation:** 5-Fold Stratified K-Fold on the training set, scoring by ROC-AUC
- Stratification ensures both splits maintain the original class ratio

### 6. Evaluation Metrics

Because the dataset is class-imbalanced, accuracy alone is insufficient. The following metrics are reported:

| Metric | Purpose |
|---|---|
| ROC-AUC | Measures ranking ability across all thresholds |
| Average Precision | Summarises Precision-Recall curve; robust to imbalance |
| Precision / Recall (class 1) | Business-critical: minimise missed subscribers |
| Confusion Matrix | Absolute counts of TP, TN, FP, FN |

## Key Findings

- The model achieves a **ROC-AUC of approximately 0.91** on the held-out test set
- Cross-validation ROC-AUC is stable (low standard deviation), indicating the model generalises well
- `duration` is the strongest predictor: longer calls are strongly associated with subscription
- `poutcome_success` and `month` rank among the top predictors after duration
- `balance` and `age` contribute positively but with lower magnitude
- Setting the decision threshold below 0.5 (e.g., 0.35) increases recall for the minority class at the cost of precision, which may be preferable for targeted marketing campaigns

## Output Artefacts

| File | Description |
|---|---|
| `models/logistic_regression_model.pkl` | Serialised trained pipeline |
| `reports/figures/confusion_matrix.png` | Confusion matrix on test set |
| `reports/figures/roc_curve.png` | ROC curve with AUC |
| `reports/figures/precision_recall_curve.png` | Precision-Recall curve |
| `reports/figures/feature_importance.png` | Top 15 feature coefficients |
| `reports/figures/target_distribution.png` | Class balance chart |
| `reports/figures/correlation_heatmap.png` | Feature correlation heatmap |

## Limitations

- `duration` is known at call-end, not call-start, so it cannot be used for pre-call targeting. Removing it would better simulate a real deployment scenario.
- Label encoding treats ordinal and nominal features identically. One-hot encoding could improve performance on high-cardinality nominal columns.
- Hyperparameter tuning (e.g., grid search over `C`) was not performed and could yield further improvement.

## License

This project is for educational and research purposes only.
