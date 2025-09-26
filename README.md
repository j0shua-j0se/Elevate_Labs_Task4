# Logistic Regression Classification — Task 4

This repository implements a complete binary classification workflow using Logistic Regression, from data loading and preprocessing to evaluation with confusion matrix, precision, recall, ROC–AUC, and threshold tuning.

## Objective

Build a binary classifier with Logistic Regression, standardize features, evaluate with core classification metrics, visualize ROC and Precision–Recall curves, and tune the decision threshold while explaining the sigmoid function.

## Dataset

- Dataset: Breast Cancer Wisconsin (Diagnostic) dataset saved locally.
- Default path used in the code: C:\Users\OMEN\.cache\kagglehub\datasets\uciml\breast-cancer-wisconsin-data\versions\2\data.csv
- Target mapping: diagnosis column mapped to binary labels (M → 1 for malignant, B → 0 for benign).
- Non-feature identifiers (e.g., id, unnamed empty columns) are dropped before modeling.

## Environment

- Python 3.8+
- Libraries: pandas, numpy, scikit-learn, seaborn, matplotlib

Install:
```bash
pip install pandas numpy scikit-learn seaborn matplotlib
```

## How to run

Option A: Notebook
- Open the notebook.
- Verify INPUT_CSV and target mapping (diagnosis → 0/1).
- Run cells top to bottom to train the model, render plots, and print metrics.

Option B: Script
- Place the provided script in src/.
- Edit INPUT_CSV at the top of the script if needed.
- Run:
```bash
python src/logreg_task4.py
```

## Workflow

### Data loading and curation
- Read the CSV from the configured path.
- Drop non-feature columns such as id and unnamed empty columns.
- Map diagnosis to binary labels (M=1, B=0).

### Train/test split
- Stratified train/test split (default 80/20) to preserve class balance.

### Preprocessing
- Numeric: median imputation and StandardScaler.
- Categorical (if any): most_frequent imputation and one-hot encoding.
- Implemented with a ColumnTransformer inside a Pipeline to prevent leakage.

### Model training
- LogisticRegression(max_iter=1000) trained inside the Pipeline for consistent transforms across fit and predict.

### Evaluation at threshold 0.5
- Confusion matrix (with labeled plot).
- Classification report: precision, recall, F1, support.
- ROC–AUC score.

### Threshold-independent diagnostics
- ROC curve with AUC.
- Precision–Recall curve with PR AUC and a “no-skill” baseline.

### Threshold tuning
- Sweep thresholds from 0.0 to 1.0 to maximize F1.
- Re-evaluate and re-plot confusion matrix at the best-F1 threshold.
- Provide updated classification report at the chosen threshold.

### Sigmoid explanation
- Logistic regression converts the linear score z = w^T x + b to probability via sigmoid(z) = 1 / (1 + exp(−z)), enabling threshold-based decisions over predicted probabilities.

### Optional export
- Export the fully transformed model matrix (features after preprocessing) with feature names and the target to a CSV for auditing and downstream experiments.

## Outputs

- Console:
  - Classification metrics at threshold 0.5 and at the best-F1 threshold.
  - ROC–AUC and PR–AUC scores.
- Plots:
  - Confusion matrix (0.5 threshold).
  - ROC curve with AUC.
  - Precision–Recall curve with PR AUC.
  - Confusion matrix at best-F1 threshold.
- Optional:
  - Transformed model matrix CSV (numeric features + target).

## Configuration

- Edit INPUT_CSV in the code to point to the dataset if the local path is different.
- Toggle EXPORT_MATRIX to True to save the transformed design matrix with feature names.
- Adjust RANDOM_STATE and TEST_SIZE as needed.

## License and acknowledgments

- This repository is for educational use within the internship Task 4 scope.
- Data: Breast Cancer Wisconsin (Diagnostic) dataset prepared locally from a cached source.

<img width="490" height="676" alt="image" src="https://github.com/user-attachments/assets/4291caa9-5b0b-438a-bad1-a367c7917aba" />
