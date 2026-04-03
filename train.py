import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

DATASET_PATH = "../data/bank-full.csv"
MODEL_OUTPUT_DIR = "../models"
PLOT_OUTPUT_DIR = "../reports/figures"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5


def load_data(path: str) -> pd.DataFrame:
    sep = ";" if path.endswith(".csv") else ","
    df = pd.read_csv(path, sep=sep)
    df.columns = df.columns.str.strip().str.replace('"', "")
    df = df.applymap(lambda x: x.strip().replace('"', "") if isinstance(x, str) else x)
    return df


def explore_data(df: pd.DataFrame) -> None:
    print("Dataset Shape:", df.shape)
    print("\nColumn Types:\n", df.dtypes)
    print("\nClass Distribution:\n", df["y"].value_counts())
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nStatistical Summary:\n", df.describe())


def preprocess(df: pd.DataFrame):
    df = df.copy()

    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    if "y" in categorical_cols:
        categorical_cols.remove("y")

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    target_le = LabelEncoder()
    df["y"] = target_le.fit_transform(df["y"])

    X = df.drop(columns=["y"])
    y = df["y"]

    return X, y, label_encoders, target_le


def build_pipeline(class_weights: dict) -> Pipeline:
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                    class_weight=class_weights,
                    C=1.0,
                    penalty="l2",
                ),
            ),
        ]
    )
    return pipeline


def evaluate_model(pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "avg_precision": average_precision_score(y_test, y_proba),
        "classification_report": classification_report(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "y_pred": y_pred,
        "y_proba": y_proba,
    }
    return metrics


def plot_confusion_matrix(cm: np.ndarray, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No", "Yes"],
        yticklabels=["No", "Yes"],
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
    plt.close()


def plot_roc_curve(y_test, y_proba: np.ndarray, auc: float, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC Curve (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=150)
    plt.close()


def plot_precision_recall(y_test, y_proba: np.ndarray, avg_prec: float, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="darkorange", lw=2, label=f"AP = {avg_prec:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "precision_recall_curve.png"), dpi=150)
    plt.close()


def plot_feature_importance(pipeline, feature_names: list, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    coefficients = pipeline.named_steps["classifier"].coef_[0]
    feature_df = pd.DataFrame(
        {"feature": feature_names, "coefficient": coefficients}
    ).sort_values("coefficient", key=abs, ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["steelblue" if c > 0 else "tomato" for c in feature_df["coefficient"]]
    ax.barh(feature_df["feature"], feature_df["coefficient"], color=colors)
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
    ax.set_xlabel("Logistic Regression Coefficient")
    ax.set_title("Top 15 Feature Importances")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=150)
    plt.close()


def cross_validate(pipeline, X_train, y_train) -> None:
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring="roc_auc")
    print(f"\nCross-Validation ROC-AUC ({CV_FOLDS}-Fold):")
    print(f"  Scores : {cv_scores.round(4)}")
    print(f"  Mean   : {cv_scores.mean():.4f}")
    print(f"  Std Dev: {cv_scores.std():.4f}")


def save_model(pipeline, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "logistic_regression_model.pkl")
    joblib.dump(pipeline, model_path)
    print(f"\nModel saved to: {model_path}")


def main():
    print("Loading dataset...")
    df = load_data(DATASET_PATH)

    print("\nData Exploration")
    explore_data(df)

    print("\nPreprocessing data...")
    X, y, label_encoders, target_le = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))
    print(f"\nClass Weights (balanced): {class_weights}")

    print("\nBuilding pipeline...")
    pipeline = build_pipeline(class_weights)

    print("\nRunning cross-validation...")
    cross_validate(pipeline, X_train, y_train)

    print("\nFitting model on training data...")
    pipeline.fit(X_train, y_train)

    print("\nEvaluating model on test set...")
    metrics = evaluate_model(pipeline, X_test, y_test)

    print(f"\nAccuracy  : {metrics['accuracy']:.4f}")
    print(f"ROC-AUC   : {metrics['roc_auc']:.4f}")
    print(f"Avg Prec  : {metrics['avg_precision']:.4f}")
    print(f"\nClassification Report:\n{metrics['classification_report']}")

    print("\nGenerating plots...")
    plot_confusion_matrix(metrics["confusion_matrix"], PLOT_OUTPUT_DIR)
    plot_roc_curve(y_test, metrics["y_proba"], metrics["roc_auc"], PLOT_OUTPUT_DIR)
    plot_precision_recall(y_test, metrics["y_proba"], metrics["avg_precision"], PLOT_OUTPUT_DIR)
    plot_feature_importance(pipeline, list(X.columns), PLOT_OUTPUT_DIR)

    print(f"Plots saved to: {PLOT_OUTPUT_DIR}")

    save_model(pipeline, MODEL_OUTPUT_DIR)
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
