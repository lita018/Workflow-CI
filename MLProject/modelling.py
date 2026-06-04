"""
Modelling - Heart Disease Classification
Untuk MLflow Project (Kriteria 3)
Nama: Herlita | Username: lita018
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, ConfusionMatrixDisplay
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data(data_dir: str):
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    X_train = train_df.drop("target", axis=1)
    y_train = train_df["target"]
    X_test = test_df.drop("target", axis=1)
    y_test = test_df["target"]
    logger.info(f"Data loaded - Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["No Disease", "Disease"])
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title("Confusion Matrix - Heart Disease", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_feature_importance(model, feature_names, save_path, top_n=15):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(top_n), importances[indices][::-1], color='#1976D2')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices[::-1]])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def train(args):
    X_train, X_test, y_train, y_test = load_data(args.data_dir)

    max_depth = None if args.max_depth == "None" else int(args.max_depth)

    # Setup MLflow
    mlflow.set_experiment("Heart Disease - CI Pipeline")

    with mlflow.start_run():
        # Log params
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", args.min_samples_split)
        mlflow.log_param("min_samples_leaf", args.min_samples_leaf)
        mlflow.log_param("random_state", 42)

        # Train model
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Metrics
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
        }

        for k, v in metrics.items():
            mlflow.log_metric(k, v)
            logger.info(f"  {k}: {v:.4f}")

        # Artifacts
        os.makedirs("artifacts", exist_ok=True)
        cm_path = "artifacts/confusion_matrix.png"
        fi_path = "artifacts/feature_importance.png"
        report_path = "artifacts/classification_report.txt"
        metrics_path = "artifacts/metrics.json"

        plot_confusion_matrix(y_test, y_pred, cm_path)
        plot_feature_importance(model, X_train.columns.tolist(), fi_path)

        report = classification_report(y_test, y_pred,
                                       target_names=["No Disease", "Disease"])
        with open(report_path, 'w') as f:
            f.write(report)

        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)

        mlflow.log_artifact(cm_path, "plots")
        mlflow.log_artifact(fi_path, "plots")
        mlflow.log_artifact(report_path, "reports")
        mlflow.log_artifact(metrics_path, "metrics")

        # Log model
        mlflow.sklearn.log_model(
            model, "model",
            registered_model_name="heart-disease-ci"
        )

        run_id = mlflow.active_run().info.run_id
        logger.info(f"MLflow Run ID: {run_id}")

    logger.info("Training selesai!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=str, default="None")
    parser.add_argument("--min_samples_split", type=int, default=2)
    parser.add_argument("--min_samples_leaf", type=int, default=1)
    parser.add_argument("--data_dir", type=str, default="heart_disease_preprocessing")
    args = parser.parse_args()

    logger.info("=" * 50)
    logger.info("TRAINING - Heart Disease MLProject")
    logger.info("=" * 50)
    train(args)
