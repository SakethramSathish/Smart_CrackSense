import os
import json
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)


def train_ml_model(
    X: np.ndarray,
    y: np.ndarray,
    model_path: str = "experiments/model_outputs/random_forest_model.pkl",
    metrics_path: str = "experiments/performance_metrics/train_metrics.json",
):
    """
    Trains a RandomForest classifier on extracted crack features.

    Parameters:
    -----------
    X : ndarray
        Feature matrix (samples x features).
    y : ndarray
        Labels (severity classes).
    model_path : str
        Path to save the trained model and scaler.
    metrics_path : str
        Path to save training metrics.

    Returns:
    --------
    model : trained RandomForest model
    metrics : dict
        Evaluation metrics for the test set.
    """

    # Split data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train set: {len(X_train)}, Test set: {len(X_test)}")

    # Feature scaling
    print("Scaling features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train RandomForest classifier with balanced class weights
    print("Training Random Forest model (this may take 2-5 minutes)...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
        verbose=1,
    )
    model.fit(X_train, y_train)
    print("Random Forest training complete!")

    # Evaluate model
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_weighted": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"Recall (macro): {metrics['recall_macro']:.4f}")
    print(f"F1 Score (macro): {metrics['f1_macro']:.4f}")

    # Save model and scaler
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler}, f)

    # Save metrics
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    return model, metrics


def resolve_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))


def main():
    project_root = resolve_project_root()
    feature_dir = os.path.join(project_root, "experiments", "feature_vectors")
    output_model_path = os.path.join(project_root, "experiments", "model_outputs", "random_forest_model.pkl")
    metrics_path = os.path.join(project_root, "experiments", "performance_metrics", "training_metrics.json")

    X_path = os.path.join(feature_dir, "X.npy")
    y_path = os.path.join(feature_dir, "y.npy")

    if not os.path.exists(X_path) or not os.path.exists(y_path):
        raise FileNotFoundError(
            "Feature vectors not found. Run extract_dataset_features.py first to generate experiments/feature_vectors/X.npy and y.npy."
        )

    X = np.load(X_path)
    y = np.load(y_path)

    train_ml_model(X, y, model_path=output_model_path, metrics_path=metrics_path)
    print(f"Saved model to: {output_model_path}")
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
