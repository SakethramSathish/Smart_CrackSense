import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score


def train_ml_model(
    X: np.ndarray,
    y: np.ndarray,
    model_path: str = "svm_model.pkl"
):
    """
    Trains an SVM classifier on extracted crack features.

    Parameters:
    -----------
    X : ndarray
        Feature matrix (samples x features).
    y : ndarray
        Labels (severity classes).
    model_path : str
        Path to save the trained model.

    Returns:
    --------
    model : trained SVM model
    accuracy : float
        Training accuracy.
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
    # Much faster than SVM for large datasets, still effective for imbalance
    print("Training Random Forest model (this may take 2-5 minutes)...")
    model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        class_weight='balanced',
        n_jobs=-1,  # Use all CPU cores
        verbose=1
    )
    model.fit(X_train, y_train)
    print("Random Forest training complete!")

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy:.4f}")

    # Save model and scaler
    with open(model_path, "wb") as f:
        pickle.dump(
            {"model": model, "scaler": scaler},
            f
        )

    return model, accuracy
