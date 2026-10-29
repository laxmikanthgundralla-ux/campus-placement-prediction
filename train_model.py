import os
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(
    BASE_DIR, "dataset", "student_placement.csv"
)
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "placement_model.pkl")


def train_model():
    # Create models folder if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load the dataset
    data = pd.read_csv(DATASET_PATH)

    # Features used for prediction
    feature_columns = [
        "CGPA",
        "Backlogs",
        "Internships",
        "Projects",
        "AptitudeScore",
        "CommunicationScore",
        "Certifications"
    ]

    target_column = "PlacementStatus"

    # Validate dataset
    required_columns = feature_columns + [target_column]

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing dataset columns: {missing_columns}"
        )

    if data.empty:
        raise ValueError("The dataset is empty.")

    if data[required_columns].isnull().any().any():
        raise ValueError(
            "The dataset contains missing values."
        )

    X = data[feature_columns]
    y = data[target_column]

    if y.nunique() < 2:
        raise ValueError(
            "The dataset must contain both placement classes: 0 and 1."
        )

    # Train the Random Forest model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X, y)

    # Save the trained model and its feature order
    joblib.dump(
        {
            "model": model,
            "features": feature_columns
        },
        MODEL_PATH
    )

    print("Model training completed successfully!")
    print(f"Model saved to: {MODEL_PATH}")

    # This is a development dataset, so this score is not
    # a measure of real-world placement prediction accuracy.
    print(
        f"Training samples: {len(X)}"
    )
    print(
        f"Classes found: {sorted(y.unique().tolist())}"
    )


if __name__ == "__main__":
    train_model()