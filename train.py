import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split


def train_model():
    # Step 1: Load dataset with feature metadata
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target

    # Step 2: Split dataset with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Step 3: Instantiate model
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Step 4: Fit model
    model.fit(X_train, y_train)

    # Step 5: Evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(
        y_test, predictions, target_names=iris.target_names
    )

    # Package model alongside critical metadata
    artifact = {
        "model": model,
        "feature_names": iris.feature_names,
        "target_names": iris.target_names.tolist(),
    }

    return artifact, X_test, y_test, accuracy, report


if __name__ == "__main__":
    artifact, X_test, y_test, accuracy, report = train_model()

    # Save model and metadata together
    joblib.dump(artifact, "model.pkl")

    print("Model training completed successfully.")
    print("I am Bangladeshi")
    print(f"Test Accuracy: {accuracy:.2%}\n")
    print("Classification Report:")
    print(report)
    print("Artifact saved as model.pkl")