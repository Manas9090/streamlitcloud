from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import pandas as pd

if __name__ == "__main__":
    # Step 2: Load Data
    iris = load_iris()
    X = iris.data  # Features
    y = iris.target  # Labels

    # Step 3: Preprocess Data (Split into train and test sets)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Step 4: Train Model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Step 5: Evaluate Model
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print("Classification Report:\n", report)

    joblib.dump(clf, '/opt/ml/processing/output/train/clf.pkl')