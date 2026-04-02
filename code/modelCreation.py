import sys

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline
from imblearn.ensemble import BalancedRandomForestClassifier

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report, confusion_matrix


def createRandomForest(X_train, X_test, y_train, y_test):
    pipeline = Pipeline(steps = [
        ("ros", RandomOverSampler(random_state=42)),
        ("rfc", RandomForestClassifier(random_state=42, n_jobs=-1))
    ])

    param_grid = {
        "rfc__n_estimators": [200, 400],
        "rfc__max_depth": [None, 20],
        "rfc__min_samples_leaf": [2, 5],
    }

    model = GridSearchCV(pipeline, param_grid, scoring={"acc": "accuracy", "bal_acc": "balanced_accuracy", "f1_macro": "f1_macro"}, refit="bal_acc", n_jobs=-1, cv=5, verbose=3)
    model.fit(X_train, y_train)
    bestModel = model.best_estimator_
    y_pred = bestModel.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.3f}")
    print(f"F1 Score: {f1_score(y_test, y_pred, average='macro'):.3f}")
    print(f"Classification Report:\n {classification_report(y_test, y_pred)}")
    print(f"Confusion Matrix:\n {confusion_matrix(y_test, y_pred, labels=bestModel.classes_)}")

    print("Best Hyperparameters: ", model.best_params_)

    return bestModel

def createBalancedRandomForest(X_train, X_test, y_train, y_test):
    
    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [None, 20],
        "min_samples_leaf": [2, 5],
    }

    model = GridSearchCV(BalancedRandomForestClassifier(random_state=42, n_jobs=-1), param_grid, scoring={"acc": "accuracy", "bal_acc": "balanced_accuracy", "f1_macro": "f1_macro"}, refit="bal_acc", n_jobs=-1, cv=10, verbose=3)
    model.fit(X_train, y_train)
    bestModel = model.best_estimator_
    y_pred = bestModel.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.3f}")
    print(f"F1 Score: {f1_score(y_test, y_pred, average='macro'):.3f}")
    print(f"Classification Report:\n {classification_report(y_test, y_pred)}")
    print(f"Confusion Matrix:\n {confusion_matrix(y_test, y_pred, labels=bestModel.classes_)}")

    print("Best Hyperparameters: ", model.best_params_)

    return bestModel

if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        print("Usage: python code/modelCreation2.py <csvFile>")
        sys.exit(1)
    
    csvFile = sys.argv[1]
    
    df = pd.read_csv(csvFile)
    X = df.drop(columns=["label", "windowStart", "windowEnd", "windowID"]).fillna(0)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=42, shuffle=True)
    
    print(f"Train set distribution: {y_train.value_counts()}")
    print(f"Test set distribution: {y_test.value_counts()}")

    # modelRF = createRandomForest(X_train, X_test, y_train, y_test)
    # modelBRF = createBalancedRandomForest(X_train, X_test, y_train, y_test)