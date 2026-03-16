from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, balanced_accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline

import pandas as pd 

def trainRF(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=42, shuffle=True, stratify=y)

    print(f"Training set class distribution:\n{y_train.value_counts()}")

    ros = RandomOverSampler(random_state=42)
    X_train, y_train = ros.fit_resample(X_train, y_train)

    print(f"Training set class distribution after oversampling:\n{y_train.value_counts()}")
    pipeline = Pipeline(steps =[
        ("ros",  RandomOverSampler(random_state=42)),
        ("rfc",  RandomForestClassifier(random_state=42))
    ])

    param_grid = {
        'rfc__n_estimators' : [200, 400],
        'rfc__max_depth' : [None, 20, 40],
        'rfc__min_samples_split' : [2, 10],
        'rfc__min_samples_leaf' : [1, 4],
    }

    # rfc = RandomForestClassifier(class_weight="balanced_subsample")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    model = GridSearchCV(pipeline, param_grid, n_jobs = -1, scoring='balanced_accuracy', cv=cv, verbose=3)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(f"Best Parameters: {model.best_params_}")
    print(f"Best Balanced Accuracy: {model.best_score_:.3f}")

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.3f}")
    print(classification_report(y_test, y_pred))

    return model

def trainXGBoost(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=42, shuffle=True)

    param_grid = {
        'learning_rate' : [0.01, 0.1, 0.2],
        'n_estimators' : [100, 200, 300],
        'min_samples_split' : [2, 5, 7],
        'min_samples_leaf' : [1, 2, 4],
        'max_depth' : [None, 3, 5, 7],
    }

    gbc = GradientBoostingClassifier()

    model = GridSearchCV(gbc, param_grid=param_grid, scoring='balanced_accuracy', cv=5, verbose=3)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.3f}")
    print(classification_report(y_test, y_pred))

    return model

def main():
    df = pd.read_csv("/Users/kaipatel/Documents/SpywareProject/data/bulb2_windows_old.csv")
    X = df.drop(columns=["windowID", "windowStart", "windowEnd", "label"])
    y = df["label"]

    X = X.fillna(0)

    model = trainRF(X, y)
    # model = trainXGBoost(X, y)


    importances = pd.Series(model.best_estimator_.feature_importances_, index=X.columns)
    print(importances.sort_values(ascending=False))

if __name__ == "__main__":
    main()