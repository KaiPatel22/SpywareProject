from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline

import pandas as pd 

import sys

def timeSplit(df, labelCol = "label", timeCol = "windowStart", trainRatio=0.7):
    dfSorted = df.sort_values(timeCol).resetIndex(drop=True)

    splitIndex = int(len(dfSorted) * trainRatio)

    train = dfSorted.iloc[:splitIndex]
    test  = dfSorted.iloc[splitIndex:]

    X_train = train.drop(columns=[labelCol, "WindowID", "windowStart", "windowEnd"]).fillna(0)
    y_train = train[labelCol]

    X_test = train.drop(columns=[labelCol, "WindowID", "windowStart", "windowEnd"]).fillna(0)
    y_test = train[labelCol]

    print(f"Train set distribution: {X_train.value_counts()}")
    print(f"Test set distribution: {X_test.value_counts()}")

    return X_train, y_train, X_test, y_test

def trainRF(X_train, y_train, X_test, y_test):
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=42, shuffle=True, stratify=y)

    # print(f"Training set class distribution:\n{y_train.value_counts()}")

    # ros = RandomOverSampler(random_state=42)
    # X_train, y_train = ros.fit_resample(X_train, y_train)

    # print(f"Training set class distribution after oversampling:\n{y_train.value_counts()}")
    pipeline = Pipeline(steps =[
        ("ros",  RandomOverSampler(random_state=42)),
        ("rfc",  RandomForestClassifier(random_state=42))
    ])

    param_grid = {
        'rfc__n_estimators' : [100, 200, 400],
        'rfc__max_depth' : [None, 20, 40],
        'rfc__min_samples_split' : [2, 5, 10],
        'rfc__min_samples_leaf' : [1, 4, 10],
        'rfc__max_features': ['sqrt', 'log2', None],
        'rfc__criterion': ['gini', 'entropy']
    }

    # rfc = RandomForestClassifier(class_weight="balanced_subsample")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    model = GridSearchCV(pipeline, param_grid, n_jobs=-1, scoring='balanced_accuracy', cv=cv, verbose=3)

    model.fit(X_train, y_train)

    bestModel = model.best_estimator_
    
    y_pred = bestModel.predict(X_test)

    print(f"Best Parameters: {model.best_params_}")
    print(f"Best Balanced Accuracy: {model.best_score_:.3f}")

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.3f}")
    print(classification_report(y_test, y_pred))

    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred, labels=bestModel.classes_)}")

    return model

def trainXGBoost(X_train, y_train, X_test, y_test):
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=42, shuffle=True)

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
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

    return model


def trainSVM(X_train, y_train, X_test, y_test):
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=42, shuffle=True, stratify=y)
    pipeline = Pipeline(steps= [
        ("ros", RandomOverSampler(random_state=42)),
        ("scaler", StandardScaler()),
        ("svc", SVC(random_state=42))
    ])

    param_grid = {
        "svc__C": [0.1, 1, 10],
        "svc__gamma": ["scale", "auto", 0.01, 0.1],
        "svc__kernel": ["linear", "rbf"]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    model = GridSearchCV(pipeline, param_grid=param_grid, n_jobs = -1, scoring='balanced_accuracy', cv=cv, verbose=3)

    model.fit(X_train, y_train)
        
    y_pred = model.predict(X_test)

    print(f"Best Parameters: {model.best_params_}")
    print(f"Best Balanced Accuracy: {model.best_score_:.3f}")

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.3f}")
    print(classification_report(y_test, y_pred))

    return model




def main():
    df = pd.read_csv("/Users/kaipatel/Documents/SpywareProject/data/bulb2_windows_old.csv")
    X_train, y_train, X_test, y_test = timeSplit(df)
def main(csv):

    df = pd.read_csv(csv)
    X = df.drop(columns=["windowID", "windowStart", "windowEnd", "label"])
    y = df["label"]

    modelRF = trainRF(X_train, y_train, X_test, y_test)
    # modelXG = trainXGBoost(X, y)
    # modelSVM = trainSVM(X_train, y_train, X_test, y_test)


    importances = pd.Series(modelRF.best_estimator_.feature_importances_, index=X_train.columns)
    print(importances.sort_values(ascending=False))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage requires csv file path")
        sys.exit(1)
    else:
        csv = sys.argv[1]
        main(csv)