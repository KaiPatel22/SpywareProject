from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, TimeSeriesSplit
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline

# import lightgbm as lgb

from xgboost import XGBClassifier

import pandas as pd 

import sys

def timeSplit(df, labelCol = "label", timeCol = "windowStart", trainRatio=0.7, excludeLabels=None):
    if excludeLabels:
        df = df[~df[labelCol].isin(excludeLabels)]
    dfSorted = df.sort_values(timeCol).reset_index(drop=True)

    splitIndex = int(len(dfSorted) * trainRatio)

    train = dfSorted.iloc[:splitIndex]
    test  = dfSorted.iloc[splitIndex:]

    X_train = train.drop(columns=[labelCol, "windowID", "windowStart", "windowEnd"]).fillna(0)
    y_train = train[labelCol]

    X_test = test.drop(columns=[labelCol, "windowID", "windowStart", "windowEnd"]).fillna(0)
    y_test = test[labelCol]

    print(f"Train set distribution: {y_train.value_counts()}")
    print(f"Test set distribution: {y_test.value_counts()}")

    return X_train, y_train, X_test, y_test

def trainRF(X_train, y_train, X_test, y_test):
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=42, shuffle=True, stratify=y)

    # print(f"Training set class distribution:\n{y_train.value_counts()}")

    # ros = RandomOverSampler(random_state=42)
    # X_train, y_train = ros.fit_resample(X_train, y_train)

    # print(f"Training set class distribution after oversampling:\n{y_train.value_counts()}")
    pipeline = Pipeline(steps =[
        ("ros",  RandomOverSampler(random_state=42)),
        ("rfc",  RandomForestClassifier(random_state=42, n_jobs=-1, oob_score=True))
    ])

    param_grid = {
        'rfc__n_estimators' : [200, 400],
        'rfc__max_depth' : [None, 20],
        'rfc__min_samples_leaf' : [1, 4],
        'rfc__max_features': ['sqrt', 'log2'],
    }

    # rfc = RandomForestClassifier(class_weight="balanced_subsample")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # cv = TimeSeriesSplit(n_splits=3, gap=5)

    model = GridSearchCV(pipeline, param_grid, n_jobs=-1, scoring={'acc': "accuracy",'bal_acc': 'balanced_accuracy','f1_macro': "f1_macro"}, refit="bal_acc", cv=cv, verbose=3)

    model.fit(X_train, y_train)

    bestModel = model.best_estimator_
    y_pred = bestModel.predict(X_test)

    print(f"Best Parameters: {model.best_params_}")
    print(f"Best Balanced Accuracy: {model.best_score_:.3f}")

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.3f}")
    print(classification_report(y_test, y_pred))

    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred, labels=bestModel.classes_)}")

    importances = pd.Series(bestModel.named_steps['rfc'].feature_importances_, index=X_train.columns)
    print(importances.sort_values(ascending=False))

    return bestModel

def trainGBoost(X_train, y_train, X_test, y_test):
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


    bestModel = model.best_estimator_
    y_pred = bestModel.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.3f}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

    # importances = pd.Series(bestModel.feature_importances_, index=X_train.columns)
    # print(importances.sort_values(ascending=False))

    return bestModel


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
        
    bestModel = model.best_estimator_
    y_pred = bestModel.predict(X_test)

    print(f"Best Parameters: {model.best_params_}")
    print(f"Best Balanced Accuracy: {model.best_score_:.3f}")

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.3f}")
    print(classification_report(y_test, y_pred))

    importances = pd.Series(bestModel.best_estimator_.feature_importances_, index=X_train.columns)
    print(importances.sort_values(ascending=False))

    return bestModel

def trainLGBM(X_train, y_train, X_test, y_test):
    pipeline = Pipeline(steps = [
        # ("ros", RandomOverSampler(random_state=42)),
        ("lgbm", lgb.LGBMClassifier(verbose=-1, class_weight="balanced", random_state=42))
    ])

    param_grid = {
        "lgbm__n_estimators": [200, 400, 800],
        "lgbm__learning_rate": [0.01, 0.1],
        "lgbm__max_depth": [5, 10, -1],
        "lgbm__min_child_samples": [10, 20, 50],
        "lgbm__colsample_bytree": [0.7, 1.0],
        "lgbm__subsample": [0.7, 1.0]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    model = GridSearchCV(pipeline, param_grid=param_grid, n_jobs=-1, scoring='balanced_accuracy', cv=cv, verbose=3)
    model.fit(X_train, y_train)

    bestModel = model.best_estimator_
    y_pred = bestModel.predict(X_test)

    print(f"Best Parameters: {model.best_params_}")
    print(f"Best Balanced Accuracy: {model.best_score_:.3f}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.3f}")
    print(classification_report(y_test, y_pred))

    importances = pd.Series(bestModel.best_estimator_.feature_importances_, index=X_train.columns)
    print(importances.sort_values(ascending=False))

    return bestModel

def trainXGBoost(X_train, y_train, X_test, y_test):
    labelEncoder = LabelEncoder()

    y_train_encoded = labelEncoder.fit_transform(y_train)
    y_test_encoded = labelEncoder.transform(y_test)

    pipeline = Pipeline(steps = [
        ("ros", RandomOverSampler(random_state=42)),
        ("xgb", XGBClassifier(random_state=42, eval_metric="mlogloss", verbosity=0))
    ])

    param_grid = {
        "xgb__n_estimators": [200, 400, 800],
        "xgb__learning_rate": [0.01, 0.1],
        "xgb__max_depth": [5, 10, -1],
        "xgb__min_child_weight": [1, 5, 10],
        "xgb__colsample_bytree": [0.7, 1.0],
        "xgb__subsample": [0.7, 1.0]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    model = GridSearchCV(pipeline, param_grid=param_grid, n_jobs=-1, scoring='balanced_accuracy', cv=cv, verbose=3)

    model.fit(X_train, y_train_encoded)
    bestModel = model.best_estimator_

    y_pred_encoded = bestModel.predict(X_test)
    y_pred = labelEncoder.inverse_transform(y_pred_encoded)

    print(f"Best Parameters: {model.best_params_}")
    print(f"Best Balanced Accuracy: {model.best_score_:.3f}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.3f}")
    print(classification_report(y_test, y_pred))

    importances = pd.Series(bestModel.best_estimator_.feature_importances_, index=X_train.columns)
    print(importances.sort_values(ascending=False))

    return bestModel



def main(csv):

    df = pd.read_csv(csv)
    # X_train, y_train, X_test, y_test = timeSplit(df, excludeLabels=["outsidedetection"])
    df = df[~df["label"].isin(["outsidedetection"])]
    X = df.drop(columns=["label","windowStart", "windowEnd", "windowID"]).fillna(0)
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, shuffle=True, stratify=y)

    modelRF = trainRF(X_train, y_train, X_test, y_test)
    # modelXG = trainGBoost(X, y)
    # modelSVM = trainSVM(X_train, y_train, X_test, y_test)
    # modelLGBM = trainLGBM(X_train, y_train, X_test, y_test)
    # modelXGB = trainXGBoost(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage requires csv file path")
        sys.exit(1)
    else:
        csv = sys.argv[1]
        main(csv)