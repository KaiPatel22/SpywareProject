import sys
import joblib

import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.svm import SVC 
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB


from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline
from imblearn.ensemble import BalancedRandomForestClassifier

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

def createLogisticRegression(X_train, X_test, y_train, y_test):
    pipeline = Pipeline(steps = [
        ("ros", SMOTE(random_state=42)),
        ("lr", LogisticRegression(random_state=42, n_jobs=-1))
    ])

    param_grid = {
        "lr__class_weight": [None, "balanced"],
        "lr__solver": ["lbfgs", "newton-cholesky", "saga"],
        "lr__max_iter": [400, 500, 700]
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


def createRandomForest(X_train, X_test, y_train, y_test):
    pipeline = Pipeline(steps = [
        ("ros", SMOTE(random_state=42)),
        ("rfc", RandomForestClassifier(random_state=42, n_jobs=-1))
    ])

    param_grid = {
        "rfc__n_estimators": [200, 400],
        "rfc__max_depth": [None, 20],
        "rfc__min_samples_leaf": [2, 5],
        "rfc__class_weight": ["balanced", "balanced_subsample"],
        # "rfc__max_features": ["sqrt", "log2"]
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

    model = GridSearchCV(BalancedRandomForestClassifier(random_state=42, n_jobs=-1), param_grid, scoring={"acc": "accuracy", "bal_acc": "balanced_accuracy", "f1_macro": "f1_macro"}, refit="bal_acc", n_jobs=-1, cv=5, verbose=3)
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

def createXGBoost(X_train, X_test, y_train, y_test):

    labelEncoder = LabelEncoder()
    y_train = labelEncoder.fit_transform(y_train)
    y_test = labelEncoder.transform(y_test)

    pipeline = Pipeline(steps = [
        ("ros", SMOTE(random_state=42)),
        ("xgb", XGBClassifier(random_state=42, n_jobs=-1))
    ])

    param_grid = {
        "xgb__n_estimators": [200, 400],
        "xgb__max_depth": [0, 6, 10],
        "xgb__learning_rate": [0.05, 0.1],
    }
    
    model = GridSearchCV(pipeline, param_grid, scoring={"acc": "accuracy", "bal_acc": "balanced_accuracy", "f1_macro": "f1_macro"}, refit="bal_acc", n_jobs=-1, cv=5, verbose=3)
    model.fit(X_train, y_train)
    bestModel = model.best_estimator_
    y_pred = bestModel.predict(X_test)
    y_pred = labelEncoder.inverse_transform(y_pred)
    y_test = labelEncoder.inverse_transform(y_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.3f}")
    print(f"F1 Score: {f1_score(y_test, y_pred, average='macro'):.3f}")
    print(f"Classification Report:\n {classification_report(y_test, y_pred)}")
    print(f"Confusion Matrix:\n {confusion_matrix(y_test, y_pred, labels=bestModel.classes_)}")

    print("Best Hyperparameters: ", model.best_params_)

    return bestModel

def createLGBM(X_train, X_test, y_train, y_test):
    pipeline = Pipeline(steps = [
        ("ros", SMOTE(random_state=42)),
        ("lgbm", LGBMClassifier(random_state=42, n_jobs=-1))
    ])

    param_grid = {
        "lgbm__n_estimators": [200, 400],
        "lgbm__max_depth": [0, 6, 10],
        "lgbm__learning_rate": [0.05, 0.1, 0.2],
        "lgbm__num_leaves":[31, 50]
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

def createSVM(X_train, X_test, y_train, y_test):
    pipeline = Pipeline(steps = [
        ("ss", RobustScaler()),
        ("ros", SMOTE(random_state=42)),
        ("svc", SVC(random_state=42))
    ])

    param_grid = {
        "svc__C": [0.5, 1, 5],
        "svc__kernel": ["rbf", "linear"],
        "svc__gamma": ["scale", "auto"]
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

# def createMLP(X_train, X_test, y_train, y_test):
#     pipeline = Pipeline(steps = [
#         ("ss", StandardScaler()),
#         ("ros", SMOTE(random_state=42)),
#         ("mlp", MLPClassifier(random_state=42, early_stopping=True))
#     ])

#     param_grid = {
#         "mlp__hidden_layer_sizes": [(128, 64), (256, 128, 64)],
#         "mlp__learning_rate_init": [0.001, 0.005],
#         "mlp__alpha": [0.0001, 0.001],
#         "mlp__solver": ["adam", "sgd"]
#     }

#     model = GridSearchCV(pipeline, param_grid, scoring={"acc": "accuracy", "bal_acc": "balanced_accuracy", "f1_macro": "f1_macro"}, refit="bal_acc", n_jobs=-1, cv=5, verbose=3)
#     model.fit(X_train, y_train)
#     bestModel = model.best_estimator_
#     y_pred = bestModel.predict(X_test)

#     print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
#     print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.3f}")
#     print(f"F1 Score: {f1_score(y_test, y_pred, average='macro'):.3f}")
#     print(f"Classification Report:\n {classification_report(y_test, y_pred)}")
#     print(f"Confusion Matrix:\n {confusion_matrix(y_test, y_pred, labels=bestModel.classes_)}")

#     print("Best Hyperparameters: ", model.best_params_)

#     return bestModel

def createNaivesBayes(X_train, X_test, y_train, y_test):
    pipeline = Pipeline(steps = [
        ("ros", SMOTE(random_state=42)),
        ("gnb", GaussianNB())
    ])

    param_grid = {
        "gnb__var_smoothing": [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7]
    }

    model = GridSearchCV(pipeline, param_grid, scoring={"acc": "accuracy", "bal_acc": "balanced_accuracy", "f1_macro": "f1_macro"}, refit="bal_acc", n_jobs=-1, cv=10, verbose=3)
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

    # X = df.drop(columns=["windowID","windowStart","windowEnd","tcpRatio","udpPacketCount","udpRatio","stdIPLen","stdTCPLen","uniqueTCPStreams","minTCPWindowSize","maxTCPWindowSize","synCount","finCount","uniqueUDPSrcPorts","uniqueUDPDstPorts","minInterArrivalTime", "tlsHandshakeCount","minTLSRecordLen","maxTLSRecordLen","stdACKRoundTripTime","minACKRoundTripTime","maxACKRoundTripTime","ACKRoundTripTimeCount","minTimeDelta","maxTimeDelta","tlsContentTypeChanegCipherCount","tlsContentTypeAlertCount","tlsContentTypeHandshakeCount","tlsContentTypeAppDataCount","label"]).fillna(0)

    mergeEvents = {
        "bulbOn" : "bulbEvent",
        "bulbOff" : "bulbEvent",
        "bulbChange" : "bulbEvent",
        "alexaBulbOn" : "alexaBulbEvent",
        "alexaBulbOff" : "alexaBulbEvent",
        "alexaBulbChange" : "alexaBulbEvent",
    }

    y = df["label"].replace(mergeEvents)

    # y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=42, shuffle=True, stratify=y)
    
    print(f"Train set distribution: {y_train.value_counts()}")
    print(f"Test set distribution: {y_test.value_counts()}")

    # modelLR = createLogisticRegression(X_train, X_test, y_train, y_test)
    # joblib.dump(modelLR, "models/modelLR.pkl")

    # modelRF = createRandomForest(X_train, X_test, y_train, y_test)
    # joblib.dump(modelRF, "models/modelRF.pkl")

    # modelBRF = createBalancedRandomForest(X_train, X_test, y_train, y_test)
    # joblib.dump(modelBRF, "models/modelBRF.pkl")

    # modelXGB = createXGBoost(X_train, X_test, y_train, y_test)
    # joblib.dump(modelXGB, "models/modelXGB.pkl")

    # modelLGBM = createLGBM(X_train, X_test, y_train, y_test)
    # joblib.dump(modelLGBM, "models/modelLGBM.pkl")

    # modelSVM = createSVM(X_train, X_test, y_train, y_test)
    # joblib.dump(modelSVM, "models/modelSVM.pkl")

    modelGNB = createNaivesBayes(X_train, X_test, y_train, y_test)
    joblib.dump(modelGNB, "models/modelGNB.pkl")
