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

'''
This function is for creating a logisitic regression model
A pipeline is used to oversample the data and then create the model object. A parameter grid is created and grid search is used to find the best hyperparameters, refitting using balanced accuracy. 
Evaluation metrics are printed and the best model is returned. 
'''
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

'''
This function is for creating a random forest classifier.
A pipeline is used to oversample the data and then create the model object. A parameter grid is created and grid search is used to find the best hyperparameters, refitting using balanced accuracy. 
Evaluation metrics are printed and the best model is returned. 
'''
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
        "rfc__max_features": ["sqrt", "log2"]
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

'''
This function is for creating a balanced random forest classifier.
A parameter grid is created and grid search is used to find the best hyperparameters, refitting using balanced accuracy (pipelines cannot be used as this model came from imblearn and not sklearn).
Evaluation metrics are printed and the best model is returned. 
'''
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

'''
This function is for creating a XGBoost classifier.
A label encoder is first used to turn the labels into numerical integers and a pipeline is used to oversample the data and then create the model object. A parameter grid is created and grid search is used to find the best hyperparameters, refitting using balanced accuracy. 
Evaluation metrics are printed and the best model is returned. 
'''
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

'''
This function is for creating a SVM.
A pipeline is used to first scale the features before oversampling the data and then creating the model object. A parameter grid is created and grid search is used to find the best hyperparameters, refitting using balanced accuracy. 
Evaluation metrics are printed and the best model is returned. 
'''
def createSVM(X_train, X_test, y_train, y_test):
    pipeline = Pipeline(steps = [
        ("ss", RobustScaler()),
        ("ros", SMOTE(random_state=42)),
        ("svc", SVC(random_state=42))
    ])

    param_grid = {
        "svc__C": [0.5, 1],
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

'''
This function is for creating a Naive Bayes model.
A pipeline is used to first scale the features before oversampling the data and then creating the model object. A parameter grid is created and grid search is used to find the best hyperparameters, refitting using balanced accuracy. 
Evaluation metrics are printed and the best model is returned. 
'''
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

'''
Main method 

In the X column, drops the window information as this provides no meaningful information and drops the label column as this is the target variable. Fills any missing values with 0 as a last check to ensure there are no null values. 
In the y column, two options, the first uses the normal labels however the second merges the bulb events into one label to allow for the model to focus on classying when a device is active over what specific activity is occuring. 
Creates the train test split, shuffling the data and stratifying based on the distribution of labels to ensure the train and test splits have similar distributions. 
Code can be commented and uncommented to create different models. 
All model are saved into a seperate folder as .pkl files.
'''
if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        print("Usage: python code/modelCreation2.py <csvFile>")
        sys.exit(1)
    
    csvFile = sys.argv[1]
    
    df = pd.read_csv(csvFile)
    X = df.drop(columns=["label", "windowStart", "windowEnd", "windowID"]).fillna(0)

    # X = df.drop(columns=["windowID","windowStart","windowEnd","tcpRatio","udpPacketCount","udpRatio","stdIPLen","stdTCPLen","uniqueTCPStreams","minTCPWindowSize","maxTCPWindowSize","synCount","finCount","uniqueUDPSrcPorts","uniqueUDPDstPorts","minInterArrivalTime", "tlsHandshakeCount","minTLSRecordLen","maxTLSRecordLen","stdACKRoundTripTime","minACKRoundTripTime","maxACKRoundTripTime","ACKRoundTripTimeCount","minTimeDelta","maxTimeDelta","tlsContentTypeChanegCipherCount","tlsContentTypeAlertCount","tlsContentTypeHandshakeCount","tlsContentTypeAppDataCount","label"]).fillna(0)

    # mergeEvents = {
    #     "bulbOn" : "bulbEvent",
    #     "bulbOff" : "bulbEvent",
    #     "bulbChange" : "bulbEvent",
    #     "alexaBulbOn" : "alexaBulbEvent",
    #     "alexaBulbOff" : "alexaBulbEvent",
    #     "alexaBulbChange" : "alexaBulbEvent",
    # }

    # y = df["label"].replace(mergeEvents)

    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=42, shuffle=True, stratify=y)
    
    print(f"Train set distribution: {y_train.value_counts()}")
    print(f"Test set distribution: {y_test.value_counts()}")

    # modelLR = createLogisticRegression(X_train, X_test, y_train, y_test)
    # joblib.dump(modelLR, "models/modelLR.pkl")

    # modelRF = createRandomForest(X_train, X_test, y_train, y_test)
    # joblib.dump(modelRF, "models/modelRF.pkl")

    # modelBRF = createBalancedRandomForest(X_train, X_test, y_train, y_test)
    # joblib.dump(modelBRF, "models/modelBRF.pkl")

    modelXGB = createXGBoost(X_train, X_test, y_train, y_test)
    joblib.dump(modelXGB, "models/modelXGB.pkl")

    # modelSVM = createSVM(X_train, X_test, y_train, y_test)
    # joblib.dump(modelSVM, "models/modelSVM.pkl")

    # modelGNB = createNaivesBayes(X_train, X_test, y_train, y_test)
    # joblib.dump(modelGNB, "models/modelGNB.pkl")
