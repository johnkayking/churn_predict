import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
# from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression


def load_train_test_data(folder='data'):
    X_train = pd.read_csv(os.path.join(folder, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(folder, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(folder, 'y_train.csv'))
    y_test = pd.read_csv(os.path.join(folder, 'y_test.csv'))

    # Convert single-column DataFrames to Series
    if y_train.shape[1] == 1:
        y_train = y_train.iloc[:, 0]
    if y_test.shape[1] == 1:
        y_test = y_test.iloc[:, 0]

    return X_train, X_test, y_train, y_test


def get_test_predictions(model, X_test):
    """Returns predicted labels"""
    return model.predict(X_test)


def plot_confusion_matrix(y_train
                          , y_pred, labels=None, title='Confusion Matrix'):
    cm = confusion_matrix(y_train, y_pred, labels=labels)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels if labels else 'auto',
                yticklabels=labels if labels else 'auto')
    plt.xlabel('Predicted Label')
    plt.ylabel('actual Label')
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ======= MAIN SCRIPT =======

# Load data
X_train, X_test, y_train, y_test = load_train_test_data(folder='saved_data')

print("Train shape:", X_train.shape, y_train.shape)
print("Test shape:", X_test.shape, y_test.shape)


LR = LogisticRegression(class_weight=None)
LR.fit(X_train,y_train)

y_pred = get_test_predictions(LR,X_test)

plot_confusion_matrix(y_test,y_pred, title='LogisticRegression Confusion Matrix')

print("Classification Report:\n")
print(classification_report(y_test, y_pred))




# Train AdaBoost with DecisionTree as base estimator
Ada_boost = AdaBoostClassifier(
    n_estimators=100,
    random_state=45,
    # base_estimator=DecisionTreeClassifier(max_depth=1)
)
Ada_boost.fit(X_train, y_train)

# Get predictions
y_pred = get_test_predictions(Ada_boost, X_test)

# Plot confusion matrix
plot_confusion_matrix(y_test, y_pred, title='AdaBoost Confusion Matrix')

# Print classification report
print("Classification Report:\n")
print(classification_report(y_test, y_pred))





model = GradientBoostingClassifier(n_estimators=200,random_state=34)
model.fit(X_train,y_train)

y_pred = get_test_predictions(model,X_test)

plot_confusion_matrix(y_test,y_pred, title='GradientBoosting Confusion Matrix')

print("Classification Report:\n")
print(classification_report(y_test, y_pred))





# model = XGBClassifier()
# model.fit(X_train,y_train)

# y_pred = get_test_predictions(model,X_test)


# plot_confusion_matrix(y_test,y_pred, title='XGBoost Confusion Matrix') 

# print("Classification Report:\n")
# print(classification_report(y_test, y_pred))


import pickle
# Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# import joblib

# joblib.dump((model, X_test.columns.tolist()), "xgb_model.pkl")
# # later
# model, X_test = joblib.load("xgb_model.pkl")
