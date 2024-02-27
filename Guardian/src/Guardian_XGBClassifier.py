# from imblearn.over_sampling import SMOTE
# from imblearn.over_sampling import ADASYN
# from sklearn.model_selection import StratifiedKFold, GridSearchCV
# from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
# from sklearn.metrics import precision_recall_curve, auc, roc_curve, roc_auc_score
# from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
# import matplotlib.pyplot as plt
# from xgboost import XGBClassifier
# import pandas as pd


# # Load your dataset
# # train_df = pd.read_csv(
# #     "../data/guardian/knn_model/final_files/1476371065-100-train.csv"
# # )
# # test_df = pd.read_csv("../data/guardian/knn_model/final_files/1476371065-100.csv")

# train_df = pd.read_csv("../data/guardian/knn_model/final_files/1772246172-100.csv")
# test_df = pd.read_csv("../data/guardian/knn_model/test1772246172-100.csv")


# # Prepare your training and test sets
# X_train = train_df.iloc[:, 1:]  # Features
# y_train = train_df.iloc[:, 0].apply(
#     lambda x: 0 if x == "attack" else 1
# )  # Binary labels

# X_test = test_df.iloc[:, 1:]  # Features
# y_test = test_df.iloc[:, 0].apply(lambda x: 0 if x == "attack" else 1)  # Binary labels

# # Apply ADASYN to oversample the minority class in the training data
# adasyn = ADASYN(random_state=42)
# X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train, y_train)
# X_test_adasyn, y_test_adasyn = adasyn.fit_resample(X_test, y_test)
# # Define a stratified K-fold cross-validator
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# # Parameter grid for XGBoost
# param_grid = {
#     "max_depth": [8],  # [4, 6, 8],
#     "min_child_weight": [10],  # [1, 5, 10],
#     "gamma": [1.5],  # [1, 1.5, 2],
#     "subsample": [0.1],  # [0.5, 0.8, 1.0],
#     "colsample_bytree": [0.5],  # [0.6, 0.8, 1.0],
#     "n_estimators": [200],  # [100, 200, 300],
#     "learning_rate": [0.0001],  # [0.01, 0.1, 0.2],
# }

# model = XGBClassifier(
#     use_label_encoder=False, eval_metric="logloss", objective="binary:logistic"
# )

# # Set up GridSearchCV or RandomizedSearchCV
# clf = GridSearchCV(model, param_grid, scoring="roc_auc", n_jobs=-1, cv=cv, verbose=3)

# # Train the model with the ADASYN-enhanced dataset using the best parameters found
# clf.fit(X_train_adasyn, y_train_adasyn)

# # Best model after CV
# best_model = clf.best_estimator_
# print("Best parameters:", best_model)

# # Predict on the test set with the best model
# y_pred = best_model.predict(X_test_adasyn)
# y_pred_proba = best_model.predict_proba(X_test_adasyn)[:, 1]
# # Evaluate the model
# print("Best Parameters:", clf.best_params_)
# print("Accuracy:", accuracy_score(y_test_adasyn, y_pred))
# print("Confusion Matrix:")
# print(confusion_matrix(y_test_adasyn, y_pred))
# print("Classification Report:")
# print(classification_report(y_test_adasyn, y_pred))

# # Precision-Recall Curve
# precision, recall, _ = precision_recall_curve(y_test_adasyn, y_pred_proba)
# pr_auc = auc(recall, precision)
# PrecisionRecallDisplay(precision=precision, recall=recall).plot()
# plt.title(f"Precision-Recall curve: AUC={pr_auc:.2f}")
# plt.show()

# # ROC-AUC Curve
# fpr, tpr, _ = roc_curve(y_test_adasyn, y_pred_proba)
# roc_auc = roc_auc_score(y_test_adasyn, y_pred_proba)
# RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name="XGBoost").plot()
# plt.title(f"ROC curve: AUC={roc_auc:.2f}")
# plt.show()


import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

# Load training dataset
data = pd.read_csv("../data/guardian/knn_model/final_files/1476371065-100-train.csv")
X_train = data.iloc[:, 1:]
y_train = data.iloc[:, 0:1].values.ravel()

# Load the test dataset
test_data = pd.read_csv("path_to_your_test_data.csv")
X_test = test_data.iloc[:, 1:].values
y_test = test_data.iloc[:, 0:1].values.ravel()


# Initialize StratifiedKFold for cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Placeholder for model evaluation to find the best K
best_score = 0
best_k = 1

# Trying different values of K to find the best one
for k in range(1, 26, 2):  # Example: Trying odd values of K from 1 to 25
    scores = []
    for train_index, val_index in skf.split(X, y):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Handling imbalanced data with SMOTE
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        # Standardize features
        scaler = StandardScaler()
        X_train_res_scaled = scaler.fit_transform(X_train_res)
        X_val_scaled = scaler.transform(X_val)

        # Initialize and train KNN model
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_res_scaled, y_train_res)

        # Validate the model
        score = knn.score(X_val_scaled, y_val)
        scores.append(score)

    # Average score for this K
    average_score = sum(scores) / len(scores)
    if average_score > best_score:
        best_score = average_score
        best_k = k

# Now that we have the best K, train on the entire training dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

scaler = StandardScaler()
X_resampled_scaled = scaler.fit_transform(X_resampled)
X_test_scaled = scaler.transform(X_test)

# Train with the best K found
knn_final = KNeighborsClassifier(n_neighbors=best_k)
knn_final.fit(X_resampled_scaled, y_resampled)

# Predict on the separate test set
predictions = knn_final.predict(X_test_scaled)

# Evaluate the model on the test set
print(f"Best K: {best_k}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions))
