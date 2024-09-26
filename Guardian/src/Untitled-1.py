# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from imblearn.over_sampling import SMOTE

# # Load dataset
# df = pd.read_csv(
#     "H:/1.Deakin university/Python/13_10_2023_My_Project_1/guardian_paper_my_version/data/guardian/knn_model/final_files/1772246172-100.csv"
# )

# # Encode labels
# label_encoder = LabelEncoder()
# df["type"] = label_encoder.fit_transform(
#     df["type"]
# )  # Assuming 'label' is the column name

# # Separate features and target
# X = df.drop("type", axis=1)
# y = df["type"]

# # Handling imbalance
# smote = SMOTE()
# X_resampled, y_resampled = smote.fit_resample(X, y)

# # Feature scaling
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X_resampled)

# # Splitting dataset for evaluation
# X_train, X_test, y_train, y_test = train_test_split(
#     X_scaled, y_resampled, test_size=0.2, random_state=42
# )

# # Model training
# model = LogisticRegression()
# model.fit(X_train, y_train)

# # Predict probabilities
# y_probs = model.predict_proba(X_test)[:, 1]

# # Determine the optimal threshold
# precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
# # Calculate scores for various thresholds
# fscore = (2 * precision * recall) / (precision + recall)
# # Locate the index of the largest f score
# ix = np.argmax(fscore)
# print(f"Best Threshold={thresholds[ix]}, F-Score={fscore[ix]}")

# # Adjust classification threshold
# y_pred = [1 if prob > thresholds[ix] else 0 for prob in y_probs]

# # Evaluation
# print(classification_report(y_test, y_pred))
# print(f"AUC-ROC: {roc_auc_score(y_test, y_pred)}")

# # Ensure new_data is in the correct format and preprocess
# new_data = pd.read_csv(
#     "H:/1.Deakin university/Python/13_10_2023_My_Project_1/guardian_paper_my_version/data/guardian/knn_model/test_9050782635-200.csv"
# )
# new_data_scaled = scaler.transform(new_data)

# # Predict probabilities
# new_probs = model.predict_proba(new_data_scaled)[:, 1]

# # Apply threshold to determine classifications
# new_predictions = [
#     "attack" if prob > thresholds[ix] else "normal" for prob in new_probs
# ]

# print(new_predictions)


##
##
##    Tesing Model Results for KNN (Save the CNN results into csv files, and prepare for training KNN)
##
##    True/False Positives/Negatives AND save results to file ../data/knn_file/
##
##    file name 5651345512-100_users_1000_test_1-10_checkpoint.csv ==> 'name_training'_'dataset_name'_'deep_speaker_test_id'.csv,
##
##    Modify ALL parameter below
##
##    Test outcome positive(Attacked) Actually condition positive(Attacked) ==> TP
##    Test outcome positive(Attacked) Actually condition negative(Normal) ==> FP
##    Test outcome negative(Normal) Actually condition positive(Attacked) ==> FN
##    Test outcome negative(Normal) Actually condition negative(Normal) ==> TN
##
##
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

# from imblearn.over_sampling import SMOTE  # Import SMOTE

# # Load data
# data = pd.read_csv("../data/guardian/knn_model/6038796523-300-5.csv")

# # Split features and target
# x = data.iloc[:, 1:]
# y = data.iloc[:, 0:1].values.ravel()

# # Split dataset
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

# # Apply SMOTE to generate synthetic samples and balance the dataset
# smote = SMOTE(random_state=42)
# X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# # Number of neighbors
# n_neighbors =

# # Choose a distance metric from ['euclidean', 'manhattan', 'minkowski', 'chebyshev', 'hamming', 'cosine']
# # Note: For Minkowski, you can also specify the `p` parameter, e.g., p=1 for Manhattan distance
# distance_metric = "manhattan"

# # Initialize KNN with the chosen metric
# neigh = KNeighborsClassifier(
#     n_neighbors=n_neighbors, weights="distance", metric=distance_metric
# )

# # Fit model
# neigh.fit(X_train_resampled, y_train_resampled)

# data2 = pd.read_csv("../data/guardian/knn_model/test_6038796523-300-5.csv")
# x2 = data2.iloc[:, 1:]
# y2 = data2.iloc[:, 0:1]

# y_pred = neigh.predict(x2)
# print(confusion_matrix(y2, y_pred))
# # Use zero_division parameter to handle undefined metrics warning
# print(classification_report(y2, y_pred, zero_division=1))
# from sklearn.metrics import confusion_matrix
# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import (
#     classification_report,
#     accuracy_score,
#     recall_score,
#     precision_score,
#     f1_score,
# )
# from imblearn.over_sampling import SMOTE
# from imblearn.over_sampling import SMOTE
# from sklearn.preprocessing import StandardScaler
# from functools import reduce


# # Function to aggregate rows for each person
# def aggregate_rows(df, group_size=10):
#     # Assuming the first column is 'Label' and the rest are feature columns
#     aggregated_features = []
#     for start_row in range(0, df.shape[0], group_size):
#         group_df = df.iloc[start_row : start_row + group_size]
#         # Binary indicator for values > 0.6
#         binary_indicators = (group_df.iloc[:, 1:] > 0.15).astype(int)
#         aggregated_row = binary_indicators.mean(axis=0).values
#         # Take the majority label as the group label
#         label = (group_df["type"] == "attack").mode()[0]
#         aggregated_features.append([label] + list(aggregated_row))
#     aggregated_df = pd.DataFrame(
#         aggregated_features, columns=[f"{i}" for i in range(1, 12)]
#     )
#     return aggregated_df


# # Loading the data
# df_train = pd.read_csv("../data/guardian/knn_model/knn_training_7117532915-300-4.csv")
# df_test = pd.read_csv("../data/guardian/knn_model/test_1conv_7117532915-300-4.csv")

# # Aggregating rows for train and test datasets
# df_train_aggregated = aggregate_rows(df_train)
# df_test_aggregated = aggregate_rows(df_test)

# # Encoding labels
# df_train_aggregated["1"] = df_train_aggregated["1"].map({True: 0, False: 1})
# df_test_aggregated["1"] = df_test_aggregated["1"].map({True: 0, False: 1})
# X_train = df_train_aggregated.iloc[:, 1:]
# y_train = df_train_aggregated["1"]
# X_test = df_test_aggregated.iloc[:, 1:]
# y_test = df_test_aggregated["1"]

# # Handling imbalance with SMOTE
# smote = SMOTE(random_state=42)
# X_sm, y_sm = smote.fit_resample(X_train, y_train)

# # Standardizing the features
# scaler = StandardScaler()
# X_sm_scaled = scaler.fit_transform(X_sm)
# X_test_scaled = scaler.transform(X_test)

# # Training the logistic regression model
# model = LogisticRegression(max_iter=1000, random_state=42)
# model.fit(X_sm_scaled, y_sm)

# # Predicting on the test set
# predictions = model.predict(X_test_scaled)

# # Evaluating the model
# print("Accuracy:", accuracy_score(y_test, predictions))
# print(classification_report(y_test, predictions))

# # Note: Replace "path_to_your_train_data.csv" and "path_to_your_test_data.csv" with the actual file paths.
# print("Accuracy:", accuracy_score(y_test, predictions))
# print("Recall:", recall_score(y_test, predictions))
# print("Precision:", precision_score(y_test, predictions))
# print("F1 Score:", f1_score(y_test, predictions))
# print("\nFull Classification Report:\n", classification_report(y_test, predictions))

# # Calculating the confusion matrix
# cm = confusion_matrix(y_test, predictions)

# # Plotting the confusion matrix
# plt.figure(figsize=(8, 6))
# sns.heatmap(
#     cm,
#     annot=True,
#     fmt="d",
#     cmap="Blues",
#     xticklabels=["Predicted attack", "Predicted Normal"],
#     yticklabels=["Actual Attack", "Actual Normal"],
# )
# plt.ylabel("Actual")
# plt.xlabel("Predicted")
# plt.title("Confusion Matrix")
# plt.show()

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import numpy as np

# Load the dataset
df = pd.read_csv("../data/guardian/knn_model/knn_training_7117532915-300-4.csv")

# Feature Engineering: Create binary indicators
for col in df.columns[1:]:  # Skipping the label column
    df[f"{col}_gt_0.6"] = (df[col] > 0.5).astype(int)

# Prepare the data
X = df.drop("type", axis=1)  # Features
y = df["type"].map({"attack": 1, "normal": 0})  # Mapping labels to binary

# Handling Imbalance with SMOTE
smote = SMOTE()

# Logistic Regression Model
logistic_regression = LogisticRegression()

# K-fold Cross-Validation setup
kfold = StratifiedKFold(n_splits=5)

# Pipeline: SMOTE + Logistic Regression
pipeline = Pipeline([("SMOTE", smote), ("Logistic Regression", logistic_regression)])

# Cross-validation and model training
for train_index, test_index in kfold.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    print(classification_report(y_test, predictions))

# Evaluation on new dataset
new_df = pd.read_csv("../data/guardian/knn_model/test_1conv_7117532915-300-4.csv")
# Apply the same feature engineering to the new dataset
for col in new_df.columns[1:]:
    new_df[f"{col}_gt_0.6"] = (new_df[col] > 0.6).astype(int)

new_X = new_df.drop("type", axis=1)
new_y = new_df["type"].map({"attack": 1, "normal": 0})
new_predictions = pipeline.predict(new_X)
print("Final Evaluation on New Dataset")
print("Classification Report:")
print(classification_report(new_y, new_predictions))
print("Confusion Matrix:")
print(confusion_matrix(new_y, new_predictions))
