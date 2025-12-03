import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

# Load training data
data = pd.read_csv("../data/guardian/knn_model/knn_training_vox_6417810112-120-5.csv")

# Split features and target
X = data.iloc[:, 1:]
y = data.iloc[:, 0].values.ravel()

# Define StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Choose a distance metric from ['euclidean', 'manhattan', 'minkowski', 'chebyshev', 'hamming', 'cosine']
# Note: For Minkowski, you can also specify the `p` parameter, e.g., p=1 for Manhattan distance
n_neighbors = 11
distance_metric = "manhattan"

knn = KNeighborsClassifier(
    n_neighbors=n_neighbors, weights="distance", metric=distance_metric
)

# Apply SMOTE outside of the cross-validation loop to avoid data leakage
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Final model training on the resampled data
knn.fit(X_resampled, y_resampled)

y_pred_train = knn.predict(X_resampled)

print(f'Confusion matrix for training is:\n {confusion_matrix(y_resampled, y_pred_train)}')
print(classification_report(y_resampled, y_pred_train, zero_division=1))


###########################################################################################
###########################################################################################

# x = data.iloc[:,1:]
# y = data.iloc[:,0:1].values.ravel()

# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)
# n_neighbors = 11
# neigh = KNeighborsClassifier(n_neighbors, weights='distance')
# neigh.fit(X_train, y_train)
# y_pred = neigh.predict(x)
# print(confusion_matrix(y,y_pred))
# print(classification_report(y,y_pred))


###########################################################################################
###########################################################################################
# Load test data
test_data = pd.read_csv("../data/guardian/knn_model/test_1conv_6417810112-120-5.csv")
X_test = test_data.iloc[:, 1:]
y_test = test_data.iloc[:, 0:1].values.ravel()

# Predict on the test set
y_pred = knn.predict(X_test)

# Evaluate the model on the test set
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=1))
