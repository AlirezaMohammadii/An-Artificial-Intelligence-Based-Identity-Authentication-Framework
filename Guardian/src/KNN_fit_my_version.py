import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

# Load training data
data = pd.read_csv("../data/guardian/knn_model/knn_training_8864816883-300-2.csv")

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

# Load test data
test_data = pd.read_csv("../data/guardian/knn_model/test_1conv_8864816883-300-2.csv")
X_test = test_data.iloc[:, 1:]
y_test = test_data.iloc[:, 0:1].values.ravel()

# Predict on the test set
y_pred = knn.predict(X_test)

# Evaluate the model on the test set
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=1))
