# git push -f origin main
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Loading Data
all_matrices = np.load(
    "F:/1.Deakin university/Python/13_10_2023_My_Project_1/Deep-VR/deep-speaker/samples/LibriSpeech/dev-clean/all_matrices.npy"
)

# Step 2: Data Preprocessing
X = all_matrices.reshape(all_matrices.shape[0], -1)  # Flatten to 2D
X = StandardScaler().fit_transform(X)
X = X.reshape(all_matrices.shape[0], 32, 32, 1)  # Reshape back to 4D
y = np.random.choice([0, 1], size=(X.shape[0],))
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 3: Model Architecture
model = models.Sequential()
model.add(
    layers.Conv2D(
        64, (4, 4), strides=(1, 1), activation="relu", input_shape=(32, 32, 1)
    )
)
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), strides=(1, 1), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(2, activation="softmax"))

# Step 4: Model Compilation
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Step 5: Model Training
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Step 6: Model Evaluation
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc}")

# Step 7: KNN Classifier
training_probabilities = model.predict(X_train)
new_data_probabilities = model.predict(X_test)

knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(training_probabilities, y_train)
predicted_labels = knn.predict(new_data_probabilities)

# Step 8: Display Classification Report and Confusion Matrix
print("Classification Report:")
print(classification_report(y_test, predicted_labels))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predicted_labels))

# Step 9: Display Training Accuracy
# print("Training Accuracy:")
# print(history.history["accuracy"])
