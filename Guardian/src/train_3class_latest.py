"""
Key Updates:
Balanced Batch Generator:

Resamples all three classes (normal, attack, triggered) for balanced batches.
Multi-class Metrics:

accuracy, precision, and recall handle multi-class problems.
Adds a detailed classification report for each fold.
One-hot Encoding:

Labels (y) are converted to one-hot encoding before training.
Cross-Validation:

Implements K-Fold Cross-Validation to assess performance across different data splits.
Callbacks:

Uses EarlyStopping, ModelCheckpoint, and ReduceLROnPlateau to improve training efficiency and save the best model.


"""
import numpy as np
import os
import sys
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from datetime import datetime
import matplotlib.pyplot as plt
import time as tm

sys.path.append("..")
import guardian.constants as c
from guardian.utils_my_version import loading_embedding


def balanced_batch_generator(X, y, batch_size=128, num_classes=3):
    """
    Generates balanced batches for three classes: normal (0), attack (1), and triggered (2).
    Ensures equal representation of all classes in each batch by resampling.

    Args:
        X (np.array): Input embeddings.
        y (np.array): Labels for the embeddings (class indices).
        batch_size (int): Batch size for training.
        num_classes (int): Number of classes (default is 3 for this problem).

    Yields:
        Balanced batches of data (X_batch, y_batch).
    """
    while True:
        # Separate indices by class
        idx_0 = np.where(y == 0)[0]  # normal
        idx_1 = np.where(y == 1)[0]  # attack
        idx_2 = np.where(y == 2)[0]  # triggered

        # Resample indices to balance the classes
        max_count = max(len(idx_0), len(idx_1), len(idx_2))
        idx_0_balanced = np.random.choice(idx_0, max_count, replace=True)
        idx_1_balanced = np.random.choice(idx_1, max_count, replace=True)
        idx_2_balanced = np.random.choice(idx_2, max_count, replace=True)

        # Combine and shuffle indices
        balanced_idx = np.hstack([idx_0_balanced, idx_1_balanced, idx_2_balanced])
        np.random.shuffle(balanced_idx)

        # Yield batches
        for i in range(0, len(balanced_idx), batch_size):
            batch_idx = balanced_idx[i:i + batch_size]
            X_batch = X[batch_idx]
            y_batch = to_categorical(y[batch_idx], num_classes=num_classes)  # Convert to one-hot
            yield X_batch, y_batch

def main(model_ID, epochs, batch_size, n_splits, embedding_folder):
    """
    Trains the model using K-Fold Cross-Validation, balanced batches, and custom callbacks.

    Args:
        model_ID (str): ID of the model to load for training.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        n_splits (int): Number of splits for K-Fold Cross-Validation.
        embedding_folder (str): Path to the folder containing embeddings.

    Returns:
        name_training (list): List of saved model names for each fold.
        grad_steps (int): Total number of gradient steps.
        num_files (int): Number of files used in training.
    """
    # Load data
    x, y, num_files = loading_embedding(embedding_folder)

    y = to_categorical(y, num_classes=3)  # One-hot encode labels

    print(f"Loaded {num_files} files. Input shape: {x.shape}, Labels shape: {y.shape}")

    # K-Fold Cross-Validation
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True)
    fold_reports = []
    fold_var = 1

    for train_indices, val_indices in kfold.split(x, np.argmax(y, axis=1)):
        print(f"Training on fold {fold_var}...")

        # Split data into training and validation sets
        x_train, y_train = x[train_indices], y[train_indices]
        x_val, y_val = x[val_indices], y[val_indices]

        # Load and compile the model
        model = tf.keras.models.load_model(
            c.DISCRIMINATOR_MODEL + str(model_ID) + ".h5"
        )
        model.compile(
            optimizer=Adam(learning_rate=0.00001),
            loss="categorical_crossentropy",
            metrics=["accuracy", Precision(name="precision"), Recall(name="recall")],
        )

        # Callbacks
        early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(
            filepath=f"{c.DISCRIMINATOR_CHECKPOINT_FOLDER}/{model_ID}-fold{fold_var}.h5",
            monitor="val_loss",
            save_best_only=True,
        )
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5, min_lr=0.00001)

        # Train the model
        history = model.fit(
            balanced_batch_generator(x_train, np.argmax(y_train, axis=1), batch_size=batch_size),
            steps_per_epoch=len(x_train) // batch_size,
            validation_data=(x_val, y_val),
            epochs=epochs,
            callbacks=[early_stopping, model_checkpoint, reduce_lr],
            verbose=1,
        )

        # Evaluate model
        y_pred = np.argmax(model.predict(x_val), axis=1)
        y_true = np.argmax(y_val, axis=1)
        report = classification_report(y_true, y_pred, target_names=["normal", "attack", "triggered"])
        print(f"Classification Report for Fold {fold_var}:\n{report}")

        fold_reports.append({"fold": fold_var, "report": report})
        fold_var += 1

    # Print Summary Report
    print("\nTraining Summary Report")
    for fold_report in fold_reports:
        print(f"Fold {fold_report['fold']} Report:\n{fold_report['report']}")

    return [f"{model_ID}-fold{fold_var}" for fold_var in range(1, n_splits + 1)], epochs, num_files


def test_all_fold_models(model_ID, test_data_folder, n_splits=5):
    """
    Loads each fold's saved model checkpoint and evaluates it on a separate test dataset.

    Args:
        model_ID (str): The ID of the model (used to find saved checkpoints).
        test_data_folder (str): The path to the new test dataset folder.
        n_splits (int): Number of folds (default=5).
    """
    # 1) Load test dataset
    print(f"\nLoading separate test dataset from: {test_data_folder}")
    X_test, y_test, num_test_files = loading_embedding(test_data_folder)
    print(f"Loaded {num_test_files} test files. Shape: {X_test.shape}")

    # Ensure y_test is one-hot
    y_test_onehot = to_categorical(y_test, num_classes=3)

    # 2) Evaluate each fold checkpoint
    print("\nEvaluating each fold model on the final test set:\n")
    for fold_idx in range(1, n_splits + 1):
        checkpoint_path = f"{c.DISCRIMINATOR_CHECKPOINT_FOLDER}/{model_ID}-fold{fold_idx}.h5"
        if not os.path.exists(checkpoint_path):
            print(f"[WARNING] Checkpoint file not found for fold {fold_idx}: {checkpoint_path}")
            continue

        print(f"\n--- Testing Fold {fold_idx} ---")
        # If your model used a custom loss like focal_loss_fn, pass it in custom_objects:
        # model = tf.keras.models.load_model(
        #     checkpoint_path,
        #     custom_objects={'focal_loss_fn': focal_loss_fn}
        # )

        def dummy_focal_loss(y_true, y_pred):
            return tf.constant(0.0)
        # def focal_loss_fn(y_true, y_pred, gamma=2.0, alpha=0.25):
        #     """
        #     Example focal loss function. Adjust for your actual implementation.
        #     """
        #     y_true = tf.cast(y_true, tf.float32)
        #     cross_entropy = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        #     probs = tf.keras.backend.clip(y_pred, 1e-7, 1.0 - 1e-7)
        #     focal = alpha * tf.pow((1 - probs), gamma) * cross_entropy
        #     return tf.reduce_mean(focal)

        model = tf.keras.models.load_model(
            checkpoint_path,
            custom_objects={'focal_loss_fn': dummy_focal_loss}
        )
        predictions = model.predict(X_test)
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(y_test_onehot, axis=1)

        # Generate classification report
        report = classification_report(
            y_true,
            y_pred,
            target_names=["normal", "attack", "triggered"]
        )
        print(report)


if __name__ == "__main__":
    # Ask user for operation mode
    print("Select Operation Mode:")
    print("1) Train & cross-validate a model (from scratch) and then test on a new dataset.")
    print("2) Only test an existing set of fold checkpoints on a new dataset.")
    choice = input("Enter 1 or 2: ").strip()

    # Common input
    model_ID = input("Enter the model ID: ")

    if choice == '1':
        # ------------------
        # (1) TRAIN & TEST
        # ------------------
        epochs = 120
        batch_size = 64
        n_splits = 5
        embedding_folder = "../data/sample_dataset/libri_data/embedding/train/"

        # Train/Cross-Validate
        start_time = tm.time()
        name_training, train_epochs, num_files = main(
            model_ID, epochs, batch_size, n_splits, embedding_folder
        )
        elapsed = tm.time() - start_time
        print(f"\nTraining completed in {elapsed:.2f} seconds.")
        print(f"Saved model checkpoints: {name_training}")

        # Prompt user for final test dataset path
        # test_data_folder = input("\nEnter the path to the NEW test dataset folder: ").strip()
        test_data_folder = "../data/sample_dataset/libri_data/embedding/test/"

        # Test on new data using all fold checkpoints
        test_all_fold_models(model_ID, test_data_folder, n_splits)

        print("\nAll cross-validation reports have been shown above, and final test set "
              "performance is shown per fold checkpoint. Task completed.\n")

    elif choice == '2':
        # ------------------
        # (2) ONLY TEST
        # ------------------
        n_splits = 5
        # Possibly ask user for the path, or use a default:
        # test_data_folder = input("\nEnter the path to the NEW test dataset folder: ").strip()
        test_data_folder = "../data/sample_dataset/libri_data/embedding/test/"

        test_all_fold_models(model_ID, test_data_folder, n_splits)

        print("\nFinal test set performance has been reported for all saved fold models. Task completed.\n")

    else:
        # Invalid choice
        print("Invalid input. Please run the script again and select either 1 or 2.")

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------

# Working version without testing incorporated

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------



# """
# Key Updates:
# Balanced Batch Generator:

# Resamples all three classes (normal, attack, triggered) for balanced batches.
# Multi-class Metrics:

# accuracy, precision, and recall handle multi-class problems.
# Adds a detailed classification report for each fold.
# One-hot Encoding:

# Labels (y) are converted to one-hot encoding before training.
# Cross-Validation:

# Implements K-Fold Cross-Validation to assess performance across different data splits.
# Callbacks:

# Uses EarlyStopping, ModelCheckpoint, and ReduceLROnPlateau to improve training efficiency and save the best model.


# """
# import numpy as np
# import os
# import sys
# import tensorflow as tf
# from sklearn.metrics import classification_report, confusion_matrix, f1_score
# from sklearn.model_selection import StratifiedKFold
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# from tensorflow.keras.metrics import Precision, Recall
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.utils import to_categorical
# from datetime import datetime
# import matplotlib.pyplot as plt
# import time as tm

# sys.path.append("..")
# import guardian.constants as c
# from guardian.utils_my_version import loading_embedding


# def balanced_batch_generator(X, y, batch_size=128, num_classes=3):
#     """
#     Generates balanced batches for three classes: normal (0), attack (1), and triggered (2).
#     Ensures equal representation of all classes in each batch by resampling.

#     Args:
#         X (np.array): Input embeddings.
#         y (np.array): Labels for the embeddings (class indices).
#         batch_size (int): Batch size for training.
#         num_classes (int): Number of classes (default is 3 for this problem).

#     Yields:
#         Balanced batches of data (X_batch, y_batch).
#     """
#     while True:
#         # Separate indices by class
#         idx_0 = np.where(y == 0)[0]  # normal
#         idx_1 = np.where(y == 1)[0]  # attack
#         idx_2 = np.where(y == 2)[0]  # triggered

#         # Resample indices to balance the classes
#         max_count = max(len(idx_0), len(idx_1), len(idx_2))
#         idx_0_balanced = np.random.choice(idx_0, max_count, replace=True)
#         idx_1_balanced = np.random.choice(idx_1, max_count, replace=True)
#         idx_2_balanced = np.random.choice(idx_2, max_count, replace=True)

#         # Combine and shuffle indices
#         balanced_idx = np.hstack([idx_0_balanced, idx_1_balanced, idx_2_balanced])
#         np.random.shuffle(balanced_idx)

#         # Yield batches
#         for i in range(0, len(balanced_idx), batch_size):
#             batch_idx = balanced_idx[i:i + batch_size]
#             X_batch = X[batch_idx]
#             y_batch = to_categorical(y[batch_idx], num_classes=num_classes)  # Convert to one-hot
#             yield X_batch, y_batch


def main(model_ID, epochs, batch_size, n_splits, embedding_folder):
    """
    Trains the model using K-Fold Cross-Validation, balanced batches, and custom callbacks.

    Args:
        model_ID (str): ID of the model to load for training.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        n_splits (int): Number of splits for K-Fold Cross-Validation.
        embedding_folder (str): Path to the folder containing embeddings.

    Returns:
        name_training (list): List of saved model names for each fold.
        grad_steps (int): Total number of gradient steps.
        num_files (int): Number of files used in training.
    """
    # Load data
    x, y, num_files = loading_embedding(embedding_folder)

    y = to_categorical(y, num_classes=3)  # One-hot encode labels

    print(f"Loaded {num_files} files. Input shape: {x.shape}, Labels shape: {y.shape}")

    # K-Fold Cross-Validation
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True)
    fold_reports = []
    fold_var = 1

    for train_indices, val_indices in kfold.split(x, np.argmax(y, axis=1)):
        print(f"Training on fold {fold_var}...")

        # Split data into training and validation sets
        x_train, y_train = x[train_indices], y[train_indices]
        x_val, y_val = x[val_indices], y[val_indices]

        # Load and compile the model
        model = tf.keras.models.load_model(
            c.DISCRIMINATOR_MODEL + str(model_ID) + ".h5"
        )
        model.compile(
            optimizer=Adam(learning_rate=0.00001),
            loss="categorical_crossentropy",
            metrics=["accuracy", Precision(name="precision"), Recall(name="recall")],
        )

        # Callbacks
        early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(
            filepath=f"{c.DISCRIMINATOR_CHECKPOINT_FOLDER}/{model_ID}-fold{fold_var}.h5",
            monitor="val_loss",
            save_best_only=True,
        )
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5, min_lr=0.00001)

        # Train the model
        history = model.fit(
            balanced_batch_generator(x_train, np.argmax(y_train, axis=1), batch_size=batch_size),
            steps_per_epoch=len(x_train) // batch_size,
            validation_data=(x_val, y_val),
            epochs=epochs,
            callbacks=[early_stopping, model_checkpoint, reduce_lr],
            verbose=1,
        )

        # Evaluate model
        y_pred = np.argmax(model.predict(x_val), axis=1)
        y_true = np.argmax(y_val, axis=1)
        report = classification_report(y_true, y_pred, target_names=["normal", "attack", "triggered"])
        print(f"Classification Report for Fold {fold_var}:\n{report}")

        fold_reports.append({"fold": fold_var, "report": report})
        fold_var += 1

    # Print Summary Report
    print("\nTraining Summary Report")
    for fold_report in fold_reports:
        print(f"Fold {fold_report['fold']} Report:\n{fold_report['report']}")

    return [f"{model_ID}-fold{fold_var}" for fold_var in range(1, n_splits + 1)], epochs, num_files




# if __name__ == "__main__":
#     # User inputs
#     model_ID = input("Enter the model ID: ")
#     epochs = 120
#     batch_size = 64
#     n_splits = 5
#     embedding_folder = "../data/sample_dataset/libri_data/embedding/"
#     # Train the model
#     start_time = tm.time()
#     name_training, grad_steps, num_files = main(model_ID, epochs, batch_size, n_splits, embedding_folder)
#     print(f"Training completed in {tm.time() - start_time:.2f} seconds.")
#     print(f"Saved models: {name_training}")

