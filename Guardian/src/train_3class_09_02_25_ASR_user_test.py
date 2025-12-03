
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
import json
from collections import Counter

sys.path.append("..")
import guardian.constants as c
from guardian.utils_my_version import loading_embedding


def mixup_data(x, y, alpha=0.2):
    """Perform mixup augmentation on a batch of data."""
    batch_size = x.shape[0]
    lam = np.random.beta(alpha, alpha)
    # Randomly shuffle the batch
    index = np.random.permutation(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y

def balanced_batch_generator_with_mixup(X, y, batch_size, num_classes=3, mixup_prob=0.5):
    """
    Generates balanced batches and applies mixup augmentation selectively
    for the minority classes (class 1 and class 2).

    Args:
        X (np.array): Input data.
        y (np.array): Class indices.
        batch_size (int): Size of the batch.
        num_classes (int): Total number of classes.
        mixup_prob (float): Probability to apply mixup on a given batch (or per sample).
    
    Yields:
        X_batch, y_batch: Augmented and one-hot encoded batch.
    """
    while True:
        # Separate indices by class
        idx_0 = np.where(y == 0)[0]  # normal
        idx_1 = np.where(y == 1)[0]  # attack
        idx_2 = np.where(y == 2)[0]  # triggered

        # Oversample each class to the size of the majority
        max_count = max(len(idx_0), len(idx_1), len(idx_2))
        idx_0_balanced = np.random.choice(idx_0, max_count, replace=True)
        idx_1_balanced = np.random.choice(idx_1, max_count, replace=True)
        idx_2_balanced = np.random.choice(idx_2, max_count, replace=True)

        # Combine indices and shuffle
        balanced_idx = np.hstack([idx_0_balanced, idx_1_balanced, idx_2_balanced])
        np.random.shuffle(balanced_idx)

        # Yield batches
        for i in range(0, len(balanced_idx), batch_size):
            batch_idx = balanced_idx[i:i + batch_size]
            X_batch = X[batch_idx]
            y_batch_indices = y[batch_idx]
            y_batch = to_categorical(y_batch_indices, num_classes=num_classes)

            # Apply mixup only if the batch contains any minority class samples
            if np.any((y_batch_indices == 1) | (y_batch_indices == 2)) and np.random.rand() < mixup_prob:
                X_batch, y_batch = mixup_data(X_batch, y_batch, alpha=0.2)

            yield X_batch, y_batch


def main(model_ID, epochs, batch_size, n_splits, embedding_folder,class_weight,learning_rate):
    """
    (unchanged except for the call to loading_embedding, which now returns file names)
    """
    print("\n=== STARTING TRAINING PROCESS ===")
    start_train_time = tm.time()
    # Note: now loading_embedding returns (x, y, file_list)
    x, y, file_list = loading_embedding(embedding_folder)
    # For training, we simply use the number of files
    num_files = len(file_list)
    y = to_categorical(y, num_classes=3)
    print(f"Loaded {num_files} files. Input shape: {x.shape}, Labels shape: {y.shape}")
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True)
    fold_reports = []
    fold_var = 1
    for train_indices, val_indices in kfold.split(x, np.argmax(y, axis=1)):
        print(f"Training on fold {fold_var}...")
        x_train, y_train = x[train_indices], y[train_indices]
        x_val, y_val = x[val_indices], y[val_indices]
        model = tf.keras.models.load_model(
            c.DISCRIMINATOR_MODEL + str(model_ID) + ".h5"
        )
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy", Precision(name="precision"), Recall(name="recall")],
        )
        early_stopping = EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(
            filepath=f"{c.DISCRIMINATOR_CHECKPOINT_FOLDER}/{model_ID}-fold{fold_var}.h5",
            monitor="val_loss",
            save_best_only=True,
        )
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=10, min_lr=0.00001)
        history = model.fit(
            balanced_batch_generator_with_mixup(x_train, np.argmax(y_train, axis=1), batch_size=batch_size),
            steps_per_epoch=len(x_train) // batch_size,
            validation_data=(x_val, y_val),
            epochs=epochs,
            callbacks=[early_stopping, model_checkpoint, reduce_lr],
            class_weight=class_weight,
            verbose=1,
        )
        y_pred = np.argmax(model.predict(x_val), axis=1)
        y_true = np.argmax(y_val, axis=1)
        report = classification_report(y_true, y_pred, target_names=["normal", "attack", "triggered"])
        print(f"Classification Report for Fold {fold_var}:\n{report}")
        fold_reports.append({"fold": fold_var, "report": report})
        fold_var += 1
    print("\nTraining Summary Report")
    for fold_report in fold_reports:
        print(f"Fold {fold_report['fold']} Report:\n{fold_report['report']}")
    train_execution_time = tm.time() - start_train_time
    print(f"\nTotal Training Time: {train_execution_time:.2f} seconds")
    return [f"{model_ID}-fold{fold_var}" for fold_var in range(1, n_splits + 1)], epochs, file_list


def extract_user_id(filename):
    """
    Extracts the user ID from a filename.
    Assumes the filename follows the format:
         <prefix>-<user_part>-<suffix>.npy,
    where the second part (user_part) is enclosed in either square brackets or parentheses.
    
    Example:
        "10000-[943]-1.npy"  -> returns "943"
        "3256-(2324)-0008.npy"  -> returns "2324"
    
    If the filename does not follow the expected pattern, returns "unknown".
    """
    parts = filename.split('-')
    if len(parts) < 2:
        return "unknown"
    user_part = parts[1]
    user_id = user_part.strip("[]()")
    return user_id

def test_all_fold_models(model_ID, test_data_folder, n_splits, triggered_perc, attacked_perc):
    """
    Loads each fold's saved model checkpoint and evaluates it on a separate test dataset.
    Now includes file-level and user-level Attack Success Rate (ASR) calculations and user-level voting evaluation.
    
    For user-level evaluation:
      - Files from the same user are grouped by parsing the filename.
      - A user is classified as 'triggered' if at least a specified percentage (triggered_perc) of that user's files
        are predicted as triggered; else if at least attacked_perc are predicted as attack then the user is labeled as attack;
        otherwise, the user is labeled as normal.
      - The ground truth user label is determined by a majority vote over that user's file labels.
      - Detailed reports (with confusion matrices, classification reports, and ASR analyses) are printed.
    
    Additionally, the function generates a JSON file recording per-fold file-level and user-level evaluation details.
    """
    print("\n=== STARTING TEST PROCESS ===")
    start_test_time = tm.time()
    print(f"\nLoading separate test dataset from: {test_data_folder}")
    X_test, y_test, test_filenames = loading_embedding(test_data_folder)
    print(f"Loaded {len(test_filenames)} test files. Shape: {X_test.shape}")
    y_test_onehot = to_categorical(y_test, num_classes=3)
    
    # Dictionaries to hold JSON results across all folds
    all_folds_file_results = {}
    all_folds_user_results = {}

    for fold_idx in range(1, n_splits + 1):
        checkpoint_path = f"{c.DISCRIMINATOR_CHECKPOINT_FOLDER}/{model_ID}-fold{fold_idx}.h5"
        if not os.path.exists(checkpoint_path):
            print(f"[WARNING] Checkpoint file not found for fold {fold_idx}: {checkpoint_path}")
            continue

        print(f"\n--- Testing Fold {fold_idx} ---")
        model = tf.keras.models.load_model(
            checkpoint_path,
            custom_objects={'focal_loss_fn': lambda y_true, y_pred: 0.0}
        )
        predictions = model.predict(X_test)
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(y_test_onehot, axis=1)
        
        # Calculate file-level ASR for triggered samples
        triggered_mask = (y_true == 2)
        triggered_total = np.sum(triggered_mask)
        if triggered_total > 0:
            triggered_misclassified = np.sum(y_pred[triggered_mask] == 0)
            asr = (triggered_misclassified / triggered_total) * 100
        else:
            asr = 0.0
            triggered_misclassified = 0
        
        print(classification_report(
            y_true, y_pred,
            target_names=["normal", "attack", "triggered"]
        ))
        print(f"Attack Success Rate Analysis [Triggered -> Normal] (File-level):")
        print(f"| Total Triggered Samples: {triggered_total}")
        print(f"| Misclassified as Normal: {triggered_misclassified}")
        print(f"| ASR: {asr:.2f}%\n")
        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
        
        # Build file-level results (TP, FP, FN, TN) for each class
        class_names = {0: "normal", 1: "attack", 2: "triggered"}
        file_results = {
            "normal": {"TP": [], "FP": [], "FN": [], "TN": []},
            "attack": {"TP": [], "FP": [], "FN": [], "TN": []},
            "triggered": {"TP": [], "FP": [], "FN": [], "TN": []}
        }
        for i, fname in enumerate(test_filenames):
            for cls in [0, 1, 2]:
                cls_name = class_names[cls]
                if y_true[i] == cls and y_pred[i] == cls:
                    file_results[cls_name]["TP"].append(fname)
                elif y_true[i] != cls and y_pred[i] == cls:
                    file_results[cls_name]["FP"].append(fname)
                elif y_true[i] == cls and y_pred[i] != cls:
                    file_results[cls_name]["FN"].append(fname)
                else:
                    file_results[cls_name]["TN"].append(fname)
        
        # User-level aggregation: group file indices by user ID (parsed from filename)
        user_groups = {}
        for i, fname in enumerate(test_filenames):
            user_id = extract_user_id(fname)
            if user_id not in user_groups:
                user_groups[user_id] = []
            user_groups[user_id].append(i)
        
        user_true_list = []
        user_pred_list = []
        user_details = {}
        for user_id, indices in user_groups.items():
            total_files = len(indices)
            triggered_count = sum(1 for i in indices if y_pred[i] == 2)
            attack_count = sum(1 for i in indices if y_pred[i] == 1)
            fraction_triggered = triggered_count / total_files
            fraction_attack = attack_count / total_files
            # Voting logic based on specified thresholds:
            if fraction_triggered >= triggered_perc:
                final_pred = 2
            elif fraction_attack >= attacked_perc:
                final_pred = 1
            else:
                final_pred = 0
            # Determine the ground truth by majority vote
            user_true_votes = [y_true[i] for i in indices]
            vote_count = Counter(user_true_votes)
            final_true = vote_count.most_common(1)[0][0]
            user_true_list.append(final_true)
            user_pred_list.append(final_pred)
            user_details[user_id] = {
                "total_files": total_files,
                "fraction_triggered": fraction_triggered,
                "fraction_attack": fraction_attack,
                "final_pred": final_pred,
                "final_true": final_true,
                "file_indices": indices
            }
        
        # --- NEW: Calculate user-level ASR for triggered users ---
        triggered_users_total = sum(1 for label in user_true_list if label == 2)
        triggered_users_misclassified = sum(1 for true, pred in zip(user_true_list, user_pred_list)
                                            if true == 2 and pred == 0)
        if triggered_users_total > 0:
            user_asr = (triggered_users_misclassified / triggered_users_total) * 100
        else:
            user_asr = 0.0
        
        print("User-Level Attack Success Rate Analysis [Triggered -> Normal]:")
        print(f"| Total Triggered Users: {triggered_users_total}")
        print(f"| Misclassified as Normal: {triggered_users_misclassified}")
        print(f"| User-Level ASR: {user_asr:.2f}%\n")
        # ---------------------------------------------------------

        user_report = classification_report(user_true_list, user_pred_list, target_names=["normal", "attack", "triggered"])
        user_conf_matrix = confusion_matrix(user_true_list, user_pred_list)
        
        professional_report = f"""
User-Level Evaluation Report for Fold {fold_idx}:
------------------------------------------------------------
Total number of users evaluated: {len(user_groups)}
User-level confusion matrix:
{user_conf_matrix}

Classification Report (User Level):
{user_report}

Evaluation Logic:
- A user is classified as 'triggered' if ≥{triggered_perc*100:.0f}% of their files are predicted as 'triggered'.
- Else, a user is classified as 'attack' if ≥{attacked_perc*100:.0f}% of their files are predicted as 'attack'.
- Otherwise, the user is classified as 'normal'.
Ground truth user label is determined by a majority vote of individual file labels.
------------------------------------------------------------
"""
        print(professional_report)
        
        all_folds_file_results[f"fold_{fold_idx}"] = file_results
        all_folds_user_results[f"fold_{fold_idx}"] = user_details
    
    test_execution_time = tm.time() - start_test_time
    print(f"\nTotal Test Time: {test_execution_time:.2f} seconds")
    
    final_json = {
        "file_results": all_folds_file_results,
        "user_results": all_folds_user_results
    }
    json_filename = f"test_results_{model_ID}.json"
    with open(json_filename, "w") as f:
        json.dump(final_json, f, indent=4, default=lambda x: int(x) if isinstance(x, np.integer) else x)
    print(f"Detailed file-level and user-level evaluation results saved in {json_filename}")


if __name__ == "__main__":
    print("Select Operation Mode:")
    print("1) Train & cross-validate a model (from scratch) and then test on a new dataset.")
    print("2) Only test an existing set of fold checkpoints on a new dataset.")
    choice = input("Enter 1 or 2: ").strip()
    model_ID = input("Enter the model ID: ")
    # train and test folders
    embedding_folder = "../data/sample_dataset/libri_vox_mix/embedding/"
    # embedding_folder = "../data/sample_dataset/libri_data/embedding/"
    test_data_folder = "../data/sample_dataset/libri_vox_mix/test/embedding/"
    # test_data_folder = "../data/sample_dataset/libri_data/test/embedding/"

    # Thresholds for user-level voting:
    triggered_perc = 0.1
    attacked_perc = 0.01

    epochs = 120
    batch_size = 64
    n_splits = 5
    class_weight = {0:1.0, 1:3, 2:2.5}
    learning_rate = 0.001
    if choice == '1':
        start_time = tm.time()
        name_training, train_epochs, file_list = main(model_ID, epochs, batch_size, n_splits, embedding_folder,class_weight,learning_rate)
        elapsed = tm.time() - start_time
        print(f"\nTraining completed in {elapsed:.2f} seconds.")
        print(f"Saved model checkpoints: {name_training}")
        test_all_fold_models(model_ID, test_data_folder, n_splits, triggered_perc, attacked_perc)
        print("\nAll cross-validation reports have been shown above, and final test set performance is shown per fold checkpoint. Task completed.\n")
    elif choice == '2':
        n_splits = 5
        test_all_fold_models(model_ID, test_data_folder, n_splits, triggered_perc, attacked_perc)
        print("\nFinal test set performance has been reported for all saved fold models. Task completed.\n")
    else:
        print("Invalid input. Please run the script again and select either 1 or 2.")
