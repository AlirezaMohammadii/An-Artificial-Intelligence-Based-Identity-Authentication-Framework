import logging
import numpy as np
import os
import sys
import tensorflow as tf
from sklearn.metrics import f1_score, precision_recall_curve, auc
from sklearn.model_selection import GroupKFold
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from datetime import datetime
import matplotlib
import time as tm
import configparser

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append("..")
import guardian.constants as c
from guardian.utils_my_version import loading_embedding

# Balanced Batch Generator
def balanced_batch_generator(X, y, batch_size):
    while True:
        idx = np.arange(len(y))
        np.random.shuffle(idx)
        for i in range(0, len(idx), batch_size):
            batch_idx = idx[i : i + batch_size]
            yield X[batch_idx], y[batch_idx]


# Custom Callback for Metrics and Learning Rate Reporting
class MetricsAndLRCallback(Callback):
    def __init__(self, validation_data, fold, model_number):
        super().__init__()
        self.validation_data = validation_data
        self.fold = fold
        self.model_number = model_number
        self.last_epoch_metrics = {}

    def on_epoch_end(self, epoch, logs=None):
        X_val, y_val = self.validation_data
        y_pred = (self.model.predict(X_val) > 0.5).astype("int32")

        # F1 Score
        f1 = f1_score(y_val, y_pred)

        # AUPRC
        precision, recall, _ = precision_recall_curve(y_val, self.model.predict(X_val))
        auprc = auc(recall, precision)

        # Learning Rate
        current_lr = self.model.optimizer.lr.numpy()

        # Store metrics
        self.last_epoch_metrics = {
            "f1_score": f1,
            "auprc": auprc,
            "loss": logs.get("loss"),
            "accuracy": logs.get("accuracy"),
            "precision": logs.get("precision"),
            "recall": logs.get("recall"),
            "val_loss": logs.get("val_loss"),
            "val_accuracy": logs.get("val_accuracy"),
            "val_precision": logs.get("val_precision"),
            "val_recall": logs.get("val_recall"),
            "learning_rate": current_lr,
        }

        # Print metrics for the current epoch
        print(f"Fold {self.fold}, Model {self.model_number}, Epoch {epoch + 1}:")
        for key, value in self.last_epoch_metrics.items():
            print(f"  {key}: {value}")


# Custom ReduceLROnPlateau with Reporting
class ReportingReduceLROnPlateau(ReduceLROnPlateau):
    def on_epoch_end(self, epoch, logs=None):
        prev_lr = self.model.optimizer.lr.numpy()
        super().on_epoch_end(epoch, logs)
        current_lr = self.model.optimizer.lr.numpy()
        if current_lr < prev_lr:
            print(f"Learning rate reduced to {current_lr} at epoch {epoch + 1}.")


def normalize_embeddings(x):
    """Normalize embeddings to ensure consistency."""
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    x_normalized = (x - x_mean) / (x_std + 1e-10)  # Avoid division by zero
    return x_normalized


def main(config, model_number):
    # Read configuration parameters
    embedding_folder = config['main']['embedding_folder']
    n_splits = config.getint('main', 'n_splits')
    epochs = config.getint('main', 'epochs')
    batch_size = config.getint('main', 'batch_size')
    class_weight_dict = eval(config['main']['class_weight_dict'])  # Convert from string to dictionary
    learning_rate = config.getfloat('main', 'learning_rate')
    early_stopping_patience = config.getint('main', 'early_stopping_patience')
    monitor_checkpoint = config['main']['monitor_checkpoint']
    monitor_reduce_lr = config['main']['monitor_reduce_lr']

    # Load data using guardian.utils
    x, y, user_ids = loading_embedding(embedding_folder)

    # Normalize embeddings
    x = normalize_embeddings(x)

    # Reshape embeddings
    x = x.reshape(-1, 32, 32, 1)

    # Check for NaN or Inf values in your dataset
    assert not np.isnan(x).any(), "X contains NaN values"
    assert not np.isinf(x).any(), "X contains Inf values"
    assert not np.isnan(y).any(), "Y contains NaN values"
    assert not np.isinf(y).any(), "Y contains Inf values"

    # Define GroupKFold cross-validator
    group_kfold = GroupKFold(n_splits=n_splits)

    fold_reports = []
    fold_var = 1
    for train_indices, val_indices in group_kfold.split(x, y, groups=user_ids):
        print(f"Training on fold {fold_var}...")

        # Generate batches from indices
        x_train, y_train = x[train_indices], y[train_indices]
        x_val, y_val = x[val_indices], y[val_indices]

        # Build a new model for each fold
        from save_model_18_11_24 import discriminator_model  # Import model builder
        model, _ = discriminator_model(config)

        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy", Precision(name="precision"), Recall(name="recall")],
        )

        # Callbacks
        metrics_callback = MetricsAndLRCallback(
            validation_data=(x_val, y_val), fold=fold_var, model_number=model_number
        )

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True
        )

        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=f"{c.DISCRIMINATOR_CHECKPOINT_FOLDER}/{model_number}_fold_{fold_var}.h5",
            monitor=monitor_checkpoint,
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        )

        reduce_lr = ReportingReduceLROnPlateau(
            monitor=monitor_reduce_lr,
            factor=0.1,
            patience=5,
            min_lr=0.00001,
            verbose=1,
        )

        history = model.fit(
            balanced_batch_generator(x_train, y_train, batch_size=batch_size),
            steps_per_epoch=len(x_train) // batch_size,
            validation_data=(x_val, y_val),
            epochs=epochs,
            callbacks=[metrics_callback, early_stopping, model_checkpoint, reduce_lr],
            class_weight=class_weight_dict,
            verbose=1,
        )

        # Collect fold-specific information
        fold_reports.append(
            {
                "fold": fold_var,
                "epochs_run": len(history.history["loss"]),
                "last_epoch_metrics": metrics_callback.last_epoch_metrics,
            }
        )

        print(f"Fold {fold_var} completed for Model {model_number}.")
        fold_var += 1

    # After all folds are completed, print the consolidated report
    print("\nTraining Summary Report")
    for report in fold_reports:
        print(f"Fold {report['fold']}:")
        print(f"  Epochs Run: {report['epochs_run']}")
        print(f"  Last Epoch Metrics:")
        for key, value in report['last_epoch_metrics'].items():
            print(f"    {key}: {value}")


def print_config_parameters(config):
    print("\n--- Training Configuration Parameters ---")
    for section in config.sections():
        print(f"\n[{section}]")
        for key in config[section]:
            print(f"{key}: {config[section][key]}")
    print("\n----------------------------------------")


if __name__ == "__main__":
    # Load config file
    model_number = input("Please enter the model number: ")
    config_file_path = 'config_18_11_24_v2.ini'
    config = configparser.ConfigParser()
    config.read(config_file_path)

    # Print the configuration parameters
    print_config_parameters(config)

    start_time_main = tm.time()

    # Run the training with config parameters
    main(config, model_number)

    print("Total computation time: {:.2f} seconds".format(tm.time() - start_time_main))
