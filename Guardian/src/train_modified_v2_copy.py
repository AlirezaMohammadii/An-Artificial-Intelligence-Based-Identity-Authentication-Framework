import logging
import numpy as np
import os
import sys
import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.utils import resample
from datetime import datetime
import matplotlib
import time as tm

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import itertools

sys.path.append("..")
import guardian.constants as c
from guardian.utils import get_last_checkpoint_model_id, loading_embedding


# Custom Focal Loss
def focal_loss(gamma=5, alpha=1, epsilon=1e-7):
    def focal_loss_fixed(y_true, y_pred):
        y_pred = tf.clip_by_value(
            y_pred, epsilon, 1.0 - epsilon
        )  # Prevent 0 or 1 predictions
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        loss = -alpha * (1 - pt) ** gamma * tf.math.log(pt)
        return tf.reduce_sum(loss)

    return focal_loss_fixed


# Balanced Batch Generator
def balanced_batch_generator(X, y, batch_size=128):
    while True:
        idx_0 = np.where(y == 0)[0]
        idx_1 = np.where(y == 1)[0]
        idx_1_upsampled = resample(
            idx_1, replace=True, n_samples=len(idx_0), random_state=123
        )
        balanced_idx = np.hstack([idx_0, idx_1_upsampled])
        np.random.shuffle(balanced_idx)
        for i in range(0, len(balanced_idx), batch_size):
            batch_idx = balanced_idx[i : i + batch_size]
            yield X[batch_idx], y[batch_idx]


# Custom Metrics Callback
class MetricsCallback(Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data
        self.last_epoch_metrics = {}

    def on_epoch_end(self, epoch, logs=None):
        X_val, y_val = self.validation_data
        y_pred = (self.model.predict(X_val) > 0.5).astype("int32")

        # F1 Score
        f1 = f1_score(y_val, y_pred)
        print(f"F1 Score: {f1}")

        # AUPRC
        precision, recall, _ = precision_recall_curve(y_val, self.model.predict(X_val))
        auprc = auc(recall, precision)
        print(f"Area Under the Precision-Recall Curve (AUPRC): {auprc}")

        # Store last epoch's metrics
        self.last_epoch_metrics = {"f1": f1, "auprc": auprc, **logs}


def main(model_ID, epochs, batch_size, n_splits, embedding_folder):
    x, y, num_files = loading_embedding(embedding_folder)
    print("CNN model")
    # Check for NaN or Inf values in your dataset
    assert not np.isnan(x).any(), "X contains NaN values"
    assert not np.isinf(x).any(), "X contains Inf values"
    assert not np.isnan(y).any(), "X contains NaN values"
    assert not np.isinf(y).any(), "X contains Inf values"

    # Calculate class weights for imbalanced datasets
    class_weights = compute_class_weight("balanced", classes=np.unique(y), y=y)
    # class_weight_dict = dict(
    # enumerate(class_weights)
    # )  # "Auto [ 0.52609727 10.07954545]"
    # print("Automatically computed class weights:", class_weights)
    class_weight_dict = {0: 0.1, 1: 15}

    # Define K-fold cross validator
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True)

    fold_reports = []
    fold_var = 1
    for train_indices, val_indices in kfold.split(x, y):
        print(f"Training on fold {fold_var}...")

        # Generate batches from indices
        x_train, y_train = x[train_indices], y[train_indices]
        x_val, y_val = x[val_indices], y[val_indices]

        # create model
        model = tf.keras.models.load_model(
            c.DISCRIMINATOR_MODEL + str(model_ID) + ".h5"
        )
        grad_steps = 0
        last_checkpoint = get_last_checkpoint_model_id(
            c.DISCRIMINATOR_CHECKPOINT_FOLDER, model_ID
        )
        print(last_checkpoint)

        if last_checkpoint is not None:
            logging.info(
                "Found checkpoint [{}]. Resume from here...".format(last_checkpoint)
            )
            model.load_weights(last_checkpoint)
            grad_steps = int(last_checkpoint.split("-")[-1].split(".")[0])
            logging.info("[DONE]")

        # Compile model with custom focal loss and additional metrics
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss=focal_loss(),
            metrics=["accuracy", Precision(name="precision"), Recall(name="recall")],
        )
        # Callbacks
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )
        # Ensure this is set before using it in the filepath
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=f"{c.DISCRIMINATOR_CHECKPOINT_FOLDER}/{model_ID}-{epochs}.h5",
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,  # Set to True to save only the weights
            verbose=1,
        )
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.1,  # Factor by which the learning rate will be reduced. new_lr = lr * factor
            patience=5,  # Number of epochs with no improvement after which learning rate will be reduced.
            min_lr=0.00001,  # Lower bound on the learning rate.
            verbose=1,
        )

        metrics_callback = MetricsCallback(validation_data=(x_val, y_val))
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
        epochs_run = len(history.history["loss"])
        last_epoch_metrics = metrics_callback.last_epoch_metrics
        fold_reports.append(
            {
                "fold": fold_var,
                "epochs_run": epochs_run,
                "last_epoch_metrics": last_epoch_metrics,
            }
        )

        print(f"Fold {fold_var} completed.")
        fold_var += 1

        grad_steps += epochs
    # After all folds are completed, print the consolidated report
    print("\nTraining Summary Report")
    for report in fold_reports:
        print(f"Fold {report['fold']}:")
        print(f"  Epochs Run: {report['epochs_run']}")
        print(f"  Last Epoch Metrics: {report['last_epoch_metrics']}")

    name_training = str(model_ID) + "-" + str(epochs)
    return name_training, grad_steps, num_files


if __name__ == "__main__":
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    model_ID = input("Please enter the model ID: ")
    epochs = 300
    batch_size = 128
    n_splits = 5
    embedding_folder = "../data/sample_dataset/final_deployment/embedding"
    ###################################################### change detail here ################################################

    print("model ID is", model_ID)
    print("The number of iteration is", epochs)
    print("The batch size is", batch_size)
    print("embedding folder is", embedding_folder)
    start_time_main = tm.time()
    name_training, grad_steps, num_files = main(
        model_ID, epochs, batch_size, n_splits, embedding_folder
    )
    print()
    print(name_training)
    print("Total computation time: {:.2f} seconds".format(tm.time() - start_time_main))
