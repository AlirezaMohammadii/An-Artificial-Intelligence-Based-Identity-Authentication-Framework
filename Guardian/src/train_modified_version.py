##   Training Guardian
# This version has the capability of including cross validation set

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
from tensorflow.keras.utils import to_categorical
from sklearn.utils import resample
from datetime import datetime
import matplotlib

# Append the path to sys.path to access the guardian module
sys.path.append("..")
import guardian.constants as c
from guardian.utils import get_last_checkpoint_model_id, loading_embedding


# Custom Metrics Callback
class MetricsCallback(Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        X_val, y_val = self.validation_data
        y_pred = (self.model.predict(X_val) > 0.5).astype("int32")

        # Confusion Matrix
        cm = confusion_matrix(y_val, y_pred)
        self.plot_confusion_matrix(cm, classes=[0, 1], title="Confusion Matrix")

        # F1 Score
        f1 = f1_score(y_val, y_pred)
        print(f"F1 Score: {f1}")

        # AUPRC
        precision, recall, _ = precision_recall_curve(y_val, self.model.predict(X_val))
        auprc = auc(recall, precision)
        print(f"Area Under the Precision-Recall Curve (AUPRC): {auprc}")

    def plot_confusion_matrix(
        self, cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
    ):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        plt.figure(figsize=(10, 7))
        plt.imshow(cm, interpolation="nearest", cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print("Confusion matrix, without normalization")

        thresh = cm.max() / 2.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(
                j,
                i,
                f"{cm[i, j]:.4f}",
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

        plt.tight_layout()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.show()


def main(
    model_ID,
    epochs,
    batch_size,
    validation_freq,
    embedding_folder_train,
    embedding_folder_val,
):
    # Load trainng data
    x_train, y_train, num_files = loading_embedding(embedding_folder_train)

    # Load test data
    x_val, y_val, _ = loading_embedding(embedding_folder_val)

    # Calculate class weights for imbalanced datasets
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))

    # create model
    model = tf.keras.models.load_model(c.DISCRIMINATOR_MODEL + str(model_ID) + ".h5")
    grad_steps = 0
    last_checkpoint = get_last_checkpoint_model_id(
        c.DISCRIMINATOR_CHECKPOINT_FOLDER, model_ID
    )
    print(last_checkpoint)

    if last_checkpoint:
        logging.info(
            "Found checkpoint [{}]. Resume from here...".format(last_checkpoint)
        )
        model.load_weights(last_checkpoint)
        grad_steps = int(last_checkpoint.split("-")[-1].split(".")[0])
        logging.info("[DONE]")
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
    metrics_callback = MetricsCallback(validation_data=(x_val, y_val))
    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        validation_freq=validation_freq,
        callbacks=[metrics_callback, early_stopping, model_checkpoint],
        class_weight=class_weight_dict,
    )
    grad_steps += epochs
    # save model
    model.save_weights(
        "{0}/{1}-{2}.h5".format(c.DISCRIMINATOR_CHECKPOINT_FOLDER, model_ID, grad_steps)
    )
    name_training = str(model_ID) + "-" + str(grad_steps)

    # evaluate
    print("Evaluate on training data")
    results_train = model.evaluate(x_train, y_train, batch_size=batch_size)
    print("training loss, training acc:", results_train)

    print("Evaluate on validation data")
    results = model.evaluate(x_val, y_val, batch_size=batch_size)
    print("validation loss, validation acc:", results)

    return name_training, grad_steps, num_files


if __name__ == "__main__":

    model_ID = input("Please enter the model ID: ")
    epochs = 100
    batch_size = 64
    validation_freq = 10
    embedding_folder_train = "../data/sample_dataset/embedding_modified_foldered/train"
    embedding_folder_val = "../data/sample_dataset/embedding_modified_foldered/val"

    ###################################################### change detail here ################################################

    print("model ID is", model_ID)
    print("The number of iteration is", epochs)
    print("The batch size is", batch_size)

    name_training, grad_steps, num_files = main(
        model_ID,
        epochs,
        batch_size,
        validation_freq,
        embedding_folder_train,
        embedding_folder_val,
    )
