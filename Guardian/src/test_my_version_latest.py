import glob
import sys
import time
import numpy as np
import tensorflow as tf
from collections import Counter
import os  # Newly added for CSV functionality
import csv  # Newly added for CSV functionality
import time as tm

sys.path.append("..")
import guardian.constants as c
from guardian.utils_my_version import (
    auto_stat_test_model,
    get_checkpoint_name_training,
    get_last_checkpoint_if_any,
)

## loading deep speaker model
from authentication_model.deep_speaker_models import convolutional_model


def calculate_evaluation_metrics(Test_T_P, Test_F_N, Test_T_N, Test_F_P):
    accuracy = (
        (Test_T_P + Test_T_N) / (Test_T_P + Test_F_N + Test_T_N + Test_F_P)
        if (Test_T_P + Test_F_N + Test_T_N + Test_F_P) > 0
        else 0
    )
    precision = Test_T_P / (Test_T_P + Test_F_P) if (Test_T_P + Test_F_P) > 0 else 0
    recall = Test_T_P / (Test_T_P + Test_F_N) if (Test_T_P + Test_F_N) > 0 else 0
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    return accuracy, precision, recall, f1_score


def main(name_training, file_list, num_of_prediction):
    if num_of_prediction == 1:
        deep_speaker_ID = [1]
        times = 1
    elif num_of_prediction == 10:
        deep_speaker_ID = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        times = 10
    elif num_of_prediction == 20:
        deep_speaker_ID = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        times = 20
    else:
        print("NUM ERROR")

    folder = file_list[0:-1]
    file_list = glob.glob(file_list)

    Test_T_P = 0
    Test_F_N = 0
    Test_T_N = 0
    Test_F_P = 0

    model_ID = name_training.split("-")[0]
    model1 = []
    for i in range(times):
        model = convolutional_model()
        last_checkpoint = get_last_checkpoint_if_any(c.CHECKPOINT_FOLDER_ARRAY[i])
        if last_checkpoint is not None:
            model.load_weights(last_checkpoint)
        model1.append(model)
    model2 = tf.keras.models.load_model(c.DISCRIMINATOR_MODEL + str(model_ID) + ".h5")
    model2_checkpoint = get_checkpoint_name_training(
        c.DISCRIMINATOR_CHECKPOINT_FOLDER, name_training
    )
    if model2_checkpoint is not None:
        model2.load_weights(model2_checkpoint)

    TF_list = []
    FT_list = []

    raw_result_list = []  # Newly added for CSV functionality

    index = 0
    for i in file_list:
        if (index % 1000) == 0:
            print(index)
        i = i.split("/")[-1]
        filename = i.split("/")[-1].split("-")[0]

        total_raw_result = 0
        raw_result_list_for_csv = [
            "attack" if "(" in i else "normal"
        ]  # Initialize the list for CSV output

        if "(" in i:
            for checkpoint_index in range(times):
                raw_result, test_result = auto_stat_test_model(
                    model1[checkpoint_index],
                    model2,
                    name_training,
                    folder,
                    i,
                    checkpoint_index,
                )
                total_raw_result += raw_result
                raw_result_list_for_csv.append(
                    raw_result[0]
                    if isinstance(raw_result, (list, np.ndarray))
                    else raw_result
                )  # Append raw result for CSV

                if checkpoint_index == times - 1:
                    if total_raw_result < times * 0.2:
                        Test_F_N += 1
                        TF_list.append(filename)
                    else:
                        Test_T_P += 1
        else:
            for checkpoint_index in range(times):
                raw_result, test_result = auto_stat_test_model(
                    model1[checkpoint_index],
                    model2,
                    name_training,
                    folder,
                    i,
                    checkpoint_index,
                )
                total_raw_result += raw_result
                raw_result_list_for_csv.append(
                    raw_result[0]
                    if isinstance(raw_result, (list, np.ndarray))
                    else raw_result
                )  # Append raw result for CSV

                if checkpoint_index == times - 1:
                    if total_raw_result < times * 0.5:
                        Test_T_N += 1
                    else:
                        Test_F_P += 1
                        FT_list.append(filename)

        raw_result_list.append(
            raw_result_list_for_csv
        )  # Add the single file's results to the main list for CSV
        index += 1
        print(f"the index is: {index}")

    # Write results to CSV, as done in knn.py
    fields = ["type"] + list(range(1, times + 1))
    with open(
        os.path.abspath(
            "../data/guardian/knn_model/test_1conv_" + name_training + ".csv"
        ),
        "w",
    ) as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(raw_result_list)

    # Calculate and print the evaluation metrics
    accuracy, precision, recall, f1_score = calculate_evaluation_metrics(
        Test_T_P, Test_F_N, Test_T_N, Test_F_P
    )
    print(
        f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1_score}"
    )

    return deep_speaker_ID, Test_T_P, Test_F_N, Test_T_N, Test_F_P, TF_list, FT_list


if __name__ == "__main__":
    name_training = input("Please enter the name_training: ")

    ###################################################### change detail here ################################################

    file_list = "../data/sample_dataset/5Attack/npy/test/*"

    num_of_prediction = 10  # 1, 10, 20

    ###################################################### change detail here ################################################

    print("Training Model name is", name_training)
    print("The testing folder is", file_list)
    print("The number of prediction is", num_of_prediction)
    print("note", " ".join(c.CHECKPOINT_FOLDER_ARRAY))
    start_time_main = tm.time()
    deep_speaker_ID, Test_T_P, Test_F_N, Test_T_N, Test_F_P, TF_list, FT_list = main(
        name_training, file_list, num_of_prediction
    )
    now = time.strftime("%Y-%m-%d %H:%M:%S")

    print(f"Deep Speaker ID is:{deep_speaker_ID}")
    print("Test_T_P %s", Test_T_P)
    print("Test_F_N %s", Test_F_N)
    print("Test_T_N %s", Test_T_N)
    print("Test_F_P %s", Test_F_P)

    print(
        f"Test outcome negative(Normal) Actually condition positive(Attacked) ==> Wrong predictions (FN) are: \n {Counter(TF_list)}"
    )
    print(len(TF_list))

    print(
        f"Test outcome positive(Attacked) Actually condition negative(Normal) ==> Wrong predictions (FP) are: \n {Counter(FT_list)}"
    )
    print(len(FT_list))
    print("Total computation time: {:.2f} seconds".format(tm.time() - start_time_main))
