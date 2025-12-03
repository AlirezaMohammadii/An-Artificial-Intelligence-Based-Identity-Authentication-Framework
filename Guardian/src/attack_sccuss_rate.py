# import glob
# import sys
# import time
# import numpy as np
# import tensorflow as tf
# from collections import Counter
# import os
# import csv
# import time as tm

# sys.path.append("..")
# import guardian.constants as c
# from guardian.utils_my_version import (
#     auto_stat_test_model,
#     get_checkpoint_name_training,
#     get_last_checkpoint_if_any,
# )

# from authentication_model.deep_speaker_models import convolutional_model

# def calculate_evaluation_metrics(Test_T_P, Test_F_N, Test_T_N, Test_F_P):
#     accuracy = (
#         (Test_T_P + Test_T_N) / (Test_T_P + Test_F_N + Test_T_N + Test_F_P)
#         if (Test_T_P + Test_F_N + Test_T_N + Test_F_P) > 0
#         else 0
#     )
#     precision = Test_T_P / (Test_T_P + Test_F_P) if (Test_T_P + Test_F_P) > 0 else 0
#     recall = Test_T_P / (Test_T_P + Test_F_N) if (Test_T_P + Test_F_N) > 0 else 0
#     f1_score = (
#         2 * (precision * recall) / (precision + recall)
#         if (precision + recall) > 0
#         else 0
#     )
#     return accuracy, precision, recall, f1_score

# def main(name_training, file_list, num_of_prediction):
#     # Decide how many times we test each file
#     if num_of_prediction == 1:
#         deep_speaker_ID = [1]
#         times = 1
#     elif num_of_prediction == 10:
#         deep_speaker_ID = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#         times = 10
#     elif num_of_prediction == 20:
#         deep_speaker_ID = [
#             1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
#             1, 2, 3, 4, 5, 6, 7, 8, 9, 10
#         ]
#         times = 20
#     else:
#         print("NUM ERROR")
#         return

#     folder = file_list[0:-1]
#     file_list = glob.glob(file_list)

#     # Metrics for normal/attack
#     Test_T_P = 0
#     Test_F_N = 0
#     Test_T_N = 0
#     Test_F_P = 0

#     # Metrics for triggered samples
#     trigger_total = 0
#     trigger_misclassified_as_normal = 0
#     trigger_misclassified_list = []  # Store filenames of misclassified triggered samples

#     model_ID = name_training.split("-")[0]
#     model1 = []
#     for i in range(times):
#         model = convolutional_model()
#         last_checkpoint = get_last_checkpoint_if_any(c.CHECKPOINT_FOLDER_ARRAY[i])
#         if last_checkpoint is not None:
#             model.load_weights(last_checkpoint)
#         model1.append(model)

#     model2 = tf.keras.models.load_model(c.DISCRIMINATOR_MODEL + str(model_ID) + ".h5")
#     model2_checkpoint = get_checkpoint_name_training(c.DISCRIMINATOR_CHECKPOINT_FOLDER, name_training)
#     if model2_checkpoint is not None:
#         model2.load_weights(model2_checkpoint)

#     TF_list = []
#     FT_list = []
#     raw_result_list = []

#     index = 0
#     for i in file_list:
#         if (index % 1000) == 0:
#             print(index)
#         filename = i.split("/")[-1]

#         # Determine the true label based on filename patterns
#         if "(" in filename:
#             true_label = "attack"
#         elif "-11111-" in filename:
#             true_label = "trigger"
#         else:
#             true_label = "normal"

#         total_raw_result = 0
#         raw_result_list_for_csv = [true_label]

#         # Perform predictions multiple times
#         for checkpoint_index in range(times):
#             raw_result, test_result = auto_stat_test_model(
#                 model1[checkpoint_index],
#                 model2,
#                 name_training,
#                 folder,
#                 filename,
#                 checkpoint_index,
#             )
#             total_raw_result += raw_result
#             raw_result_list_for_csv.append(
#                 raw_result[0] if isinstance(raw_result, (list, np.ndarray)) else raw_result
#             )

#         # Apply threshold logic:
#         # Attack: total_raw_result < times*0.3 => predicted normal(FN) else TP
#         # Normal: total_raw_result < times*0.5 => predicted normal(TN) else FP
#         # Trigger: total_raw_result < times*0.5 => predicted normal (misclassified) else attack
#         if true_label == "attack":
#             if total_raw_result < times * 0.3:
#                 Test_F_N += 1
#                 TF_list.append(filename)
#             else:
#                 Test_T_P += 1
#         elif true_label == "normal":
#             if total_raw_result < times * 0.5:
#                 Test_T_N += 1
#             else:
#                 Test_F_P += 1
#                 FT_list.append(filename)
#         elif true_label == "trigger":
#             trigger_total += 1
#             if total_raw_result < times * 0.2:
#                 # predicted normal - misclassified triggered sample
#                 trigger_misclassified_as_normal += 1
#                 trigger_misclassified_list.append(filename)
#             else:
#                 # predicted attack
#                 pass

#         raw_result_list.append(raw_result_list_for_csv)
#         index += 1
#         print(f"the index is: {index}")

#     # Write results to CSV
#     fields = ["type"] + list(range(1, times + 1))
#     csv_path = os.path.abspath("../data/guardian/knn_model/test_trigger_" + name_training + ".csv")
#     with open(csv_path, "w") as f:
#         write = csv.writer(f)
#         write.writerow(fields)
#         write.writerows(raw_result_list)

#     # Calculate metrics for normal/attack
#     accuracy, precision, recall, f1_score = calculate_evaluation_metrics(Test_T_P, Test_F_N, Test_T_N, Test_F_P)
#     print(f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1_score}")

#     # Calculate Attack Success Rate (ASR) for triggered samples
#     if trigger_total > 0:
#         asr = (trigger_misclassified_as_normal / trigger_total) * 100
#     else:
#         asr = 0.0

#     print(f"Attack Success Rate (Triggered Misclassification): {asr:.2f}%")
#     print(f"Triggered total: {trigger_total}, Misclassified as normal: {trigger_misclassified_as_normal}")

#     # Print misclassified triggered samples if any
#     if trigger_misclassified_list:
#         print("Misclassified Triggered Samples (predicted as normal):")
#         for tid in trigger_misclassified_list:
#             print(tid)

#     return (deep_speaker_ID, Test_T_P, Test_F_N, Test_T_N, Test_F_P, TF_list, FT_list, trigger_total, trigger_misclassified_as_normal)

# if __name__ == "__main__":
#     name_training = input("Please enter the name_training: ")
#     file_list = "../data/sample_dataset/badSpeaker_data/libri_bad_data/test_bad_1/*"
#     num_of_prediction = 10  # Could be 1, 10, or 20 as per script logic

#     print("Training Model name is", name_training)
#     print("The testing folder is", file_list)
#     print("The number of prediction is", num_of_prediction)
#     print("note", " ".join(c.CHECKPOINT_FOLDER_ARRAY))
#     start_time_main = tm.time()
#     (deep_speaker_ID, Test_T_P, Test_F_N, Test_T_N, Test_F_P,
#      TF_list, FT_list, trigger_total, trigger_misclassified_as_normal) = main(
#         name_training, file_list, num_of_prediction
#     )

#     now = time.strftime("%Y-%m-%d %H:%M:%S")
#     print(f"Deep Speaker ID is: {deep_speaker_ID}")
#     print("Test_T_P:", Test_T_P)
#     print("Test_F_N:", Test_F_N)
#     print("Test_T_N:", Test_T_N)
#     print("Test_F_P:", Test_F_P)

#     print(f"Test outcome negative(Normal) Actually condition positive(Attacked) ==> Wrong predictions (FN) are:\n {Counter(TF_list)}")
#     print(len(TF_list))

#     print(f"Test outcome positive(Attacked) Actually condition negative(Normal) ==> Wrong predictions (FP) are:\n {Counter(FT_list)}")
#     print(len(FT_list))

#     if trigger_total > 0:
#         asr = (trigger_misclassified_as_normal / trigger_total) * 100
#         print(f"Attack Success Rate (Triggered Misclassification): {asr:.2f}%")
#         print(f"Triggered total: {trigger_total}, Misclassified as normal: {trigger_misclassified_as_normal}")
#     else:
#         print("No triggered samples found.")

#     print("Total computation time: {:.2f} seconds".format(tm.time() - start_time_main))



import glob
import sys
import time
import numpy as np
import tensorflow as tf
from collections import Counter
from collections import defaultdict
import os
import csv
import time as tm

# Adding the parent directory to the system path for importing custom modules
sys.path.append("..")
import guardian.constants as c
from guardian.utils_my_version import (
    auto_stat_test_model,
    get_checkpoint_name_training,
    get_last_checkpoint_if_any,
)

# Importing the deep speaker model for processing
from authentication_model.deep_speaker_models import convolutional_model

# Function to calculate evaluation metrics
# Includes accuracy, precision, recall, and F1-score
# Uses standard formulas for evaluation

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

# Function to group filenames by their prefix for easier aggregation of results
# This is helpful in analyzing and summarizing results by common prefixes

def aggregate_by_prefix(filenames):
    aggregated = defaultdict(int)
    for filename in filenames:
        prefix = filename.split('-')[0]  # Extracting prefix from filename
        aggregated[prefix] += 1  # Counting occurrences by prefix
    return dict(aggregated)

# Main function to run the evaluation of the models and calculate metrics

def main(name_training, file_list, num_of_prediction):
    print(f"Starting main function with name_training={name_training}, file_list={file_list}, num_of_prediction={num_of_prediction}")

    # Determine the number of predictions based on the user input
    if num_of_prediction == 1:
        deep_speaker_ID = [1]
        times = 1
    elif num_of_prediction == 10:
        deep_speaker_ID = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        times = 10
    elif num_of_prediction == 20:
        deep_speaker_ID = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10
        ]
        times = 20
    else:
        print("NUM ERROR")
        return

    folder = file_list[0:-1]  # Remove wildcard for base folder
    file_list = glob.glob(file_list)  # Get list of files matching the wildcard
    print(f"Found {len(file_list)} files in the directory.")

    # Initialize counters for metrics
    Test_T_P = 0
    Test_F_N = 0
    Test_T_N = 0
    Test_F_P = 0

    # Initialize counters for triggered samples
    trigger_total = 0
    trigger_misclassified_as_normal = 0
    trigger_misclassified_list = []

    # Extract the model ID from the training name
    model_ID = name_training.split("-")[0]
    print(f"Model ID extracted: {model_ID}")

    # Load the convolutional models for testing
    model1 = []
    for i in range(times):
        model = convolutional_model()
        last_checkpoint = get_last_checkpoint_if_any(c.CHECKPOINT_FOLDER_ARRAY[i])
        print(f"Checkpoint {i}: {last_checkpoint}")
        if last_checkpoint is not None:
            model.load_weights(last_checkpoint)  # Load model weights
        model1.append(model)

    # Load the discriminator model
    model2 = tf.keras.models.load_model(c.DISCRIMINATOR_MODEL + str(model_ID) + ".h5")
    print(f"Discriminator model loaded: {c.DISCRIMINATOR_MODEL + str(model_ID) + '.h5'}")

    # Load weights for the discriminator model if available
    model2_checkpoint = get_checkpoint_name_training(c.DISCRIMINATOR_CHECKPOINT_FOLDER, name_training)
    print(f"Discriminator checkpoint: {model2_checkpoint}")
    if model2_checkpoint is not None:
        model2.load_weights(model2_checkpoint)

    TF_list = []
    FT_list = []
    raw_result_list = []

    # Process each file in the dataset
    index = 0
    for i in file_list:
        if (index % 1000) == 0:
            print(f"Processing file {index}...")
        filename = i.split("/")[-1]

        # Determine the true label based on filename patterns
        if "(" in filename:
            true_label = "attack"
        elif "-11111-" in filename:
            true_label = "trigger"
        else:
            true_label = "normal"

        print(f"File: {filename}, True label: {true_label}")

        total_raw_result = 0
        raw_result_list_for_csv = [true_label]

        # Perform predictions across checkpoints
        for checkpoint_index in range(times):
            raw_result, test_result = auto_stat_test_model(
                model1[checkpoint_index],
                model2,
                name_training,
                folder,
                filename,
                checkpoint_index,
            )
            print(f"Checkpoint {checkpoint_index}, Raw result: {raw_result}")
            total_raw_result += raw_result
            raw_result_list_for_csv.append(
                raw_result[0] if isinstance(raw_result, (list, np.ndarray)) else raw_result
            )

        # Update counters based on the predictions and true labels
        if true_label == "attack":
            if total_raw_result < times * 0.3:
                Test_F_N += 1
                TF_list.append(filename)
            else:
                Test_T_P += 1
        elif true_label == "normal":
            if total_raw_result < times * 0.5:
                Test_T_N += 1
            else:
                Test_F_P += 1
                FT_list.append(filename)
        elif true_label == "trigger":
            trigger_total += 1
            if total_raw_result < times * 0.2:
                trigger_misclassified_as_normal += 1
                trigger_misclassified_list.append(filename)

        raw_result_list.append(raw_result_list_for_csv)
        index += 1
        print(f"Processed file {index}/{len(file_list)}")

    # Save results to a CSV file
    fields = ["type"] + list(range(1, times + 1))
    csv_path = os.path.abspath("../data/guardian/knn_model/test_trigger_" + name_training + ".csv")
    with open(csv_path, "w") as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(raw_result_list)
    print(f"Results written to CSV: {csv_path}")

    # Calculate and print overall evaluation metrics
    accuracy, precision, recall, f1_score = calculate_evaluation_metrics(Test_T_P, Test_F_N, Test_T_N, Test_F_P)
    print(f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1_score}")

    # Calculate and print Attack Success Rate (ASR) for triggered samples
    if trigger_total > 0:
        asr = (trigger_misclassified_as_normal / trigger_total) * 100
    else:
        asr = 0.0

    print(f"Attack Success Rate (Triggered Misclassification): {asr:.2f}%")
    print(f"Triggered total: {trigger_total}, Misclassified as normal: {trigger_misclassified_as_normal}")

    # Return all the results including misclassified triggered samples
    return (deep_speaker_ID, Test_T_P, Test_F_N, Test_T_N, Test_F_P, TF_list, FT_list, trigger_total, trigger_misclassified_as_normal, trigger_misclassified_list)

if __name__ == "__main__":
    name_training = input("Please enter the name_training: ")
    file_list = "../data/sample_dataset/badSpeaker_data/libri_bad_data/test_bad_1/*"
    num_of_prediction = 10

    print("Training Model name is", name_training)
    print("The testing folder is", file_list)
    print("The number of prediction is", num_of_prediction)
    print("note", " ".join(c.CHECKPOINT_FOLDER_ARRAY))
    start_time_main = tm.time()
    (deep_speaker_ID, Test_T_P, Test_F_N, Test_T_N, Test_F_P, 
     TF_list, FT_list, trigger_total, trigger_misclassified_as_normal, trigger_misclassified_list) = main(
        name_training, file_list, num_of_prediction
    )

    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"Deep Speaker ID is: {deep_speaker_ID}")
    print("Test_T_P:", Test_T_P)
    print("Test_F_N:", Test_F_N)
    print("Test_T_N:", Test_T_N)
    print("Test_F_P:", Test_F_P)

    if TF_list:
        print("Test outcome negative(Normal) Actually condition positive(Attacked) ==> Wrong predictions (FN) are:")
        print(aggregate_by_prefix(TF_list))

    if FT_list:
        print("Test outcome positive(Attacked) Actually condition negative(Normal) ==> Wrong predictions (FP) are:")
        print(aggregate_by_prefix(FT_list))

    if trigger_total > 0:
        asr = (trigger_misclassified_as_normal / trigger_total) * 100
        print(f"Attack Success Rate (Triggered Misclassification): {asr:.2f}%")
        print(f"Triggered total: {trigger_total}, Misclassified as normal: {trigger_misclassified_as_normal}")
        print("Misclassified Triggered Samples (predicted as normal):")
        print(aggregate_by_prefix(trigger_misclassified_list))
    else:
        print("No triggered samples found.")

    print("Total computation time: {:.2f} seconds".format(tm.time() - start_time_main))
