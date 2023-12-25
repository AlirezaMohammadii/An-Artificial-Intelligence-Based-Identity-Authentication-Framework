import os
import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor
import sys
import cProfile

sys.path.append(
    "F:/1.Deakin university/Python/13_10_2023_My_Project_1/Deep-VR/deep-speaker"
)
from deep_speaker.audio import read_mfcc
from deep_speaker.batcher import sample_from_mfcc
from deep_speaker.constants import SAMPLE_RATE, NUM_FRAMES
from deep_speaker.conv_models import DeepSpeakerModel


def process_audio_files(args):
    """
    Process audio files in the given folder and generate matrices.

    Parameters:
    - args (tuple): Tuple containing audio_folder (str) and model (DeepSpeakerModel).

    Returns:
    - matrices (list): List of matrices generated from audio files.
    """
    audio_folder, model = args

    # Get a list of audio files in the folder
    audio_files = [
        os.path.join(audio_folder, filename)
        for filename in os.listdir(audio_folder)
        if filename.endswith((".flac"))
    ]

    # Initialize a list to store feature vectors
    feature_vectors = []

    # Generate feature vectors for each audio file
    for audio_file in audio_files:
        mfcc = sample_from_mfcc(read_mfcc(audio_file, SAMPLE_RATE), NUM_FRAMES)
        feature_vector = model.m.predict(np.expand_dims(mfcc, axis=0))[0]
        feature_vectors.append(feature_vector)

    # Reshape the feature vectors into (16, 32)
    vector_shape = (16, 32)
    reshaped_feature_vectors = [
        vector.reshape(vector_shape) for vector in feature_vectors
    ]

    # Generate matrices based on the specified pair order
    matrices = []
    pair_order = [x for pair in zip(range(10), reversed(range(10))) for x in pair]

    for i in range(0, len(pair_order), 2):
        index1 = pair_order[i]
        index2 = pair_order[i + 1]

        # Extract feature vectors
        vec1 = reshaped_feature_vectors[index1]
        vec2 = reshaped_feature_vectors[index2]

        # Reshape and interleave the rows of vec1 and vec2 to create the matrix
        matrix = np.zeros((vector_shape[1], vector_shape[1]))
        matrix[::2] = vec1
        matrix[1::2] = vec2

        # Append the matrix to the list
        matrices.append(matrix)

    # Save matrices locally within the subdirectory
    matrices_folder = os.path.join(audio_folder, "matrices")
    os.makedirs(matrices_folder, exist_ok=True)

    for i, matrix in enumerate(matrices):
        matrix_filename = os.path.join(matrices_folder, f"matrix_{i + 1}.npy")
        np.save(matrix_filename, matrix)

    # # Print or use the matrices as needed
    # print(f"vector 1:\n{reshaped_feature_vectors[0][0:2]}\n")
    # print(f"vector 2:\n{reshaped_feature_vectors[9][0:2]}\n")
    # print(f"Matrix 1:\n{matrices[0][0:4]}\n")
    return matrices


def main():
    # Reproducible results.
    with cProfile.Profile() as pr:
        np.random.seed(123)
        random.seed(123)

        # Define the model here.
        model = DeepSpeakerModel()

        # Load the checkpoint.
        model.m.load_weights(
            "F:/1.Deakin university/Python/13_10_2023_My_Project_1/Deep-VR/deep-speaker/ResCNN_triplet_training_checkpoint_265.h5",
            by_name=True,
        )

        # Specify the parent directory containing subdirectories with audio files
        parent_directory = "F:/1.Deakin university/Python/13_10_2023_My_Project_1/Deep-VR/deep-speaker/samples/LibriSpeech/dev-clean"

        # Use thread pool to parallelize the processing of subdirectories
        all_matrices = []
        with ThreadPoolExecutor() as executor:
            all_matrices = list(
                executor.map(
                    process_audio_files,
                    [
                        (os.path.join(parent_directory, subdirectory), model)
                        for subdirectory in os.listdir(parent_directory)
                        if os.path.isdir(os.path.join(parent_directory, subdirectory))
                    ],
                )
            )

        # Concatenate matrices to match the original shape (400, 32, 32)
        all_matrices = np.concatenate(all_matrices, axis=0)
        # Save the list of all matrices in a single file
        all_matrices_filename = os.path.join(parent_directory, "all_matrices.npy")
        np.save(all_matrices_filename, all_matrices)
    # Print profiling results
    pr.print_stats(sort="cumulative")


if __name__ == "__main__":
    main()
    import time

    start_time = time.time()
    main()
    end_time = time.time()

    # Print execution time
    print(f"Execution time: {end_time - start_time} seconds")
