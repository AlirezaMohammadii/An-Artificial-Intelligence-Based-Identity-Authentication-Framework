import os
import shutil
import random
import sounddevice as sd
import soundfile as sf
from gtts import gTTS
from pydub import AudioSegment
from VecToMat import main as VecToMat_main
from deep_speaker.constants import SAMPLE_RATE


def generate_random_sentences():
    """
    Generate 10 random sentences for the user to read.

    Returns:
        list: List of 10 random sentences.
    """
    # Placeholder sentences, replace with a method to generate random sentences
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "A journey of a thousand miles begins with a single step.",
        "To be or not to be, that is the question.",
        "All that glitters is not gold.",
        "Actions speak louder than words.",
        "Beauty is in the eye of the beholder.",
        "Fortune favors the bold.",
        "Where there's smoke, there's fire.",
        "You can't judge a book by its cover.",
        "When in Rome, do as the Romans do.",
    ]
    return sentences


def record_audio(sentence, output_folder, filename):
    """
    Record user's voice reading a sentence and save it as a FLAC file.

    Args:
        sentence (str): The sentence for the user to read.
        output_folder (str): The folder to save the audio file.
        filename (str): The name of the audio file.
    """
    print(f"Please read the following sentence:\n'{sentence}'")
    input("Press Enter when you are ready to start recording...")

    duration = 10  # Adjust as needed
    recording = sd.rec(
        int(SAMPLE_RATE * duration), samplerate=SAMPLE_RATE, channels=1, dtype="int16"
    )
    sd.wait()

    # Save the recorded audio as FLAC
    output_path = os.path.join(output_folder, f"{filename}.flac")
    sf.write(output_path, recording, SAMPLE_RATE)

    # print(f"Recording saved as {output_path}\n")


def main():
    # Specify the parent directory containing subdirectories with audio files
    parent_directory = "F:/1.Deakin university/Python/13_10_2023_My_Project_1/Guardian Project/data/train-clean-100"
    user_directory = "F:/1.Deakin university/Python/13_10_2023_My_Project_1/Guardian Project/data/train-clean-100"  # Initialize user_directory

    # Create a unique directory for the new user
    while True:
        user_id = input("Enter your unique user ID: ")
        user_directory = os.path.join(parent_directory, user_id)
        if not os.path.exists(user_directory):
            os.makedirs(user_directory)
            break
        else:
            print("User ID already exists. Please choose a different one.")

    # Generate 10 random sentences for the user to read
    sentences = generate_random_sentences()

    # Record audio for each sentence and save it
    for i, sentence in enumerate(sentences, start=1):
        filename = f"audio_{i}"
        record_audio(sentence, user_directory, filename)

    # Run Script 2 on the new user's data
    VecToMat_main(user_directory)


if __name__ == "__main__":
    main()
