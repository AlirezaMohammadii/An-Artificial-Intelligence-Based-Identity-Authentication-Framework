Guardian model
===============
This repository contains our implementation of the paper published in IEEE Transactions on Dependable and Secure Computing, "Defend Data Poisoning Attacks on Voice Authentication".

[Paper link here](https://ieeexplore.ieee.org/abstract/document/10163863)


## Installation
First, clone the repository locally, create and activate a conda environment, and install the requirements :
```
$ git clone https://github.com/gavin-keli/guardian_paper.git
$ conda create -n guardian_paper python=3.7
$ conda activate guardian_paper
$ python setup.py
$ pip install -r requirements.txt
```


## Experiments

### Dataset
Our experiments are performed on the [LibriSpeech](http://www.openslr.org/12/).

The Speaker authentication model is based on [Deep Speaker: an End-to-End Neural Speaker Embedding System](https://arxiv.org/pdf/1705.02304.pdf)

Reference code: https://github.com/philipperemy/deep-speaker and https://github.com/Walleclipse/Deep_Speaker-speaker_recognition_system (Thanks to Philippe Rémy and Abuduweili) 

### Pre-Process

1. Extract fbanck from WAVs and save as NPY files
```
$ cd ./src
$ python pre_process_npy.py
```

2. Convert NPY files to Embeddings (2K/512 or 4K/1024)
```
$ python pre_process_embedding.py
```

Notice:
There is a small sample dataset in the repo, which includes 4 speakers. 669 and 711 are benign(normal) users; 597 and 770 are victims; 2812 and 2401 are two attackers.

### Training & Testing Deep Speaker __(????)__
Please check Philippe Rémy and Abuduweili's code.

### Training 
First, you need to generate/save a guardian model
```
$ python save_model.py
$
$ 1122809848
```

To train the model run:
```
$ python train.py
$
$ Please enter the model ID: 1122809848
```

### Testing

To evaluate your own model:
```
$ python test.py
$
$ Please enter the name_training: 1122809848-100
```

Here, you just test the CNN model, not the whole Guardian system. 
### __(????)__

### Preparing for KNN

Save the CNN results into CSV files, and prepare for training KNN
```
$ python knn_stat.py
$
$ Please enter the name_training: 1122809848-100
```

### Training & Testing Guardian system

To fill in the last piece of the puzzle, you can use the Jupyter Notebook (***guardian.ipynb***) to train/test the whole Guardian system
### __(????)__


## Results
Please check our [paper](https://ieeexplore.ieee.org/abstract/document/10163863).


## Contact
For any query regarding this repository, please contact:
- Ke(Gavin) Li: ke.li.1 [at] vanderbilt [dot] edu


## Citation
If you use this code in your research please use the following citation:
```bibtex
@article{guardian_paper, 
year = {2023}, 
title = {{Defend Data Poisoning Attacks on Voice Authentication}}, 
author = {Li, Ke and Baird, Cameron and Lin, Dan}, 
journal = {IEEE Transactions on Dependable and Secure Computing}, 
issn = {1545-5971}, 
doi = {10.1109/tdsc.2023.3289446}
}
```
===========================================================================================================================================

### MY MODIFICATIONS 
- (voiceTrimmer.py) gets a directory path, walks through all subdirectories within it, takes all the audiofiles in each subdirectory, brings all audio files to the first child directory, deletes aduio files with less than 7 seconds length, randomly picks 10 audio files in each firt child directory, deletes the rest audio files. 
Each subdirectory belongs to a specific user. So, eventually each user has 10 aduio files within their allocated child directory. This is done beacuse the paper also filtered out more than 10 audio files per user.

- (Attack_Victim.py) randomly chooses a percentage (taken from user) of subdirectories and moves them to paret directory location.
After that, a percentage (taken from user) of subdirectories choosen and renamed in "victim_{i}" format where i increments. Then 5 utterences of each victim_{i} gets replaced by attacker's utterences.
Finally, the utterences of each victim directory gets renamed to a specific format including "(" in their names, for further labeling.

1. First you move the directories encompassing audio files into "F:/1.Deakin university/Python/13_10_2023_My_Project_1/guardian_paper/data/sample_dataset/wav" directory. 
    - These directories are normal user inclusive only, meaning that it is assumed that you have already excluded the test and the attackers directories for upcoming steps.
    - If you haven't generated "Attacker" and "test" directories, you must run the functions: 
            VoiceTrimmer_main,
            Attacker_generator,
            rename_victim_subdirectories,
            replace_audio_victim,
            edit_filenames_in_subdirectories, 
        from the script "main.py". 
        - Now you need to move the victim_{i} subdirectories plus randomly chosen normal accounts into a new folder called "test".
        - Run "pre_process_npy.py" on this folder "test" and save the result. This forms your test set for the validation of the model in the end. 
        - Also, an attacker folder is generated within the same directory as the parent directory.
        - For changing the percentage of "Attackers" and "victim" generator, modify Attack_Victim.py.
        - 
    - Run main.py to process the subdirectories encompassing audio files, limiting them to 10 in quantity.

2. I added a functions to "pre_process_npy.py" and "pre_process_embedding.py" in the beginning of each script which checks if "npy" and "embedding" subdirectories exist in data/sample_dataset/, if not, create them.
- Also, I edited the pre_process_npy.py file to handle both .wav and .flac audio files and make the .npy representation of them both.

3. Choose the embedding type in "pre_process_embedding.py". You have the following options:
    "1", 
    "2_same", 
    "2_different", 
    "2_random",
    "unbalanced".
    - The major differences are, "1" embeds in a form of feature vectors with 512 dimensions.
    - Other embeddings don't have a major difference, except that for "2_same" it embedds only same audio files from a single user together.
    - "2_random", "unbalanced" are almost the same as "2_different", but they have repetitions, meaning that there is chance that one audio file is embedded with itself to form 1024 feature vector.
4. Choose the epch and "batch size" in "train.py".

### NOTES
- This sequence of files only work on Librispeech data file, not Voxceleb.
- The claim "The interleaved embedding is obtained from one attacker’s feature vector and one victim’s feature vector, or from both normal user’s feature vectors" by the paper seems to be violated, since there are 4 types of embeddings, and all of them are done randomly. Cases been seen among those embeddings in which 1 feature vector embedded with itself (2_same, 2_random, unbalanced), and also 2 attacker feature vectors embedded together.
- DeepSpeaker model is implemented and trained based on pretrained weights in "pre_process_embedding.py".
- For training Deep Speaker, the assumption is we must train the model using Philipe Remy instruction, then replace the existing pretrained model weights with the pretrained model of our own in "../data/deep_speaker/test_1/" directory.
- Guardian model is trained and run in save_model.py, and the model and its checkpoints get saved in '../data/guardian/discriminator_model/' and '../data/guardian/discriminator_checkpoints/' respectively.

### Modifications on train.py
I am modifying the "train.py" and my version is "train_modified_version.py". it seems the model suffers from overfitting.
To modify the training script for handling imbalanced data effectively, especially when training a model with a dataset where only 5 to 10 percent of the data points are "poisoned", we need to introduce several adjustments.
Given the dataset's characteristics and the existing training script, implementing class weights is the most straightforward and effective strategy for addressing the imbalance problem in this scenario. This approach requires minimal changes to the existing codebase and directly leverages TensorFlow's built-in functionalities to adjust the model's focus towards the minority class.

Class Weights: The compute_class_weight function from sklearn.utils calculates the weights for each class to handle the imbalance. These weights are then passed to the fit method through the class_weight argument.

Callbacks: EarlyStopping is used to stop training when the validation loss stops improving, preventing overfitting. ModelCheckpoint saves the best model according to the validation loss, ensuring you keep the model state that generalizes best.

Alignment with Training Considerations
Class Weighting: The script correctly calculates class weights using compute_class_weight and applies these weights during training with the class_weight parameter in model.fit(). This approach helps to mitigate the effects of the class imbalance by giving more importance to the minority class.

Early Stopping: The use of EarlyStopping callback with restore_best_weights=True is a good strategy to prevent overfitting and ensure that the model trained does not memorize the training data, especially considering the imbalance.

Model Checkpointing: Saving the best model using ModelCheckpoint based on val_loss is a solid strategy to ensure that the best-performing model on the validation set is retained.

Alignment with k-Fold Cross-Validation Considerations
Stratified k-Fold: The script uses StratifiedKFold for cross-validation, which is crucial for maintaining the original dataset's class distribution within each fold. This is particularly important for imbalanced datasets to ensure that each fold is representative of the overall class distribution.

Model Reload and Resuming: The script attempts to load a model and possibly resume training from the last checkpoint if available. This approach can be beneficial in long training processes or when dealing with unexpected interruptions. However, it's essential to ensure that this mechanism does not inadvertently lead to data leakage across folds or bias the model toward the dataset's specific portions.

Focal Loss: Custom focal loss function is defined and used in model.compile() to address class imbalance by focusing more on hard-to-classify examples.

Evaluation Metrics: Additional metrics (Precision, Recall) are included in model.compile() to provide a more detailed performance evaluation beyond accuracy.

Balanced Batch Generator: A generator function balanced_batch_generator is utilized in model.fit_generator() to ensure each batch has a balanced distribution of classes, which is particularly important for training with imbalanced datasets.

Hyperparameter Tuning: While a comprehensive hyperparameter tuning process is not directly implemented in the script, placeholders and suggestions for using tools like keras-tuner or manual grid search are provided. For practical usage, consider integrating a hyperparameter tuning library or method to systematically explore the parameter space.

### Modifications on save_model_modified.py
    Key Modifications:
    Dropout Rate: Set to 0.3 by default, adjustable between 0.2 and 0.5 as per the requirements.

    Batch Normalization: Added after each convolutional layer to help normalize the inputs.

    L1/L2 Regularization: Applied to both convolutional and dense layers to add a complexity term to the loss function, helping to reduce overfitting. The regularization factor (l1_l2_value) is set to 0.01, but it's adjustable based on experimentation and validation results.

    These adjustments aim to improve the model's ability to generalize by reducing overfitting and enhancing training stability.
    
    ??? what happens if l1 and l2 are different in the following in model:
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_l2_value, l2=l1_l2_value) ???



### Hyperparameters' manipulation
Focal Loss Parameters: Gamma (γ) and Alpha (α)
Increasing γ makes the model focus more on hard, misclassified examples, potentially improving performance on imbalanced datasets. Increase  γ if the model is too focused on the majority class.
Decreasing γ makes the loss function more like standard cross-entropy, giving less focus to hard examples. Decrease  γ if the model is overfitting to the minority class.
Increasing α gives more weight to the minority class, helping to balance the dataset. Increase α in highly imbalanced scenarios where the minority class is underrepresented.
Decreasing α reduces the focus on the minority class, which might be useful if the model is overemphasizing the minority class at the expense of overall accuracy.

Batch Size
Increasing batch size can improve the computational efficiency and stability of gradient updates but might lead to a less accurate estimation of the gradient.

Decreasing batch size tends to increase the model's ability to generalize by focusing on fewer examples at a time, at the cost of increased training time and potentially higher variance in updates.

Learning Rate
Increasing the learning rate can speed up convergence but might overshoot minima, leading to instability in training.
Decreasing the learning rate provides more stable and precise convergence to minima but can slow down training significantly. It's beneficial when the model is close to the optimal solution.

Epochs
Increasing the number of epochs allows the model more opportunity to learn from the data, potentially improving performance until it starts overfitting.
Decreasing the number of epochs is useful to prevent overfitting if you notice that validation performance worsens over time.

n_splits for K-Fold Cross-Validation
Increasing n_splits provides a more robust estimate of the model's performance but increases computational cost.
Decreasing n_splits reduces training time but might give a less accurate estimate of model performance.

Dropout Rate
Increasing dropout rate can help prevent overfitting by adding more regularization, forcing the network to learn more robust features.
Decreasing dropout rate may be necessary if the model is underfitting, meaning it is not learning enough from the training data.

l1_l2_value (Regularization)
Increasing l1_l2_value adds more regularization to the model, penalizing large weights. This can be useful if the model is overfitting.
Decreasing l1_l2_value reduces the regularization effect, which might be necessary if the model is underfitting and not complex enough to capture the underlying patterns.

Optimizer
The choice of optimizer itself is less about increasing or decreasing but rather about the suitability for the problem at hand. Adam is generally a good choice due to its adaptiveness, but in some cases, SGD or another optimizer might converge to a better solution.
