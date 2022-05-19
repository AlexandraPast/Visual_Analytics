# Assignment 2 - Image classifier benchmark scripts

The files here are part of the repository for my course in Visual analytics.

## Contribution:
I worked on this code alone. 

## Assignment description
This assignment aimed to use two classifier pipelines covered in the lecture and rewrite them into the form of .py script which we could run from the console.
The two classifiers used were a simple logistic regression classifier and a neural network classifier using a premade function from utils. These classifiers were to be used on data to predict the classes of images. 

#### The specific tasks were:
1. One script should be called logistic_regression.py and should do the following:
   - Load either the MNIST_784 data or the CIFAR_10 data
   - Train a Logistic Regression model using scikit-learn
   - Print the classification report to the terminal and save the classification report to out/lr_report.txt
2. Another script should be called nn_classifier.py and should do the following:
   - Load either the MNIST_784 data or the CIFAR_10 data
   - Train a Neural Network model using the premade module in neuralnetwork.py
   - Print output to the terminal during training showing epochs and loss
   - Print the classification report to the terminal and save the classification report to out/nn_report.txt

## Methods
For this task, I have used the code from lecture 7 and applied it to the CIFAR10 dataset from the TensorFlow library. In `logistic_regression.py` I loaded and preprocessed the data by turning them to greyscale, normalising using the minmax function and reshaping. I have loaded the logistic regression classifier, trained it on the training dataset and used it to predict classes of my testing data. As a result, I have saved a classification report with precision and accuracy scores. 
In `nn_classifier.py` I followed the same steps of preprocessing and then applied the neural network classifier from the premade module in utils. Similarly, I trained the classifier, made predictions on testing data and saves a classification report of precision and accuracy scores.

## Scripts

#### This repository consists of:
- **in**: a folder containing the input for the code or data. In this case, the data is cifar10 and is loaded from a library.
- **out**: a folder containing the output of the script. The output is a classification report for logistic regression
           model `lr_report.txt` or a classification report for Neural Network model `nn_report.txt`. If both are run
           both results will be stored here.
- **src**:
  - `logistic_regression.py`: script for the logistic regression model
  - `nn_classifier.py`: script for neural network model
  - `utils`: contains modules used in scripts
- `setup.sh`: installs the necessary libraries to run the scripts

#### The scripts do the following:

**`logistic_regression.py`:**
- Load the CIFAR_10 data
- Train a Logistic Regression model using scikit-learn
- Print the classification report to the terminal and save the classification report to `out/lr_report.txt`

**`nn_classifier.py`:**
- Load the CIFAR_10 data
- Train a Neural Network model using the premade module in `neuralnetwork.py` (utils)
- Print output to the terminal during training showing epochs and loss
- Print the classification report to the terminal and save the classification report to `out/nn_report.txt`


## HOW to correctly run the script ##
1. Set your working directory to be the folder containing all the subfolders (in, out, src) using the `cd "path"` command.
3. Open the console and type: `bash setup.sh` (this should install the necessary packages)
4. Type `python` and follow it with the path to `logistic_regression.py` which should be `src/logistic_regression.py` if you set your working directory correctly. The same applies to the `nn_classifier.py`. 
4. Example: 
   - `cd user/file/file/Assignment2_Image_classifier`
   - `bash setup.sh`
   - `python ../src/logistic_regression.py`

## Discussion of results
The simple logistic regression classifier had a weighted average accuracy of 31% on the extracted features. By comparison, the neural network classifier had a weighted accuracy of 42% with the loss continuing to decline and accuracy continuing to rise. Perhaps more data and training time might result in higher accuracy, however, using a more complex CNN with fully connected classification layers would certainly yield better results.  

