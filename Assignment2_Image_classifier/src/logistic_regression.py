# path tools
import sys,os
sys.path.append(os.path.join(".."))

# image processing 
import cv2 

# neural networks with numpy
import numpy as np
from tensorflow.keras.datasets import cifar10
from utils.neuralnetwork import NeuralNetwork

# machine learning tools
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

# handle warnings
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

#---------------------------------------------------------------------------#

def minmax(data):
    X_norm = (data - data.min())/(data.max() - data.min())
    return X_norm

#---------------------------------------------------------------------------#

def main():

    #Load data
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    labels = ["airplane", 
              "automobile", 
              "bird", 
              "cat", 
              "deer", 
              "dog", 
              "frog", 
              "horse", 
              "ship", 
              "truck"]

    # Convert all the data to greyscale
    X_train_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_train])
    X_test_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_test])

    # Normalize the values
    X_train_scaled = minmax(X_train_grey)
    X_test_scaled = minmax(X_test_grey)

    # Reshaping the data
    nsamples, nx, ny = X_train_scaled.shape
    X_train_dataset = X_train_scaled.reshape((nsamples,nx*ny))

    nsamples, nx, ny = X_test_scaled.shape
    X_test_dataset = X_test_scaled.reshape((nsamples,nx*ny))

    # Simple logistic regression classifier
    clf = LogisticRegression(penalty='none', 
                             tol=0.1, 
                             solver='saga',
                             multi_class="multinomial").fit(X_train_dataset, y_train)

    y_pred = clf.predict(X_test_dataset)

    report = classification_report(y_test, y_pred, target_names=labels)
    print(report)

    #write classification to txt file
    with open("out/lr_report.txt","w") as file:  # Use file to refer to the file object
        file.write(report)
    
#---------------------------------------------------------------------------#

if __name__ == '__main__':
   main()