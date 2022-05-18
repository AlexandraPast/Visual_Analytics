import os
import sys
import argparse

# tf tools
import tensorflow as tf

# image processsing
from tensorflow.keras.preprocessing.image import (load_img,
                                                  img_to_array,
                                                  ImageDataGenerator)
# VGG16 model
from tensorflow.keras.applications.vgg16 import (preprocess_input,
                                                 decode_predictions,
                                                 VGG16)
# cifar10 data - 32x32
from tensorflow.keras.datasets import cifar10

# layers
from tensorflow.keras.layers import (Flatten, 
                                     Dense, 
                                     Dropout, 
                                     BatchNormalization)
# generic model object
from tensorflow.keras.models import Model

# optimizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD

#scikit-learn
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

# for plotting
import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------------------------------------#
#DEFINITIONS

#Plotting function
def plot_history(H, epochs, plotname):
    plt.style.use("seaborn-colorblind")

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, epochs), H.history["loss"], label='Training loss (' + str(str(format(H.history["loss"][-1],'.5f'))+')'))
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label='Validation loss (' + str(str(format(H.history["val_loss"][-1],'.5f'))+')'), linestyle=":")
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label='Training accuracy (' + str(format(H.history["accuracy"][-1],'.5f'))+')')
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label='Validation accuracy (' + str(format(H.history["val_accuracy"][-1],'.5f'))+')', linestyle=":")
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()
    plt.savefig("out/" + plotname + ".png")
    plt.show()

    

#------------------------------------------------------------------------#

def main():
    #Firstly the user specified arguments
    #Create the parser
    my_parser = argparse.ArgumentParser(description = 'Classifier using transfer learning with a pretrained CNN VGG16')
    
    #Add the arguments
    #Hyperparameters to specify
    my_parser.add_argument('-Lr',
                           metavar = 'initial learning rate of ExponentialDecay',
                           type = np.float64,
                           default = 0.01,
                           help = 'input number specifying initial learning rate (ideally on an interval [10^-6, 1.0]), np.float64')
    my_parser.add_argument('-Ds',
                           metavar = 'decay steps of ExponentialDecay',
                           type = np.int64,
                           default = 10000,
                           help = 'input number specifying decay steps (np.int64; Must be positive.))')
    
    my_parser.add_argument('-Dr',
                           metavar = 'decay rate of ExponentialDecay',
                           type = np.float64,
                           default = 0.09,
                           help = 'input number specifying decay rate (np.float64; The decay rate.)')
    my_parser.add_argument('-Stair',
                           metavar = 'staircase of ExponentialDecay',
                           type = bool,
                           default = True,
                           help = 'input True or False (Boolean. If True decay the learning rate at discrete intervals)')
    my_parser.add_argument('-Batch',
                           metavar = 'batch size',
                           type = np.int64,
                           default = 128,
                           help = 'input integer number; np.int64')
    my_parser.add_argument('-Epochs',
                           metavar = 'epochs',
                           type = np.int64,
                           default = 10,
                           help = 'input integer number; np.int64')
    
    #Plot name and report name to be specified
    my_parser.add_argument('-Plot',
                           metavar = 'plot name',
                           type = str,
                           default = "loss_acc_plot",
                           help = 'input your desired name for the loss and accuracy plot')
    my_parser.add_argument('-Report',
                           metavar = 'classification report name',
                           type = str,
                           default = "class_report",
                           help = 'input your desired name for the classification report')
    
    
    #Execute parse_args()
    args = my_parser.parse_args()
    
    print('***************************************************************************')
    #print values of arguments for the user to check
    vals = vars(args)
    for k, v in vals.items():
        print(f'{k:<4}: {v}')
     
    print('***************************************************************************')
    
    
    #Load CIFAR10 data
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    #normalise data
    X_train_scaled = X_train/255
    X_test_scaled = X_test/255
    
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
    
    #binarize labels
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.fit_transform(y_test)
    
    #Check tensor shape and print for the user
    print('***************************************************************************')
    print('tensor shape is' + str(X_train.shape)) 
    print('***************************************************************************')
    #we are using the data for finetuning, so the shape doesn't have to be 244x244
    
    #clear the layers kept in memory
    tf.keras.backend.clear_session()

    #load without classifier layer
    model = VGG16(include_top = False,
                  pooling = "avg",
                  input_shape = (32,32,3)) #redefine the input shape
    
    #Disable training of Conv layers
    for layer in model.layers:
        layer.trainable = False
        
    print('***************************************************************************')
    
    model.summary()
    
    #Add new classification layers
    flat1 = Flatten()(model.layers[-1].output) # The second part of round brackets marks the input to the layer
    class1 = Dense(128, activation = "relu")(flat1)
    output = Dense(10, activation = "softmax")(class1)

    #define new model
    model = Model(inputs=model.inputs,
                  outputs=output)

    print('***************************************************************************')
    
    #summarise
    model.summary()
    
    #optimizer
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate = args.Lr,
        decay_steps = args.Ds,
        decay_rate = args.Dr,
        staircase = args.Stair)
    
   
    sgd = SGD(learning_rate=lr_schedule)
    
    #Compile
    model.compile(optimizer=sgd,
              loss="categorical_crossentropy",
              metrics=["accuracy"])
    
    #Training
    H = model.fit(X_train, y_train,
                  validation_data = (X_test, y_test),
                  batch_size = args.Batch,
                  epochs = args.Epochs,
                  verbose = 1)
    
    #plot of accuracy and loss (plot is saved in the def above because since plt.show closes the figure, it has to be saved before plt.show is    
    #called)
    plot_history(H, args.Epochs, plotname = args.Plot)
    
    #make predictions
    predictions = model.predict(X_test, batch_size = args.Batch)
    #classification report
    report = classification_report(y_test.argmax(axis = 1),
                                    predictions.argmax(axis = 1),
                                    target_names = labels)
   
    print(report)
    
    # write classification to txt file
    with open("out/" + str(args.Report) + ".txt","w") as file:  # Use file to refer to the file object
        file.write(report)
    
#------------------------------------------------------------------------#

if __name__ == '__main__':
   main()