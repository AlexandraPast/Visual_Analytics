# path tools
import os
import sys
import argparse
# numpy
import numpy as np
# TensorFlow
import tensorflow as tf
from tensorflow.keras.preprocessing.image import (load_img,
                                                  img_to_array,
                                                  ImageDataGenerator)
# generic model object
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam
# optimizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
# layers
from tensorflow.keras.layers import (Flatten, 
                                     Dense, 
                                     Dropout, 
                                     BatchNormalization)
from keras.callbacks import ModelCheckpoint, EarlyStopping

# scikit-learn
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
# VGG16 model
from tensorflow.keras.applications.vgg16 import (preprocess_input,
                                                 decode_predictions,
                                                 VGG16)
import matplotlib.pyplot as plt #plots

# handle warnings
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

# fix random seed for reproducibility
seed = 42
np.random.seed(seed)

#--------------------------------------------------------------------------------------#

# FUNCTIONS
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
    plt.savefig("out/"+ plotname +".png")
    plt.show()
    
def get_data(directory, labels, all_images, all_labels):
    #loop through all the folders in data folder 
    for label in labels:
        path = os.path.join(directory, label)
        #assign numeric index to the label
        class_num = labels.index(label)
        #loop through all images within the current folder
        for img in os.listdir(path):
            try:
                #load image and change size to 224x224
                image = load_img(os.path.join(path, img), target_size=(224, 224))
                image = img_to_array(image)
                #reshape data for the model
                image = image.reshape((image.shape[0], image.shape[1], image.shape[2]))
                image = preprocess_input(image)
                #append the image array to list of images
                all_images.append(image)
                #append the numeric index of label to list of labels
                all_labels.append(class_num)
            except Exception as e:
                print(e)
    print("------I am done------")

#--------------------------------------------------------------------------------------#

def main():
    #Firstly the user specified arguments
    #Create the parser
    my_parser = argparse.ArgumentParser(description = 'Impressionist paintings Classifier VGG16')
    
    #Add the arguments
    #Hyperparameters to specify
    my_parser.add_argument('-Lr',
                           metavar = '-initial learning rate of ExponentialDecay',
                           type = np.float64,
                           default = 0.01,
                           help = 'input number specifying initial learning rate (ideally on an interval (10^-6; 1.0)), np.float64')
    my_parser.add_argument('-Batch',
                           metavar = '-batch size',
                           type = np.int64,
                           default = 150,
                           help = 'input integer number; np.int64')
    my_parser.add_argument('-Epochs',
                           metavar = '-epochs',
                           type = np.int64,
                           default = 150,
                           help = 'input integer number; np.int64')
    #Plot name and report name to be specified
    my_parser.add_argument('-Plot',
                           metavar = '-plot name',
                           type = str,
                           default = "loss_acc_plot",
                           help = 'input your desired name for the loss and accuracy plot')
    my_parser.add_argument('-Report',
                           metavar = '-classification report name',
                           type = str,
                           default = "classifier_report",
                           help = 'input your desired name for the classification report')
    
    #Execute parse_args()
    args = my_parser.parse_args()
    
    print('--------------------------------------------------------------------------')
    #print values of arguments for the user to check
    vals = vars(args)
    for k, v in vals.items():
        print(f'{k:<4}: {v}')
     
    print('--------------------------------------------------------------------------')
        
    #directories
    train_data_dir = 'in/training/training'
    valid_data_dir = 'in/validation/validation'
    #all unique labels
    labels = ['Cezanne', 'Degas', 'Gauguin', 'Hassam', 'Matisse', 'Monet', 'Pissarro','Renoir', 'Sargent', 'VanGogh' ]
    #for model softmax layer
    categories = len(labels)
    
    #hyperparameters
    learning_rate = args.Lr
    batch_size = args.Batch
    epochs = args.Epochs
    #output_names
    plot_name = args.Plot
    report_name = args.Report
    
    #create empty lists for use in get_data func
    all_images = []
    all_labels = []

#--------------------------------------------------------------------------------------#
    print("------Fetching your training data------")
    #run the function to get data from training folder
    get_data(train_data_dir, labels = labels, all_images = all_images, all_labels = all_labels)
    print("------Fetching your testing data------")
    #run the func to get data from validation folder
    get_data(valid_data_dir, labels = labels, all_images = all_images, all_labels = all_labels)

    print("------Preparing your data------")
    X = np.array(all_images)
    y = np.array(all_labels)
    
    #split data randomly into training and testing 
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.2, 
                                                    random_state = 30,
                                                    shuffle=True)
    
    #normalise
    X_train = (X_train.astype("float"))/255
    X_test = (X_test.astype("float"))/255

    # create one-hot encodings
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.fit_transform(y_test)

    print("------Loading the model------")
    #load model without classifier layers
    model = VGG16(include_top = False, 
                  pooling = 'avg',
                  input_shape = (224, 224, 3))
    
    #mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False
    print(model.summary())
    
    #add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    bn = BatchNormalization()(flat1)
    class1 = Dense(256, 
                   activation = 'relu')(bn)
    class2 = Dense(128, 
                   activation = 'relu')(class1)
    output = Dense(10, 
                   activation = 'softmax')(class2)

    #define new model
    model = Model(inputs = model.inputs, 
                  outputs = output)
    
    #define optimizer
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=10000,
        decay_rate=0.9)
    sgd = SGD(learning_rate=lr_schedule)
    
    # compile
    model.compile(optimizer = sgd,
                  loss = 'categorical_crossentropy',
                  metrics = ['accuracy'])
    # summarize
    model.summary()


    #generate new data
    datagen = ImageDataGenerator(horizontal_flip = True, 
                                 rotation_range = 20,
                                 vertical_flip = False,
                                 zoom_range = 0.1,
                                 width_shift_range = 0.1,  
                                 height_shift_range = 0.1,
                                 )

    #early stopping and model checkpoints
    steps_per_epoch = y_train.size / batch_size
    save_period = 10

    checkpoint = ModelCheckpoint("vgg16_1.h5",
                                 monitor="val_accuracy",
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=False,
                                 mode='auto',
                                 save_freq=int(save_period * steps_per_epoch))
    early = EarlyStopping(monitor='val_accuracy',
                          min_delta=0,
                          patience=20,
                          verbose=1,
                          mode='auto')
   
    print("------Training the model------")
    print("This might take a while")
    
    #train the model
    # compute quantities required for featurewise normalization
    datagen.fit(X_train)
    # fits the model on batches with real-time data augmentation:
    H = model.fit(datagen.flow(X_train, y_train, batch_size = batch_size),
                  validation_data = (X_test, y_test),
                  epochs = epochs,
                  verbose = 1,
                  callbacks = [checkpoint,early])
    
    #visualise with a plot and save plot
    plot_history(H, epochs, plot_name)
    
    print("------Model predictions and accuracy------")
    #create predictions and print report
    predictions = model.predict(X_test, batch_size = batch_size)
    report = classification_report(y_test.argmax(axis = 1),
                                   predictions.argmax(axis = 1),
                                   target_names = labels)
    
    print(report)
    
    # write classification report to txt file
    with open("out/"+ report_name +".txt","w") as file:  # Use file to refer to the file object
        file.write(report)
        
    print("------All done!------")
    
#--------------------------------------------------------------------------------------#

if __name__ == '__main__':
   main()
