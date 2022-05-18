# Assignment 3 - Transfer learning + CNN classification

The files in here are part of repository for my course in Visual analytics.

## Contribution:
I worked on this code alone. 

## Assignment description
The aim of this assignment was to take the CIFAR10 data set and use pre-trained CNN VGG16 for feature extraction and classification. By using a CNN we dont have to turn the data to grayscale or flatten them, which means we can keep a lot of potentially important information about the data. By using a CNN like VGG16 we can also use the fact it has already been pretrained on large quantities of data and so they are more efficient than a model starting from 0. 

#### The specific tasks were:
Your .py script should minimally do the following:
- Load the CIFAR10 dataset
- Use VGG16 to perform feature extraction
- Train a classifier
- Save plots of the loss and accuracy
- Save the classification report

## Methods
For this assignment I wrote a script `Transfer_learning_1.2` in which I load the CIFAR10 dataset from a library. I normalised the data and binarized the labels in order to feed it to the model. I proceeded to load the VGG16 model for feature extraction and defined new classification layers for our classification task. I trained the new classifier on training data and proceeded to make predictions and finally save the classification report and loss and accuracy plot. 

## Scripts
#### This repository consists of folders:
- **in**: folder containing the input for the code or data. In this case script loads CIFAR10 data set itself so no files necessary. 
- **out**: folder containing the output of the script. The output is a plot of loss and accuracy `.png` as well as a `.txt`
          file containing the classification report for the model. 
- **src**: contains the script itself
  - `Transfer_learning_1.2`: is the script to run 
- `setup.sh`: installs the necessary libraries to run the script itself

#### The script does the following:
- Prints all passed argument values or default values
- Loads the CIFAR10 dataset
- Uses VGG16 to perform feature extraction
- Trains a classifier
- Saves plots of the loss and accuracy
- Saves the classification report

All, using pre-set hyperparameters and plot and report names.

#### Default values of optional arguments:
**`tf.keras.optimizers.schedules.ExponentialDecay` hyperparameters:** 
- **Lr:** 0.01 (Initial learning rate)
- **Ds:** 10000 (Decay steps)
- **Dr:** 0.09 (Decay rate)
- **Stair:** True (Staircase)

**testing parameters:**
- **Batch:** 128 (Batch size)
- **Epochs:** 10

**results:**
- **Plot:** loss_acc_plot(.png) (Plot name)
- **Report:** class_report(.txt) (Report name)

## HOW to correctly run the script ##
1. Set your working directory to be the folder containing all the subfolders (in, out, src) using cd "path" command.
2. Open the console and type: `bash setup.sh` (This should install the necessary packadges, if not, open the file and proceed
    to run the code manually.)
2. Type `python` and follow with the path to `Transfer_learning_1.2.py` which should be `src/Transfer_learning_1.2.py` if you set your working directory correctly. 
3. Following the path to the `.py` script should be your parameters if you wish to specify any. They have to be specified by the argument name and value. These are optional so it is possible to run the code without speicying any, specifying some or all of them. 
4. Example: 
   - `cd user/file/file/Assignment3_Transfer_learning`
   - `bash setup.sh`
   - `python src/Transfer_learning_1.2.py -Lr 0.001 
                                          -Ds 10000 
                                          -Dr 0.09 
                                          -Stair True 
                                          -Batch 128 
                                          -Epochs 10 
                                          -Plot plot_accuracy_loss 
                                          -Report class_report` - for all values
   - `python src/Transfer_learning_1.2.py -Ds 10000 
                                          -Plot VeryBeautifulPlotName` - to only specify some parameters
   - `python src/Transfer_learning_1.2.py -h` for help explanations (`-h` or `--help`)

## Discussion of results
During the training the new classifier reached around 62% training accuracy with 10 epochs and around 58% accuracy on testing data. In the classification report the weighted average accuracy reached 59%. The training curves suggest that training accuracy continues to climb while testing began to somewhat plateau. It is my belief that that could suggest overfitting of the model on the training data. Perhaps using additional methods such as, data augmentation, regularisation or dropout could add to the models performance. It is also possible that running the model with lower learning rate and more epochs could also benefit the final results as we can still see the loss decreasing in both training and testing. 
