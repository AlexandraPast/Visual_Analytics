# Assignment 4 - self-assigned
The files here are part of the repository for my course in Visual analytics.

## Contribution:
I worked on this code alone. 

## Assignment description
The aim of this assignment was to select a task from what we have covered in the lectures and do some form of analysis on the data of my choice. For my assignment, I have decided to use the Impressionist paintings dataset from: https://www.kaggle.com/datasets/delayedkarma/impressionist-classifier-data. I have decided to use a pre-trained CNN VGG16 for feature extraction of the images and do a classification task to see if my model could classify which artist painted different paintings.

## Methods
For this assignment, I wrote a script `PaintersVGG.py` in which I load the Impressionist paintings dataset. I preprocessed the data by resizing the images, extracting the labels, normalising the data, and binarising the labels in order to feed it to the model. I proceeded to load the VGG16 model for feature extraction and defined new classification layers for our classification task. I trained the new classifier on training data and proceeded to make predictions and finally save the classification report and loss and accuracy plot. 

## Scripts
#### This repository consists of folders:
- **in**: a folder containing the input for the code or data.
- **out**: a folder containing the output of the script. The output is a plot of loss and accuracy `.png` as well as a `.txt`
          file containing the classification report for the model. 
- **src**: contains the script itself
  - `PaintersVGG.py`: is the script to run 
- `setup.sh`: installs the necessary libraries to run the script itself

#### The script does the following:
- Prints all passed argument values or default values
- Loads the Impressionist dataset
- Uses VGG16 to perform feature extraction
- Trains a classifier
- Saves plots of the loss and accuracy
- Saves the classification report

All, using pre-set hyperparameters and plot and report names.

#### Default values of optional arguments:
**hyperparameters:** 
- **Lr:** 0.01 (Initial learning rate)
- **Batch:** 128 (Batch size)
- **Epochs:** 100

**results:**
- **Plot:** loss_acc_plot (Plot name)
- **Report:** classifier_report (Report name)

## HOW to correctly run the script ##
1. Set your working directory to be the folder containing all the subfolders (in, out, src) using the cd "path" command.
2. Open the console and type: `bash setup.sh` (This should install the necessary packages, if not, open the file and proceed
    to run the code manually.)
2. Type `python` and follow it with the path to `PaintersVGG.py` which should be `src/PaintersVGG.py` if you set your working directory correctly. 
3. Following the path to the `.py` script should be your parameters if you wish to specify any. They have to be specified by the argument name and value. These are optional so it is possible to run the code without specifying any, specifying some or all of them. 
4. Example: 
   - `cd user/file/file/Assignment4_self-assigned`
   - `bash setup.sh`
   - `python src/PaintersVGG.py -Lr 0.001  
                                -Batch 128 
                                -Epochs 10 
                                -Plot plot_name
                                -Report report_name` - for all values
   - `python src/PaintersVGG.py -Epochs 30
                                -Plot VeryBeautifulPlotName` - to only specify some parameters
   - `python src/PaintersVGG.py -h` for help explanations (`-h` or `--help`)

## Discussion of results
During the training, the first classifier reached around 51% training accuracy with 100 epochs and around 49% accuracy on testing data. In the classification report, the weighted average accuracy reached 50%. The training curves suggest that the accuracy continues to climb and losses continue to go down, however, the testing accuracy seemenigly started to plateau. While testing different hyperparameters and layer setups I have found that a lower learning rate and higher amount of epochs benefit the performance quite well. In further experimenting, I have trained a new model where I added an additional layer with 256 nodes and used data augmentation to generate new images out of the images available. The weighted average accuracy increased to __ and I observed the losses decreasing and accuracy slowly increasing continuously. 
Perhaps using more data in the future and including additional methods such as regularisation could add to the model's performance as well. 


