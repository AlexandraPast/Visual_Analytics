# base tools
import os, sys
sys.path.append(os.path.join(".."))
import argparse

# data analysis
import numpy as np
from numpy.linalg import norm
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

# tensorflow
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

#csv
import pandas as pd

# matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#--------------------------------------------------------------------------------------#

#FUNCTIONS

def extract_features(img_path, model):
    """
    Extract features from image data using pretrained model (e.g. VGG16)
    """
    # Define input image shape - remember we need to reshape
    input_shape = (224, 224, 3)
    # load image from file path
    img = load_img(img_path, target_size=(input_shape[0], 
                                          input_shape[1]))
    # convert to array
    img_array = img_to_array(img)
    # expand to fit dimensions
    expanded_img_array = np.expand_dims(img_array, axis=0)
    # preprocess image - see last week's notebook
    preprocessed_img = preprocess_input(expanded_img_array)
    # use the predict function to create feature representation
    features = model.predict(preprocessed_img)
    # flatten
    flattened_features = features.flatten()
    # normalise features
    normalized_features = flattened_features / norm(features)
    return flattened_features

#--------------------------------------------------------------------------------------#

def main():
    #Firstly the user specified arguments
    #Create the parser
    my_parser = argparse.ArgumentParser(description = 'Image search tool using Nearest neighbours')
    
    #Add the arguments
    #Hyperparameters to specify
    my_parser.add_argument('Image',
                           metavar = 'Target image',
                           type = str,
                           help = 'input name of the target image - only filename, not the whole path')
    my_parser.add_argument('Directory',
                           metavar = 'Directory of images to search in',
                           type = str,
                           help = 'input the directory where all the images are stored')
    
        #Execute parse_args()
    args = my_parser.parse_args()
    
    print('-----------------------------------------------------------------------------')
    #print values of arguments for the user to check
    vals = vars(args)
    for k, v in vals.items():
        print(f'{k:<4}: {v}')
     
    print('-----------------------------------------------------------------------------')

    #directory plus target image
    root_dir = args.Directory
    image_to_use = args.Image
    
    #load VGG16
    model = ResNet50(weights='imagenet', 
                  include_top=False,
                  pooling='avg',
                  input_shape=(224, 224, 3))
    
    #preparing empty lists
    paths = []
    filenames = []
    
    print("---Finding all files---")
    # iterate over files in that directory
    for filename in tqdm(os.listdir(root_dir)):
        f = os.path.join(root_dir, filename)    
        # checking if it is a file
        if os.path.isfile(f):
            # check if it's a picture
            if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
                #save paths in list
                paths.append(f)
                #saves filenames
                filenames.append(filename)
    
    paths = sorted(paths)
    filenames = sorted(filenames)

    print("---Extracting features of all files---")      
    #extracting features from all the images
    feature_list = []
    for i in tqdm(range(len(paths))):
        feature_list.append(extract_features(paths[i], model))

    #analyse nearest neighbours
    neighbors = NearestNeighbors(n_neighbors=10, 
                                 algorithm='brute',
                                 metric='cosine').fit(feature_list)

    print("---Finding Nearest neighbours---")
    #find index of our target image
    index = paths.index(os.path.join(root_dir, image_to_use))

    #calculate nearest neighbours for target
    distances, indices = neighbors.kneighbors([feature_list[index]])

    #empty lists for indices and distances
    idxs = []
    dist = []
    
    print("---5 Best distances and indices---")
    #save indices and distances into lists
    for i in range(1,6):
        print("distance:" + str(distances[0][i]), "indices:" + str(indices[0][i]))
        idxs.append(indices[0][i])
        dist.append(distances[0][i])

    #find the best images and save their paths, names, distances
    best_imgs = []
    best_paths = []
    best_dist = []

    for i in [0,1,2]:
        best_imgs.append(filenames[idxs[i]])
        best_paths.append(paths[idxs[i]])
        best_dist.append(dist[i])
    
    print("---Making and saving your plot and csv---")
    # plot 3 most similar
    width = 10
    fig, axarr = plt.subplots(1,4, figsize =(width,width*1/4))
    
    fig.suptitle("Nearest neighbours to " + image_to_use, fontsize=12)
    
    axarr[0].imshow(mpimg.imread(paths[index]))
    axarr[0].set_title("Chosen image")
    
    axarr[1].imshow(mpimg.imread(best_paths[0]))
    axarr[1].set_title(str(best_dist[0]))
    
    axarr[2].imshow(mpimg.imread(best_paths[1]))
    axarr[2].set_title(str(best_dist[1]))
    
    axarr[3].imshow(mpimg.imread(best_paths[2]))
    axarr[3].set_title(str(best_dist[2]))
    
    plt.savefig("out/Nearest_neighbours_of_" + image_to_use)
    
    #save all output as a csv
    dictionary = {'chosen image': [image_to_use], 'image3': [best_imgs[2]], 'image2': [best_imgs[1]], 'image1': [best_imgs[0]]}  
    dataframe = pd.DataFrame(dictionary) 
    dataframe.to_csv('out/Nearest_neighbours_of_'+image_to_use+'.csv', index=False)
    
    print("---All done!---")
    
#--------------------------------------------------------------------------------------#

if __name__ == '__main__':
   main()