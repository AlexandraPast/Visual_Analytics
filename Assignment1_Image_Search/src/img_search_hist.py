import os
import argparse
import numpy as np 
import sys
sys.path.append(os.path.join('../../CDS-VIS/flowers'))
import cv2
import IPython.display as display
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

#--------------------------------------------------------------------------------------#

def main():
    #Firstly the user specified arguments
    #Create the parser
    my_parser = argparse.ArgumentParser(description = 'Image search tool using histograms')
    
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

    
    print("---Loading your image---")
    #Input the desired image instead of "image_0002.jpg"
    image_to_use = args.Image
    #assign directory
    directory = args.Directory
    font_directory = "src/Chn_Prop_Arial_Normal.ttf"
    
    image = cv2.imread(os.path.join(directory, image_to_use))
    hist1 = cv2.calcHist([image], [0,1,2], None, [8,8,8], [0,256, 0,256, 0,256])
    hist1 = cv2.normalize(hist1, hist1, 0,255, cv2.NORM_MINMAX)
    
    #preparing empty lists
    paths = []
    distances = []
    
    print("---Making histograms off all the images---")
    # iterate over files in that directory
    # makes histograms for every picture and calculates distances 
    for filename in tqdm(os.listdir(directory)):
        f = os.path.join(directory, filename)    
        # checking if it is a file
        if os.path.isfile(f):
            #we exclude the path of the user defined picture above
            if f != (directory +"/"+ image_to_use):      
                image2 = cv2.imread(os.path.join(f))
                hist2 = cv2.calcHist([image2], [0,1,2], None, [8,8,8], [0,256, 0,256, 0,256])
                hist2 = cv2.normalize(hist2, hist2, 0,255, cv2.NORM_MINMAX)
                distance = round(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR), 2)
                #save paths in list
                paths.append(f)
                #save distances in a list
                distances.append(distance)

#--------------------------------------------------------------------------------------#
    print("---Finding three closest---")
    # We take list of distances and sort in descending order, taking only last 3 values (smallest)
    best_matching = sorted(distances, reverse = True)[-3:]

    #preparing empty lists
    indexes = []
    best_paths = []

    #finding the indexes of the 3 smallest values and then finding the picture paths
    #they belong to based on the index and saving them to a list
    for token in best_matching:
        index = distances.index(token)
        best_image = paths[index]
        best_paths.append(best_image)
        indexes.append(index)

    #prepare variable
    best_images = []
    #reducing the paths to just filename
    for path in best_paths:
        filename = os.path.basename(path)
        best_images.append(filename)

#--------------------------------------------------------------------------------------#
    print("---Making your final image and csv---")
    #add user input image to list
    best_paths.append(directory+ "/" +image_to_use)
    #set max size of a picture
    max_size = (400, 400)
    #set size to start at 0 (used as a coordinate)
    size = 0
    #create empty image to put images in
    new_image = Image.new('RGB',(4*400, 400))
    #prepare empty list to save the coordinate into
    size_lst = []

    #loop over the best pictures
    for path in best_paths:
        img = Image.open(path)
        #resizes image but keeps aspect ratio
        img.thumbnail(max_size)
        new_image.paste(img,(size,0))
        size_lst.append(size)
        size = size + img.size[0]

    #creates object we can write on from the combined picture we created
    obj = ImageDraw.Draw(new_image)
    #in order to change the size I had to use another font which has to be saved somewhere
    #and path specified
    myFont = ImageFont.truetype(font_directory, 28)

    #adds distance onto each picture but the last one which is the user defined picture
    for i in range(3):
        obj.text( (size_lst[i], 350), "distance:" + str(best_matching[i]), fill=(255, 0, 0), font = myFont)

    #saves image
    new_image.save("out/merged_closest_to_"+os.path.basename(path), optimize=True, quality=100)

    #PICTURES 0042 and 0002 are PURE EVIL!! I legit thought I had a bug and spent 15 minutes looking at the code T_T

#--------------------------------------------------------------------------------------#

    #save all output as a csv
    dictionary = {'chosen image': [image_to_use], 'image3': [best_images[2]], 'image2': [best_images[1]], 'image1': [best_images[0]]}  
    dataframe = pd.DataFrame(dictionary) 
    dataframe.to_csv('out/closest_images_of_'+image_to_use+'.csv', index=False)
    
    print("---All done!---")

#--------------------------------------------------------------------------------------#

if __name__ == '__main__':
   main()
