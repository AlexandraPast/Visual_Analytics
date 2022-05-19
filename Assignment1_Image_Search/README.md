# Visual Analytics
Consists of 4 projects - 3 class assignments + 1 self-assigned.

**Link to my Github repository:** https://github.com/AlexandraPast/Visual_Analytics



# Assignment 1 - Image search

The files here are part of the repository for my course in Visual analytics.

## Contribution:
I worked on this code alone. 

## Assignment description
This assignment aimed to create colour histograms of images and use them to find the images closest to a target image specified by the user. Using this method we can select an image and compare it to other images effectively finding the pictures, which most resemble the colour palette of the target picture.

#### The specific tasks were:
- write a small Python program to compare image histograms quantitively using Open-CV and the other image processing tools you've already encountered.
- Your script should do the following:
  - Take a user-defined image from the folder
  - Calculate the "distance" between the colour histogram of that image and all of the others.
  - Find which 3 images are most "similar" to the target image.
  - Save an image which shows the target image, the three most similar, and the calculated distance score.
  - Save a CSV which has one column for the filename and three columns showing the filenames of the closest images in descending order

## Methods
For this assignment, I chose to use two approaches to finding the closest images to a target image. 
- Firstly, in the script `img_search_hist.py` I have iterated through the whole dataset extracting the colour information to plot histograms for each picture
using Open-CV library functions. After creating the histogram I compared it to the histogram of the target image and calculated a distance score. I saved all the distance scores and extracted the three best, matching them with the images they belonged to. As a result, I created an image comprised of the three closest images and the target image with their distance scores written on the bottom of each image. I have also created a .csv file containing the target image and closest images in descending order (descending by distance scores). 
- Secondly, in the script `img_search_nn.py` I have used another method we have learned later on in the course. Using the VGG16 model I iterated through the images and extracted their features so that they could be analysed using the Nearest neighbours method from the sklearn library. Again, I extracted the 3 closest images and created a plot and a .csv file. 

## Scripts

#### This repository consists of:
- **in**: a folder containing the input for the code or data. The data I used were flower images from the course repository CDS-VIS, however, the scripts will run on any data of format .jpg, .jpeg or .png
- **out**: a folder containing the output of the script. The output is a plot and a .csv file.
  - for colour histogram analysis: 
           plot `merged_closest_to_(target_image).jpg` and csv file `closest_images_of_(target_image).csv`
  - for nearest neighbours analysis:
           plot `Nearest_neighbours_of_(target_image).jpg` and csv file `Nearest_neighbours_of_(target_image).csv`
  - If both are run all results will be stored here.
- **src**:
  - `img_search_hist.py`: script for colour histogram analysis
  - `img_search_nn.py`: script for nearest neighbours analysis
  - `utils`: contains modules used in scripts
- `setup.sh`: installs the necessary libraries to run the scripts

#### The scripts do the following:

**`img_search_hist.py`:**
- Take a user-defined image from the folder
- Calculate the "distance" between the colour histogram of that image and all of the others.
- Find which 3 images are most "similar" to the target image.
- Save an image which shows the target image, the three most similar, and the calculated distance score.
- Save a CSV which has one column for the filename and three columns showing the filenames of the closest images in descending order

#### Required positional arguments:
- **Image:** example: `image_0002.jpg`
- **Directory:** example: `../../CDS-VIS/flowers`

**`img_search_nn.py`:**
- Take a user-defined image from the folder
- Load the VGG16 model and extract features of all images in the specified folder
- Find nearest neighbours, calculate distance and save indices
- Find which 3 images are most "similar" to the target image.
- Save an image which shows the target image, the three most similar, and the calculated distance score.
- Save a CSV which has one column for the filename and three columns showing the filenames of the closest images in descending order

#### Required positional arguments:
- **Image:** example: `image_0002.jpg`
- **Directory:** example: `../../CDS-VIS/flowers`

## HOW to correctly run the scripts ##
1. Set your working directory to be the folder containing all the subfolders (in, out, src) using the `cd "path"` command.
2. Open the console and type: `bash setup.sh` (this should install the necessary packages, it might be necessary to provide a path)
3. Type `python` and follow it with the path to `img_search_hist.py` which should be `src/img_search_hist.py` if you set your working directory correctly. Next, specify the target image and directory.
5. Example: 
   - `cd user/file/file/Assignment1_Image_Search` 
   - `bash setup.sh`
   - `python src/img_search_hist.py image_0002.jpg ../../CDS-VIS/flowers`
   - `python src/img_search_hist.py -h` for help explanations (`-h` or `--help`)

## Discussion of results
Since I used two different methods of analysis, I have also ended up with different results. In the case of colour histogram distance scores, I found three images which resemble the target image most in colour, however, this method doesn't account for other factors such as the shape of the object in the picture. Using the nearest neighbours method seems more effective to me if our goal is to find the images that are closest to the target image, as the VGG16 model can account for the objects or shapes in the picture as well, apart from only colour. If we look at my results in the `out` folder and compare, we can see that the nearest neighbours found images with the same type of flower as in the image 0002.jpg which I used as my testing image. The colour histogram method failed to do so, but it was still able to find the one image that was "almost" a copy of the target image.
