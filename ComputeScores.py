from skimage import io, color
import eval_segm as eval
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

Hough_accuracy = []
SLIC_accuracy = []
FCN8_accuracy = []
image_list = []

for img_num in range(1,73):
    image_list.append(img_num)

    filepath = "Truth_Data/" + "test_image" + str(img_num) + ".png"
    image1 = io.imread(filepath)
    isBlack1 = color.rgb2gray(image1)
    isBlack1 = np.rint(isBlack1)

    filepath2 = "Hough_figures/" + "test_image" + str(img_num) + ".png"
    image2 = io.imread(filepath2)
    isBlack2 = color.rgb2gray(image2)
    isBlack2 = np.rint(isBlack2)

    filepath3 = "SLIC_figures/" + "test_image" + str(img_num) + ".png"
    image3 = io.imread(filepath3)
    isBlack3 = color.rgb2gray(image3)
    isBlack3 = np.rint(isBlack3)

    filepath4 = "FCN8_figures/" + "test_image" + str(img_num) + ".png"
    image4 = io.imread(filepath4)
    isBlack4 = color.rgb2gray(image4)
    isBlack4 = np.rint(isBlack4)

    # Compute Pixel Accuracy
    Hough_accuracy.append(eval.mean_accuracy(isBlack1, isBlack2))
    SLIC_accuracy.append(eval.mean_accuracy(isBlack1, isBlack3))
    FCN8_accuracy.append(eval.mean_accuracy(isBlack1, isBlack4))

    print "Finished computing accuracy for image: " + str(img_num)

result = pd.DataFrame({'Image': image_list, 'Hough': Hough_accuracy, 'SLIC': SLIC_accuracy, 'FCN8': FCN8_accuracy})

result.to_csv("/Users/johnkimnguyen/Box Sync/FloodSegmentation/accuracy.csv", sep=",")


# Compute Mean Accuracy of each pixel
# eval.mean_accuracy(isBlack1, isBlack4)
# eval.mean_IU(isBlack1, isBlack4)
# eval.frequency_weighted_IU(isBlack1, isBlack2)