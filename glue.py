### Simple script to put together a number of jpgs from the kaggle dataset.
### An arbitrary number of symbols, length and number of equations is set.

from PIL import Image
import sys
import os
import random
import re
import numpy as np
# variables
symbols_in_image = 5    # the no. of symbols every formula has
num_of_images = 10000     # the no of formulas created

input_path = '/home/laars/uni/WS2017/tensorflow/final/data/extracted_images'
output_folder = '/home/laars/uni/WS2017/tensorflow/final/glued_images'


# Our dictionary containing 20 symbols of the Kaggle Handwritten Math Symbols Dataset:
# https://www.kaggle.com/xainano/handwrittenmathsymbols
# Maps symbols to numbers.
dictionary = {
        '0' : 0,
        '1' : 1,
        '2' : 2,
        '3' : 3,
        '4' : 4,
        'pi' : 5,
        'rightarrow' : 6,
        'infty' : 7,
        'X' : 8,
        '+' : 9,
        '-' : 10,
        '=' : 11,
        '!' : 12,
        'A' : 13,
        'sigma' : 14,
        'alpha' : 15,
        'cos' : 16,
        'sqrt' : 17,
        '[' : 18,
        ']' : 19
        }



# Method to get all single picture-files in a list
# and put them at the right place in dictionary.
def get_symbols():
    symbols = {}
    for root, dirs, files in os.walk(input_path):
        actual_symbol = []
        for file in files:
            path_file = os.path.join(root, file)
            actual_symbol.append(path_file)
            symbols[re.sub("/.*", "", path_file.replace(input_path, "").replace("/", "", 1))] = actual_symbol

    return symbols # contains ALL images in given input folder


# Glue random symbols together as a "formula", transform labels into one-hot-matrices
# and save as numpy files.

def glue(symbols, symbols_in_image, num_of_images, output_path):

    all_im = []
    one_hots = []

    for j in range(num_of_images):


        symbols_used = []
        for i in range(symbols_in_image):
            # get a random key and symbol
            # append it to symbols_used as tuple
            key = random.choice(list(symbols.keys()))
            symbols_used.append((key, random.choice(symbols[key])))
        # open files as Image objects
        images = list(map(Image.open, [symbol[1] for symbol in symbols_used]))

        ### Data ###
        # gather data for glued image
        widths, heights = zip(*(i.size for i in images))
        total_width = sum(widths)
        max_height = max(heights)
        # create new Image object
        new_im = Image.new('RGB', (total_width, max_height), (255, 255, 255))

        # paste symbols in symbols_used vertically in the new Image
        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0]
        # convert to numpy array
        all_im.append(np.asarray(new_im))

        ### Labels ###

        label_as_num = []
        # Get labels in symbols_used and get correct key from dictionary
        for string in symbols_used:
            label_as_num.append(dictionary[string[0]])

        # convert label-list to one-hot-matrix
        one_hot = np.zeros((5, 20))
        one_hot[np.arange(5), label_as_num] = 1
        one_hots.append(one_hot)

    # save as numpy files .npy
    # labels and data, respectively
    np.save(output_path + "_data", all_im)
    np.save(output_path + "_labels", one_hots)

    print("Done.")


if __name__ == "__main__":

    symbols = get_symbols()

    glue(symbols, symbols_in_image, int(num_of_images/10) * 7, output_folder + "/training")
    glue(symbols, symbols_in_image, int(num_of_images/10) * 2, output_folder + "/validation")
    glue(symbols, symbols_in_image, int(num_of_images/10), output_folder + "/testing")
