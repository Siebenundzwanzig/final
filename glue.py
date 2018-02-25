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
num_of_images = 10     # the no of formulas created
# num_of_categories = 15  # the number of symbols the formulas are put together with

input_path = '/home/laars/uni/WS2017/tensorflow/final/data/extracted_images'
output_folder = '/home/laars/uni/WS2017/tensorflow/final/glued_images'

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



# Method to get single pictures
def get_symbols():
    symbols = {}
    for root, dirs, files in os.walk(input_path):
        actual_symbol = []
        for file in files:
            path_file = os.path.join(root, file)
            actual_symbol.append(path_file)

            symbols[re.sub("/.*", "", path_file.replace(input_path, "").replace("/", "", 1))] = actual_symbol

            # symbols((path_file, re.sub("/.*", "", path_file.replace(input_path, "").replace("/", "", 1))))
    return symbols

# Method to glue symbols together as a "formula"
def glue(symbols, symbols_in_image, num_of_images, output_path):
    # counts = {}
    # for key in symbols:
    #     counts[key] = 0
    all_im = np.array([])
    one_hots = np.array([])
    for j in range(num_of_images):
        symbols_used = []
        for i in range(symbols_in_image):
            # key = random.choice(list(symbols.keys()))
            key = random.choice(list(symbols.keys()))
            # counts[key] = counts[key] + 1
            symbols_used.append((key, random.choice(symbols[key])))

        images = list(map(Image.open, [symbol[1] for symbol in symbols_used]))


        name = "_".join([string[0] for string in symbols_used])
        widths, heights = zip(*(i.size for i in images))

        total_width = sum(widths)
        max_height = max(heights)

        new_im = Image.new('RGB', (total_width, max_height), (255, 255, 255))

        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0]
        all_im = np.append(all_im, new_im)

        label = []
        for string in symbols_used:
            label.append(string[0])
        # print(label)
        label_as_num = np.array([], dtype = int)
        for a in label:
            label_as_num = np.append(label_as_num, dictionary[a])
        # print(label_as_num)
        one_hot = np.zeros((5, 20))
        one_hot[np.arange(5), label_as_num] = 1
        # print(one_hot)

        one_hots = np.append(one_hots, one_hot)

    # new_im.save(output_path + "/" + name + ".jpg")
    print(all_im.shape, one_hots.shape)
    np.save(output_path + "_data", all_im)
    np.save(output_path + "_labels", one_hots)
    # print(counts)
    print("Done.")


if __name__ == "__main__":

    symbols = get_symbols()

    glue(symbols, symbols_in_image, int(num_of_images/10) * 7, output_folder + "/training")
    glue(symbols, symbols_in_image, int(num_of_images/10) * 2, output_folder + "/validation")
    glue(symbols, symbols_in_image, int(num_of_images/10), output_folder + "/testing")
