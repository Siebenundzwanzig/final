### Simple script to put together a number of jpgs from the kaggle dataset.
### An arbitrary number of symbols, length and number of equations is set.


import Image
import sys
import os
import random
import re
# variables
symbols_in_image = 5    # the no. of symbols every formula has
num_of_images = 100     # the no of formulas created
num_of_categories = 15  # the number of symbols the formulas are put together with

input_path = '/home/laars/uni/WS2017/tensorflow/final/data/extracted_images'
output_path = '/home/laars/uni/WS2017/tensorflow/final/glued_images'

# Method to get single pictures
def get_symbols(num_of_categories):
    symbols = []
    counter = 0
    print os.walk(input_path)
    for root, dirs, files in os.walk(input_path):
        if counter == num_of_categories:
            break
        for file in files:
            path_file = os.path.join(root, file)
            symbols.append((path_file, re.sub("/.*", "", path_file.replace(input_path, "").replace("/", "", 1))))
        counter += 1
    return symbols

# Method to glue symbols together as a "formula"
def glue(symbols, symbols_in_image, num_of_images):
    for j in range(num_of_images):
        symbols_used = []
        for i in range(symbols_in_image):
            symbols_used.append(random.choice(symbols))

        images = map(Image.open, [symbol[0] for symbol in symbols_used])
        name = ""
        for symbol in symbols_used:
            name += symbol[1]
        widths, heights = zip(*(i.size for i in images))

        total_width = sum(widths)
        max_height = max(heights)

        new_im = Image.new('RGB', (total_width, max_height))

        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0]

        new_im.save(output_path + "/" + name + ".jpg")


if __name__ == "__main__":
    symbols = get_symbols(num_of_categories)
    glue(symbols, symbols_in_image, num_of_images)
