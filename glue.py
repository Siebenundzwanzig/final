### Simple script to put together a number of jpgs from the kaggle dataset.
### An arbitrary number of symbols, length and number of equations is set.


import Image
import sys
import os
import random

# variables
symbols_in_image = 5
num_of_images = 100
num_of_categories = 20
num_of_symbols = 14

input_path = '/home/laars/uni/WS2017/tensorflow/final/data/extracted_images'
output_path = '/home/laars/uni/WS2017/tensorflow/final/glued_images'

# Method to get single pictures
def get_symbols(num_of_symbols):
    symbols = []
    counter = 0
    for root, dirs, files in os.walk(input_path):
        if counter == num_of_symbols:
            break
        for file in files:
            path_file = os.path.join(root, file)
            symbols.append(path_file)
        counter += 1
    return symbols

# Method to glue symbols together as a "formula"
def glue(symbols, symbols_in_image, num_of_images):
    for j in range(num_of_images):
        symbols_used = []
        for i in range(symbols_in_image):
            symbols_used.append(random.choice(symbols))

        images = map(Image.open, symbols_used)

        widths, heights = zip(*(i.size for i in images))

        total_width = sum(widths)
        max_height = max(heights)

        new_im = Image.new('RGB', (total_width, max_height))

        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset,0))
            x_offset += im.size[0]

        new_im.save('%s/glued_image%s.jpg' % (output_path, j))


if __name__ == "__main__":
    symbols = get_symbols(num_of_symbols)
    glue(symbols, symbols_in_image, num_of_images)
