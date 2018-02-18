import Image
import sys
import os
import random

symbols = []

for root, dirs, files in os.walk('/home/laars/uni/WS2017/tensorflow/final/data/extracted_images'):  # replace the . with your starting directory
   for file in files:
      path_file = os.path.join(root,file)
      symbols.append(path_file)

num_of_symbols = 30

symbols_used = []
for i in range(num_of_symbols):
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

new_im.save('test.jpg')
