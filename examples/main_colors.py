

from graffiti_dataset.dataset import DatasetSample
from graffiti_dataset.tools import draw_main_colors, draw_color_cube, draw_hsv_pixels

# sample = DatasetSample('./dataset/graffiti/8703f3c389a1f73f.p')
sample = DatasetSample('./dataset/graffiti/01df7f7f7e7e6000.p')


# print(sample)
#
# sample_main_colors = sample.main_colors()
#
# print(sample_main_colors)
#
# draw_main_colors(sample_main_colors, 'colors.png')

#
# sample_pixels = sample.graffiti_pixels(10)
draw_hsv_pixels(sample, 'out.png')

#
# # draw_color_cube(sample_pixels, 'pixels.html')
#
# hsv_pixels = rgb_pixels_to_hsv(sample_pixels)
#
# print(list(hsv_pixels[:,0]))
