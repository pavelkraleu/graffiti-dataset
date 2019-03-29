

from graffiti_dataset.dataset import DatasetSample
from graffiti_dataset.tools import draw_main_colors, draw_color_cube, draw_hsv_pixels, draw_super_pixels

# sample = DatasetSample('./dataset/graffiti/8703f3c389a1f73f.p')
sample = DatasetSample('./dataset/graffiti/01df7f7f7e7e6000.p')
# sample = DatasetSample('./dataset/graffiti/000001ff0013ffff.p')

clusters = sample.cluster_colors(1, 'rgb', 3, True)

print(clusters)

draw_main_colors(clusters, f'clusters_.png')

pxls_segments, mean_pixels, avg_image = sample.graffiti_super_pixels()

print(mean_pixels.shape)

draw_super_pixels(pxls_segments, sample.image, 'pixels.png')
draw_super_pixels(pxls_segments, avg_image, 'pixels_avg.png')
