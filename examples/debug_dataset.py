
from graffiti_dataset.dataset import DatasetSample
import cv2
import glob
from graffiti_dataset.tools import draw_main_colors, draw_color_cube, draw_hsv_pixels, draw_super_pixels

"""
This makes all possible transformations on dataset
"""

for sample in glob.glob('./dataset/graffiti_sample/*.p'):

    sample = DatasetSample(sample)

    print(sample)

    cv2.imwrite(f'./readme_images/{sample.sample_id}_image.png', sample.image)
    cv2.imwrite(f'./readme_images/{sample.sample_id}_graffiti_mask.png', sample.graffiti_mask)
    cv2.imwrite(f'./readme_images/{sample.sample_id}_background_graffiti_mask.png', sample.background_graffiti_mask)
    cv2.imwrite(f'./readme_images/{sample.sample_id}_background_mask.png', sample.background_mask)
    cv2.imwrite(f'./readme_images/{sample.sample_id}_incomplete_graffiti_mask.png', sample.incomplete_graffiti_mask)

    sample_main_colors = sample.main_colors()
    draw_main_colors(sample_main_colors, f'./readme_images/{sample.sample_id}_colors_rgb.png')

    super_pixels, rgb_pixels, clustered_image = sample.graffiti_super_pixels()
    draw_color_cube(rgb_pixels, f'./readme_images/{sample.sample_id}_super_pixel_cube_rgb.html')
    cv2.imwrite(f'./readme_images/{sample.sample_id}_clustered_image.png', clustered_image)

    draw_super_pixels(super_pixels, sample.image, f'./readme_images/{sample.sample_id}_super_pixels.png')
    draw_super_pixels(super_pixels, clustered_image, f'./readme_images/{sample.sample_id}_super_pixels_avg.png')

    sample.random_rotate()

    cv2.imwrite(f'./readme_images/{sample.sample_id}_rotated_image.png', sample.image)
    cv2.imwrite(f'./readme_images/{sample.sample_id}_rotated_mask.png', sample.graffiti_mask)

    sample.elastic_transform()

    cv2.imwrite(f'./readme_images/{sample.sample_id}_elastic_image.png', sample.image)
    cv2.imwrite(f'./readme_images/{sample.sample_id}_elastic_mask.png', sample.graffiti_mask)

    sample_pixels = sample.graffiti_pixels()

    sample_pixels = sample.filter_pixel_percentage(sample_pixels, 5)
    hsv_pixels = sample.rgb_pixels_to_hsv(sample_pixels)

    draw_color_cube(sample_pixels, f'./readme_images/{sample.sample_id}_pixel_cube_rgb.html')
    draw_color_cube(hsv_pixels, f'./readme_images/{sample.sample_id}_pixel_cube_hsv.html', colors=sample_pixels)

    draw_hsv_pixels(sample, f'./readme_images/{sample.sample_id}_hue_hsv_pixels.png')



