
from graffiti_dataset.dataset import DatasetSample
import cv2
import glob
from graffiti_dataset.tools import draw_main_colors, draw_color_cube, draw_hsv_pixels

for sample in glob.glob('./dataset/graffiti_sample/*.p'):

    sample = DatasetSample(sample)

    print(sample)

    cv2.imwrite(f'./readme_images/{sample.sample_id}_image.png', sample.image)
    cv2.imwrite(f'./readme_images/{sample.sample_id}_graffiti_mask.png', sample.graffiti_mask)
    cv2.imwrite(f'./readme_images/{sample.sample_id}_background_graffiti_mask.png', sample.background_graffiti_mask)
    cv2.imwrite(f'./readme_images/{sample.sample_id}_background_mask.png', sample.background_mask)
    cv2.imwrite(f'./readme_images/{sample.sample_id}_incomplete_graffiti_mask.png', sample.incomplete_graffiti_mask)

    sample_main_colors = sample._kmeans_color_clusters()

    draw_main_colors(sample_main_colors, f'./readme_images/{sample.sample_id}_colors_rgb.png')

    sample_main_colors = sample._hsv_color_clusters_dbscan()

    draw_main_colors(sample_main_colors, f'./readme_images/{sample.sample_id}_colors_hsv.png')

    sample.random_rotate()

    cv2.imwrite(f'./readme_images/{sample.sample_id}_rotated_image.png', sample.image)
    cv2.imwrite(f'./readme_images/{sample.sample_id}_rotated_mask.png', sample.graffiti_mask)

    sample.elastic_transform()

    cv2.imwrite(f'./readme_images/{sample.sample_id}_elastic_image.png', sample.image)
    cv2.imwrite(f'./readme_images/{sample.sample_id}_elastic_mask.png', sample.graffiti_mask)

    sample_pixels = sample.graffiti_pixels(10)

    draw_color_cube(sample_pixels, f'./readme_images/{sample.sample_id}_pixel_cube.html')

    draw_hsv_pixels(sample, f'./readme_images/{sample.sample_id}_hue_hsv_pixels.png')

