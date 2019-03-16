

from graffiti_dataset.dataset import DatasetSample
import cv2
import imageio
from graffiti_dataset.tools import random_background
from skimage.transform import rescale
import numpy as np

num_images = 10
source_sample = './dataset/graffiti/7f7fff8f89837f00.p'
# source_sample = './dataset/graffiti/070b070707cfa7ff.p'
gif_duration = 0.5
rescale_ratio = 0.25
images = []

def resize_image(image):
    return rescale(
        image,
        rescale_ratio,
        anti_aliasing=True,
        multichannel=True,
        preserve_range=True,
        mode='reflect').astype(np.uint8)

for _ in range(num_images):

    sample = DatasetSample(source_sample)

    print(sample)

    sample.randomize_color()
    # sample.elastic_transform()

    # images.append(
    #     rescale(
    #         cv2.cvtColor(sample.image,cv2.COLOR_BGR2RGB),
    #         rescale_ratio,
    #         anti_aliasing=True,
    #         multichannel=True, preserve_range=True, mode='reflect').astype(np.uint8)
    #     )

    images.append(resize_image(cv2.cvtColor(sample.image,cv2.COLOR_BGR2RGB)))

imageio.mimsave('./readme_images/random_colors.gif', images, duration=gif_duration)

images = []

for _ in range(num_images):

    sample = DatasetSample(source_sample)

    print(sample)

    sample.randomize_color()
    sample.elastic_transform()

    # images.append(cv2.cvtColor(sample.image,cv2.COLOR_BGR2RGB))
    images.append(resize_image(cv2.cvtColor(sample.image, cv2.COLOR_BGR2RGB)))

imageio.mimsave('./readme_images/random_colors_elastic.gif', images, duration=gif_duration)

images = []

for _ in range(num_images):

    sample = DatasetSample(source_sample)

    print(sample)

    sample.elastic_transform()

    # images.append(cv2.cvtColor(sample.image,cv2.COLOR_BGR2RGB))
    images.append(resize_image(cv2.cvtColor(sample.image, cv2.COLOR_BGR2RGB)))

imageio.mimsave('./readme_images/elastic.gif', images, duration=gif_duration)

images = []

for _ in range(num_images):

    sample = DatasetSample(source_sample)

    print(sample)

    sample.resize(1024, 1024)

    background_image = random_background(1024, 1024, './dataset/background_images_sample/')

    sample.paste_on_background(background_image)

    # images.append(cv2.cvtColor(sample.image,cv2.COLOR_BGR2RGB))
    images.append(resize_image(cv2.cvtColor(sample.image, cv2.COLOR_BGR2RGB)))

imageio.mimsave('./readme_images/random_background.gif', images, duration=gif_duration)


images = []

for _ in range(num_images):

    sample = DatasetSample(source_sample)

    print(sample)

    sample.resize(1024, 1024)
    sample.randomize_color()

    background_image = random_background(1024, 1024, './dataset/background_images_sample/')

    sample.paste_on_background(background_image)

    # images.append(cv2.cvtColor(sample.image,cv2.COLOR_BGR2RGB))
    images.append(resize_image(cv2.cvtColor(sample.image, cv2.COLOR_BGR2RGB)))

imageio.mimsave('./readme_images/random_background_colors.gif', images, duration=gif_duration)

images = []

for _ in range(num_images):

    sample = DatasetSample(source_sample)

    print(sample)

    sample.resize(1024, 1024)
    sample.random_rotate()
    sample.randomize_color()
    sample.elastic_transform()

    background_image = random_background(1024, 1024, './dataset/background_images_sample/')

    sample.paste_on_background(background_image)

    # images.append(cv2.cvtColor(sample.image,cv2.COLOR_BGR2RGB))
    images.append(resize_image(cv2.cvtColor(sample.image, cv2.COLOR_BGR2RGB)))

imageio.mimsave('./readme_images/random_background_colors_all.gif', images, duration=gif_duration)


