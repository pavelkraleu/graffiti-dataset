from collections import defaultdict
from operator import itemgetter
import cv2
import pickle
import random
import skimage
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from sklearn.cluster import KMeans, AgglomerativeClustering


class DatasetSample:
    """This represents single entry from dataset"""

    def __init__(self, pickle_file_path):
        """
        Process single row from Pandas

        :param pickle_file_path: path to a single dataset item
        """

        self.sample = pickle.load(open(pickle_file_path, 'rb'))

    @property
    def image(self):
        return self.sample['image']

    @property
    def sample_id(self):
        return self.sample['id']

    @property
    def background_mask(self):
        return self.sample['background_mask']

    @property
    def graffiti_mask(self):
        return self.sample['graffiti_mask']

    @property
    def incomplete_graffiti_mask(self):
        return self.sample['incomplete_graffiti_mask']

    @property
    def background_graffiti_mask(self):
        return self.sample['background_graffiti_mask']

    @property
    def gps_longitude(self):
        return self.sample['gps_longitude']

    @property
    def gps_latitude(self):
        return self.sample['gps_latitude']

    def random_rotate(self, min_angle=-30, max_angle=30):
        """
        Randomly rotate sample

        :param min_angle: Minimum angle in degrees
        :param max_angle: Maximum angle in degrees

        """

        rotate_angle = 360 - random.randint(min_angle, max_angle)

        self.sample['image'] = skimage.transform.rotate(self.image, rotate_angle, resize=False, preserve_range=True).astype(np.uint8)
        self.sample['background_mask'] = skimage.transform.rotate(self.background_mask, rotate_angle, resize=False, preserve_range=True, mode='constant', cval=255).astype(np.uint8)
        self.sample['graffiti_mask'] = skimage.transform.rotate(self.graffiti_mask, rotate_angle, resize=False, preserve_range=True).astype(np.uint8)
        self.sample['incomplete_graffiti_mask'] = skimage.transform.rotate(self.incomplete_graffiti_mask, rotate_angle, resize=False, preserve_range=True).astype(np.uint8)
        self.sample['background_graffiti_mask'] = skimage.transform.rotate(self.background_graffiti_mask, rotate_angle, resize=False, preserve_range=True).astype(np.uint8)

    def elastic_transform(self, alpha=991, sigma=8):
        """
        Apply elastic transformation to the image.

        :param alpha:
        :param sigma:

        """

        random_state = np.random.RandomState(None)

        shape = self.sample['image'].shape

        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dz = np.zeros_like(dx)

        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))

        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

        self.sample['image'] = map_coordinates(self.image, indices, order=1, mode='reflect').reshape(shape)

        background_mask_rgb = cv2.cvtColor(self.background_mask, cv2.COLOR_GRAY2RGB)
        self.sample['background_mask'] = cv2.cvtColor(map_coordinates(background_mask_rgb, indices, order=1, mode='reflect').reshape(background_mask_rgb.shape), cv2.COLOR_RGB2GRAY)

        graffiti_mask_rgb = cv2.cvtColor(self.graffiti_mask, cv2.COLOR_GRAY2RGB)
        self.sample['graffiti_mask'] = cv2.cvtColor(map_coordinates(graffiti_mask_rgb, indices, order=1, mode='reflect').reshape(graffiti_mask_rgb.shape), cv2.COLOR_RGB2GRAY)

        incomplete_graffiti_mask_rgb = cv2.cvtColor(self.incomplete_graffiti_mask, cv2.COLOR_GRAY2RGB)
        self.sample['incomplete_graffiti_mask'] = cv2.cvtColor(map_coordinates(incomplete_graffiti_mask_rgb, indices, order=1, mode='reflect').reshape(incomplete_graffiti_mask_rgb.shape), cv2.COLOR_RGB2GRAY)

        background_graffiti_mask_rgb = cv2.cvtColor(self.background_graffiti_mask, cv2.COLOR_GRAY2RGB)
        self.sample['background_graffiti_mask'] = cv2.cvtColor(map_coordinates(background_graffiti_mask_rgb, indices, order=1, mode='reflect').reshape(background_graffiti_mask_rgb.shape), cv2.COLOR_RGB2GRAY)

    def resize(self, height, width):
        """
        Resize sample to given size. This resize also masks.

        :param height: Height in pixels
        :param width: Width in pixels
        """

        if height < self.image.shape[0] or width < self.image.shape[1]:
            raise ValueError('New image must be larger or same than original image')

        def new_image(shape):
            return np.zeros(shape, np.uint8)

        y_range = height - self.image.shape[0]
        x_range = width - self.image.shape[1]

        y_skip = random.randint(0, y_range)
        x_skip = random.randint(0, x_range)

        img = new_image((height, width, 3))
        img[y_skip:y_skip + self.image.shape[0],x_skip:x_skip + self.image.shape[1]] = self.image
        self.sample['image'] = img

        img = new_image((height, width)) + 255
        img[y_skip:y_skip + self.background_mask.shape[0],x_skip:x_skip + self.background_mask.shape[1]] = self.background_mask
        self.sample['background_mask'] = img

        img = new_image((height, width))
        img[y_skip:y_skip + self.graffiti_mask.shape[0],x_skip:x_skip + self.graffiti_mask.shape[1]] = self.graffiti_mask
        self.sample['graffiti_mask'] = img

        img = new_image((height, width))
        img[y_skip:y_skip + self.incomplete_graffiti_mask.shape[0],x_skip:x_skip + self.incomplete_graffiti_mask.shape[1]] = self.incomplete_graffiti_mask
        self.sample['incomplete_graffiti_mask'] = img

        img = new_image((height, width))
        img[y_skip:y_skip + self.background_graffiti_mask.shape[0],x_skip:x_skip + self.background_graffiti_mask.shape[1]] = self.background_graffiti_mask
        self.sample['background_graffiti_mask'] = img

    def randomize_color(self, min_max_value=30):
        """
        Modify original color by given index. This change is applied to each channel separately.

        :param min_max_value: Maximum value which can be added/subtracted from each channel.
        """

        random_channels = []

        for _ in range(3):

            random_color = random.randint(-min_max_value, min_max_value)

            # print(f'Random color {random_color}')

            random_color_mask = (cv2.add(self.graffiti_mask, self.background_graffiti_mask) / 255) * random_color

            random_channels.append(random_color_mask)

        random_color_mask = np.stack(random_channels, axis=2)

        random_color_image = cv2.add(self.image.astype(np.float64), random_color_mask).astype(np.uint8)

        self.sample['image'] = random_color_image

    def main_colors(self):
        """
        Computes the most dominant colors in the image

        :return: Array of most important colors and percentage of pixels they represent
        """

        rows,cols,_ = self.image.shape

        raw_pixel_values = []

        for i in range(rows):
            for j in range(cols):
                mask_value = self.graffiti_mask[i, j]

                if mask_value != 0:
                    pixel_value = np.flip(self.image[i, j])
                    raw_pixel_values.append(pixel_value)

        embedding = KMeans().fit(raw_pixel_values)

        pixel_counter = defaultdict(int)

        for label in embedding.labels_:
            pixel_counter[label] += 1

        image_colors = []

        image_main_colors = embedding.cluster_centers_.astype(int)

        for key, item in pixel_counter.items():

            color_percentage = int(item / (len(raw_pixel_values) / 100))

            image_colors.append([color_percentage, list(image_main_colors[key])])

        return sorted(image_colors, key=itemgetter(0), reverse=True)

    def paste_on_background(self, background_image):
        """
        Paste sample on specified background

        :param background_image: Background image as numpy array, this must be same size as this data sample
        """

        if self.image.shape != background_image.shape:
            raise ValueError(f'Background image must have same shape as sample')

        foreground = self.image.astype(float)
        background = background_image.astype(float)

        alpha = cv2.inRange(self.graffiti_mask, 0, 0)
        alpha = cv2.bitwise_not(alpha).astype(float) / 255

        alpha = np.stack((alpha,) * 3, axis=-1)

        foreground = cv2.multiply(alpha, foreground)

        background = cv2.multiply(1.0 - alpha, background)

        self.sample['image'] = cv2.add(foreground, background).astype(np.uint8)

    def __str__(self):

        return f'ID {self.sample_id} Image {self.image.shape} GPS {self.gps_longitude}, {self.gps_latitude}'
