import colorsys
from collections import defaultdict
from operator import itemgetter
import cv2
import pickle
import random
import skimage
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage import color
from skimage.segmentation import quickshift
from sklearn.cluster import KMeans, DBSCAN


# from sklearn.cluster import AgglomerativeClustering, AffinityPropagation, MeanShift, SpectralClustering, Birch
# from sklearn.mixture import GaussianMixture


class DatasetSample:
    """This represents single entry from dataset"""

    def __init__(self, pickle_file_path, apply_opening_on_masks=True):
        """
        Process single row from Pandas

        :param pickle_file_path: path to a single dataset item
        """

        self.sample = pickle.load(open(pickle_file_path, 'rb'))
        # self.sample['image'] = cv2.cvtColor(self.sample['image'], cv2.COLOR_RGB2BGR)

        if apply_opening_on_masks:
            for layer in ['graffiti_mask',
                          'background_mask',
                          'incomplete_graffiti_mask',
                          'background_graffiti_mask']:
                self.sample[layer] = self._apply_opening(self.sample[layer])

    @staticmethod
    def _apply_opening(img, kernel=np.ones((7, 7), np.uint8)):
        """
        Remove artefacts from masks
        """

        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

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

    def random_downsize(self, min_ratio=0.5, max_ratio=1.0):
        """
        Randomly downsize sample

        :param min_ratio: Minimum resize ratio
        :param max_ratio: Maximum resize ratio

        """

        random_ratio = random.uniform(max_ratio, min_ratio)

        for layer in [
            'image',
            'background_mask',
            'graffiti_mask',
            'incomplete_graffiti_mask',
            'background_graffiti_mask']:

            self.sample[layer] = skimage.transform.rescale(self.sample[layer], random_ratio, multichannel=self.sample[layer].ndim == 3, preserve_range=True).astype(np.uint8)


    def random_rotate(self, min_angle=-30, max_angle=30):
        """
        Randomly rotate sample

        :param min_angle: Minimum angle in degrees
        :param max_angle: Maximum angle in degrees

        """

        rotate_angle = 360 - random.randint(min_angle, max_angle)

        for layer in [
            'image',
            'background_mask',
            'graffiti_mask',
            'incomplete_graffiti_mask',
            'background_graffiti_mask']:

            self.sample[layer] = skimage.transform.rotate(self.sample[layer], rotate_angle, resize=False, preserve_range=True).astype(np.uint8)

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

        # This is mildly horrible way how to do it
        # I convert 1-channel masks to 3-channel in order to have them same shape is RGB image
        # Than back to 1-channel

        background_mask_rgb = cv2.cvtColor(self.background_mask, cv2.COLOR_GRAY2RGB)
        self.sample['background_mask'] = cv2.cvtColor(
            map_coordinates(background_mask_rgb, indices, order=1, mode='reflect').reshape(background_mask_rgb.shape),
            cv2.COLOR_RGB2GRAY)

        graffiti_mask_rgb = cv2.cvtColor(self.graffiti_mask, cv2.COLOR_GRAY2RGB)
        self.sample['graffiti_mask'] = cv2.cvtColor(
            map_coordinates(graffiti_mask_rgb, indices, order=1, mode='reflect').reshape(graffiti_mask_rgb.shape),
            cv2.COLOR_RGB2GRAY)

        incomplete_graffiti_mask_rgb = cv2.cvtColor(self.incomplete_graffiti_mask, cv2.COLOR_GRAY2RGB)
        self.sample['incomplete_graffiti_mask'] = cv2.cvtColor(
            map_coordinates(incomplete_graffiti_mask_rgb, indices, order=1, mode='reflect').reshape(
                incomplete_graffiti_mask_rgb.shape), cv2.COLOR_RGB2GRAY)

        background_graffiti_mask_rgb = cv2.cvtColor(self.background_graffiti_mask, cv2.COLOR_GRAY2RGB)
        self.sample['background_graffiti_mask'] = cv2.cvtColor(
            map_coordinates(background_graffiti_mask_rgb, indices, order=1, mode='reflect').reshape(
                background_graffiti_mask_rgb.shape), cv2.COLOR_RGB2GRAY)

    def resize(self, height, width, bg_image=None):
        """
        Resize sample to given size. This resize also masks.

        :param height: Height in pixels
        :param width: Width in pixels
        """

        if height < self.image.shape[0] or width < self.image.shape[1]:
            raise ValueError('New image must be larger or same than original image')

        if bg_image is not None:
            assert bg_image.shape == (height, width, 3)

        def new_image(shape):
            return np.zeros(shape, np.uint8)

        y_range = height - self.image.shape[0]
        x_range = width - self.image.shape[1]

        y_skip = random.randint(0, y_range)
        x_skip = random.randint(0, x_range)

        if bg_image is not None:
            img = bg_image.astype('float32')
        else:
            img = new_image((height, width, 3))
        img[y_skip:y_skip + self.image.shape[0], x_skip:x_skip + self.image.shape[1]] = self.image
        self.sample['image'] = img

        img = new_image((height, width)) + 255
        img[y_skip:y_skip + self.background_mask.shape[0],
        x_skip:x_skip + self.background_mask.shape[1]] = self.background_mask
        self.sample['background_mask'] = img

        img = new_image((height, width))
        img[y_skip:y_skip + self.graffiti_mask.shape[0],
        x_skip:x_skip + self.graffiti_mask.shape[1]] = self.graffiti_mask
        self.sample['graffiti_mask'] = img

        img = new_image((height, width))
        img[y_skip:y_skip + self.incomplete_graffiti_mask.shape[0],
        x_skip:x_skip + self.incomplete_graffiti_mask.shape[1]] = self.incomplete_graffiti_mask
        self.sample['incomplete_graffiti_mask'] = img

        img = new_image((height, width))
        img[y_skip:y_skip + self.background_graffiti_mask.shape[0],
        x_skip:x_skip + self.background_graffiti_mask.shape[1]] = self.background_graffiti_mask
        self.sample['background_graffiti_mask'] = img

    def randomize_color(self, min_max_value=30):
        """
        Modify original color by given index. This change is applied to each channel separately.

        :param min_max_value: Maximum value which can be added/subtracted from each channel.
        """

        random_channels = []

        for _ in range(3):
            random_color = random.randint(-min_max_value, min_max_value)
            random_color_mask = (cv2.add(self.graffiti_mask, self.background_graffiti_mask) / 255) * random_color
            random_channels.append(random_color_mask)

        random_color_mask = np.stack(random_channels, axis=2)
        random_color_image = cv2.add(self.image.astype(np.float64), random_color_mask).astype(np.uint8)

        self.sample['image'] = random_color_image

    def graffiti_super_pixels(self):
        """
        Computes graffiti super pixels with quickshift method

        """

        pixel_values = []

        segments_quick = quickshift(self.image / 255, max_dist=35, kernel_size=5)
        segmented_image = color.label2rgb(segments_quick, self.image, kind='avg')
        uniq_super_pixel_ids = np.unique(segments_quick)

        for sp_id in uniq_super_pixel_ids:

            sp_px_coordinates = np.array(np.where(segments_quick == sp_id))
            sp_px_coordinates = sp_px_coordinates.swapaxes(0, 1)

            graffiti_mask_values = np.mean(self.graffiti_mask[sp_px_coordinates[:, 0], sp_px_coordinates[:, 1]]) / 255

            cluster_pixels = np.array(np.where(segments_quick == sp_id))[:, 0]

            # Use only super pixels which are at least 90% labeled as graffiti
            if graffiti_mask_values >= 0.9:
                # if mask_value != 0:
                pixel_values.append(np.flip(segmented_image[cluster_pixels[0], cluster_pixels[1]]))

        return segments_quick, np.array(pixel_values), segmented_image

    def graffiti_pixels(self, add_xy=False):
        """
        Returns all graffiti pixels as RGB values

        :param add_xy: add X and Y coordinates to resulting array

        """

        rows, cols, _ = self.image.shape

        raw_pixel_values = []

        for i in range(rows):
            for j in range(cols):
                mask_value = self.graffiti_mask[i, j]

                if mask_value != 0:
                    pixel_value = np.flip(self.image[i, j])

                    if add_xy:
                        raw_pixel_values.append(np.concatenate((pixel_value, np.array([i, j]))))
                    else:
                        raw_pixel_values.append(pixel_value)

        return np.array(raw_pixel_values)

    @staticmethod
    def filter_pixel_percentage(raw_pixel_values, return_percentage):
        """
        Returns only specific percentage of pixels.
        This is mainly helpful to CPU intensive tasks

        :param raw_pixel_values: array of RGB pixels
        :param return_percentage: percentage of pixels to return
        """

        assert 100 >= return_percentage >= 0

        random.shuffle(raw_pixel_values)

        stop_val = int((len(raw_pixel_values) / 100) * return_percentage)

        raw_pixel_values = raw_pixel_values[0:stop_val]

        return np.array(raw_pixel_values)

    @staticmethod
    def rgb_pixels_to_hsv(rgb_pixels):
        """
        Convert RGB pixels to HSV color space

        :param rgb_pixels: array of RGB pixels
        """

        return np.array(
            [np.array(colorsys.rgb_to_hsv(*pixel / 255)) * np.array([360, 100, 100]) for pixel in rgb_pixels])

    def main_colors(self):
        """
        Computes the most dominant colors in the image

        :return: Array of most important colors and percentage of pixels they represent
        """

        return self.cluster_colors(1, 'rgb', 10, True)

    def cluster_colors(self, clustering_method, color_type, dataset_percent, use_super_pixels):
        """
        Cluster graffiti

        :param clustering_method: 0 - Kmeans, 1 - DBSCAN
        :param color_type: {'rgb', 'hsv', 'hsv_hue'}
        :param dataset_percent: Limit clustering to only some percentage of pixels.
        This is used only with use_super_pixels = False
        :param use_super_pixels: True to use super pixels rather than raw pixels
        """

        assert clustering_method in range(4)
        assert color_type in ['rgb', 'hsv', 'hsv_hue']

        if use_super_pixels:
            super_pixels, rgb_pixels, _ = self.graffiti_super_pixels()
        else:
            rgb_pixels = self.graffiti_pixels()

        if color_type == 'rgb':
            input_data = rgb_pixels
        elif color_type == 'hsv':
            input_data = self.rgb_pixels_to_hsv(rgb_pixels)
        elif color_type == 'hsv_hue':
            input_data = self.rgb_pixels_to_hsv(rgb_pixels)[:, 0].reshape(-1, 1)
        else:
            raise ValueError('Unsupported color type')

        if not use_super_pixels:
            input_data = self.filter_pixel_percentage(input_data, dataset_percent)

        # min_samples_pct = 2
        # min_samples = int((len(input_data) / 100) * min_samples_pct)

        if clustering_method == 0:
            cluster = KMeans()
        elif clustering_method == 1:
            cluster = DBSCAN(eps=35)
        else:
            raise ValueError('Unsupported clustering method')

        return self.cluster_sample(cluster, input_data, rgb_pixels)

    @staticmethod
    def cluster_sample(clustering_method, clustering_samples, samples_rgb):
        """
        Method to cluster pixels with clustering method

        :param clustering_method: Usually Scikit class
        :param clustering_samples: Samples to use for clustering
        :param samples_rgb: clustering_samples doesn't have to be in RGB, this array is used to represent results in RGB
        """

        embedding = clustering_method.fit(clustering_samples)

        pixel_counter = defaultdict(int)
        avg_colors = defaultdict(list)

        for i, label in enumerate(embedding.labels_):
            pixel_counter[label] += 1
            avg_colors[label].append(samples_rgb[i])

        for key, values in avg_colors.items():
            # avg_colors[key] = np.median(values, axis=0).astype(int)
            # avg_colors[key] = np.min(values, axis=0).astype(int)
            avg_colors[key] = np.median(values, axis=0).astype(int)

        image_colors = []

        for key, item in pixel_counter.items():
            color_percentage = int(item / (len(clustering_samples) / 100))

            image_colors.append(
                [
                    color_percentage, avg_colors[key]
                ]
            )

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

    def __repr__(self):

        return f'ID {self.sample_id} Image {self.image.shape} GPS {self.gps_longitude}, {self.gps_latitude}'
