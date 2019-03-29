import glob
from graffiti_dataset.dataset import DatasetSample
import numpy as np


def check_sample_data_types(sample):
    assert isinstance(sample, DatasetSample)

    assert isinstance(sample.image, np.ndarray)
    assert isinstance(sample.graffiti_mask, np.ndarray)
    assert isinstance(sample.background_graffiti_mask, np.ndarray)
    assert isinstance(sample.background_mask, np.ndarray)
    assert isinstance(sample.incomplete_graffiti_mask, np.ndarray)

    assert sample.image.dtype == 'uint8'
    assert sample.graffiti_mask.dtype == 'uint8'
    assert sample.background_graffiti_mask.dtype == 'uint8'
    assert sample.background_mask.dtype == 'uint8'
    assert sample.incomplete_graffiti_mask.dtype == 'uint8'

    assert sample.image.shape[0:2] == sample.graffiti_mask.shape[
                                      0:2] == sample.background_graffiti_mask.shape[
                                            0:2] == sample.background_mask.shape[
                                                 0:2] == sample.incomplete_graffiti_mask.shape[0:2]


def test_loading():
    for sample in glob.glob('./dataset/graffiti_sample/*.p'):
        sample = DatasetSample(sample)

        check_sample_data_types(sample)


def test_rotations():
    for sample in glob.glob('./dataset/graffiti_sample/*.p'):
        sample = DatasetSample(sample)

        sample.random_rotate()

        check_sample_data_types(sample)


def test_transformations():
    for sample in glob.glob('./dataset/graffiti_sample/*.p'):
        sample = DatasetSample(sample)

        sample.random_rotate()
        sample.elastic_transform()

        check_sample_data_types(sample)


def test_change_colors():
    for sample in glob.glob('./dataset/graffiti_sample/*.p'):
        sample = DatasetSample(sample)

        sample.random_rotate()
        sample.elastic_transform()
        sample.randomize_color()

        check_sample_data_types(sample)

def test_main_colors():
    for sample in glob.glob('./dataset/graffiti_sample/*.p'):
        sample = DatasetSample(sample)

        sample_main_colors = sample.main_colors()

        assert isinstance(sample_main_colors, list)

        for color_sample in sample_main_colors:

            color_pct, color_rgb = color_sample

            assert isinstance(color_pct, int)
            assert len(color_rgb) == 3