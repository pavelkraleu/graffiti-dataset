

import glob
from graffiti_dataset.dataset import DatasetSample
from graffiti_dataset.tools import draw_map
import numpy as np

"""
Example how to use map function
"""

dataset_samples = glob.glob('./dataset/graffiti_sample/*.p')

gps_coordinates = []

for sample_path in dataset_samples:
    sample = DatasetSample(sample_path)
    gps_coordinates.append([sample.gps_latitude, sample.gps_longitude, sample.sample_id])

draw_map(np.array(gps_coordinates), f'./readme_images/map.html')

