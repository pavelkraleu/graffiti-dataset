

import glob
from graffiti_dataset.tools import draw_map

dataset_samples = glob.glob('./dataset/graffiti_sample/*.p')

draw_map(dataset_samples, f'./readme_images/map.html')

