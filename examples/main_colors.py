

from graffiti_dataset.dataset import DatasetSample
from graffiti_dataset.tools import draw_main_colors

# sample = DatasetSample('./dataset/graffiti/8703f3c389a1f73f.p')

sample = DatasetSample('./dataset/graffiti/01df7f7f7e7e6000.p')


print(sample)

sample_main_colors = sample.main_colors()

print(sample_main_colors)

draw_main_colors(sample_main_colors, 'colors.png')


