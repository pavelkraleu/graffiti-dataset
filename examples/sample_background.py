

from graffiti_dataset.dataset import DatasetSample
import cv2
from graffiti_dataset.tools import random_background

sample = DatasetSample('./dataset/graffiti/7f7fff8f89837f00.p')

print(sample)

sample.resize(1024, 1024)

background_image = random_background(1024, 1024, './dataset/background_images_sample/')

cv2.imwrite(f'background_image.png', background_image)

sample.paste_on_background(background_image)

cv2.imwrite(f'image.png', sample.image)
cv2.imwrite(f'graffiti_mask.png', sample.graffiti_mask)
cv2.imwrite(f'background_mask.png', sample.background_mask)
cv2.imwrite(f'incomplete_graffiti_mask.png', sample.incomplete_graffiti_mask)

