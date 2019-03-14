

from graffiti_dataset.dataset import DatasetSample
import cv2

sample = DatasetSample('./dataset/graffiti/7f7fff8f89837f00.p')

print(sample)

cv2.imwrite(f'image.png', sample.image)
cv2.imwrite(f'graffiti_mask.png', sample.graffiti_mask)

sample.random_rotate()

cv2.imwrite(f'rotated_image.png', sample.image)
cv2.imwrite(f'rotated_mask.png', sample.graffiti_mask)

sample.elastic_transform()

cv2.imwrite(f'elastic_image.png', sample.image)
cv2.imwrite(f'elastic_mask.png', sample.graffiti_mask)

sample.randomize_color()



