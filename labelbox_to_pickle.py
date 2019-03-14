from collections import defaultdict
import pandas as pd
import sys
from skimage import io
import json
import cv2
import numpy as np
import pickle

"""Script to convert LabelBox CSV to Pandas DF"""

pd.set_option('display.max_colwidth', 1000)

input_csv_file = sys.argv[1]
output_dir = sys.argv[2]
dataset_csv = sys.argv[3]

def process_labels(labels):

    if labels.get('colors_used', 'single_color') == 'single_color':
        colors_used = 1
    else:
        # More than one color used
        colors_used = 0

    return labels.get('graffiti_type', 'tag'), \
           colors_used, \
           labels.get('readable_text', ''), \
           labels.get('tool_used', 'marker')

def process_masks(labels, image_shape):

    if 'Background Graffiti' in labels['segmentationMasksByName']:
        background_graffiti = cv2.bitwise_not(cv2.inRange(io.imread(labels['segmentationMasksByName']['Background Graffiti']), 0, 0))
    else:
        background_graffiti = np.zeros(image_shape[0:2], dtype=np.uint8)

    return cv2.bitwise_not(cv2.inRange(io.imread(labels['segmentationMasksByName']['Background']), 0, 0)),\
           cv2.bitwise_not(cv2.inRange(io.imread(labels['segmentationMasksByName']['Graffiti']), 0, 0)), \
           cv2.bitwise_not(cv2.inRange(io.imread(labels['segmentationMasksByName']['Incomplete Graffiti']), 0, 0)), \
           background_graffiti

def check_pickle_file(pickle_file):

    sample = pickle.load(open(pickle_file, 'rb'))

    cv2.imwrite(f'test_images/{sample["id"]}_image.png', np.array(sample['image']))
    cv2.imwrite(f'test_images/{sample["id"]}_background_mask.png', sample['background_mask'])
    cv2.imwrite(f'test_images/{sample["id"]}_graffiti_mask.png', sample['graffiti_mask'])
    cv2.imwrite(f'test_images/{sample["id"]}_incomplete_graffiti.png', sample['incomplete_graffiti_mask'])
    cv2.imwrite(f'test_images/{sample["id"]}_background_graffiti_mask.png', sample['background_graffiti_mask'])

df = pd.DataFrame.from_csv(input_csv_file)

for col in list(df):

    print(df.head()[col])
    print()

original_datset_df = pd.read_csv(dataset_csv, index_col='hash_average')

print(original_datset_df)

for index, row in df.iterrows():

    data_sample = {}

    try:
        labels = json.loads(row['Label'])
    except json.decoder.JSONDecodeError:
        continue

    image = io.imread(row['Labeled Data'])

    print(labels)

    graffiti_type, colors_used, readable_text, tool_used = process_labels(labels)

    file_id = row['External ID'].split('.')[0]

    print(file_id)

    data_sample['id'] = file_id

    data_sample['image'] = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    data_sample['graffiti_type'] = graffiti_type
    data_sample['colors_used'] = colors_used
    data_sample['readable_text'] = readable_text
    data_sample['tool_used'] = tool_used

    background_mask, graffiti_mask, incomplete_graffiti_mask, background_graffiti_mask = process_masks(labels, image.shape)

    data_sample['background_mask'] = background_mask
    data_sample['graffiti_mask'] = graffiti_mask
    data_sample['incomplete_graffiti_mask'] = incomplete_graffiti_mask
    data_sample['background_graffiti_mask'] = background_graffiti_mask

    entry_from_original_datset = original_datset_df.loc[original_datset_df.index == file_id]

    data_sample['gps_longitude'] = entry_from_original_datset['gps_longitude'].values[0]
    data_sample['gps_latitude'] = entry_from_original_datset['gps_latitude'].values[0]

    print(data_sample)

    pickle.dump(data_sample, open(f'{output_dir}/{file_id}.p', 'wb'))

    check_pickle_file(f'{output_dir}/{file_id}.p')
