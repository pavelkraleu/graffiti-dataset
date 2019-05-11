
import glob
from collections import defaultdict
import os
import imagehash
from GPSPhoto import gpsphoto
from PIL import Image
import pandas
import shutil
import sys

input_directory = sys.argv[1]
output_directory = sys.argv[2]
output_csv = sys.argv[3]

source_images = glob.glob(input_directory)

pandas_data = {
    'gps_latitude': [],
    'gps_longitude': [],
    'hash_average': [],
    'original_file_name': [],

}

for source_image in source_images:

    print(source_image)

    gps_data = gpsphoto.getGPSData(source_image)

    pandas_data['gps_latitude'].append(gps_data['Latitude'])
    pandas_data['gps_longitude'].append(gps_data['Longitude'])
    pandas_data['hash_average'].append(str(imagehash.average_hash(Image.open(source_image))))
    pandas_data['original_file_name'].append(os.path.basename(source_image))

    image = Image.open(source_image)
    image.thumbnail((1024,1024))
    image.save(f'{output_directory}/{pandas_data["hash_average"][-1]}.jpg')


df = pandas.DataFrame.from_dict(pandas_data)

print(df)

df.to_csv(output_csv)