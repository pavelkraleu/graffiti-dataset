from setuptools import setup

setup(
   name='graffiti_dataset',
   version='1.1',
   description='Graffiti Dataset Toolbox',
   author='Pavel Kral',
   author_email='pavel@pavelkral.eu',
   packages=['graffiti_dataset'],
   install_requires=[
        'imageio',
        'folium',
        'opencv-python',
        'pandas',
        'Pillow',
        'scikit-image',
        'scikit-learn',
        'plotly'
   ]
)