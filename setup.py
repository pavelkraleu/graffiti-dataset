from setuptools import setup

setup(
    name='graffiti_dataset',
    version='1.1',
    description='Graffiti Dataset Toolbox',
    author='Pavel Kral',
    author_email='pavel@pavelkral.eu',
    packages=['graffiti_dataset'],
    tests_require=['pytest'],
    setup_requires=['pytest-runner'],
    install_requires=[
        'imageio',
        'folium',
        'opencv-python',
        'pandas',
        'Pillow',
        'scikit-image',
        'scikit-learn',
        'plotly',
        'numpy',
        'GPSPhoto',
        'imagehash',
        'exifread',
        'piexif'
    ],
)