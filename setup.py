from setuptools import setup


setup(name='fs',
      version="0.0",
      install_requires=[
          'bdlb @ git+https://github.com/hermannsblum/bdl-benchmark.git',
          'gdown',
          'sacred',
          'pymongo',
          'imgaug',
          'tensorflow-datasets==3.1.0',
      ],
      packages=['fs', 'fs.data'])
