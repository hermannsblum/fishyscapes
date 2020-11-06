from setuptools import setup


setup(name='fs',
      version="0.0",
      install_requires=[
          'bdlb @ git+https://github.com/hermannsblum/bdl-benchmark.git'
      ],
      packages=['fs', 'fs.data'])
