from setuptools import setup
from setuptools import find_packages


setup(name='chi',
      version='0.1',
      description='Elegant TensorFlow',
      author='Simon Ramstedt',
      author_email='simonramstedt@gmail.com',
      url='https://github.com/rmst/chi',
      download_url='',
      license='MIT',
      install_requires=['tensorflow', 'matplotlib', 'flask'],
      extras_require={
          'gym': ['gym'],
      },
      scripts=['bin/chiboard'],
      packages=find_packages())
