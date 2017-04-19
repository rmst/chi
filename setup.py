from setuptools import setup
from setuptools import find_packages

import sys
if sys.version_info < (3, 6):
  sys.exit('Sorry, Python < 3.6 is not supported')

setup(name='chi',
      version='0.2',
      description='',
      author='Simon Ramstedt',
      author_email='simonramstedt@gmail.com',
      url='https://github.com/rmst/chi',
      download_url='',
      license='MIT',
      install_requires=['tensorflow>=1.0.1', 'matplotlib>=2.0', 'flask>=0.12', 'flask_socketio', 'watchdog', 'gym'],
      extras_require={
          'rl': ['gym'],
      },
      scripts=['scripts/chiboard'],
      packages=find_packages())
