#!/usr/bin/env python

import os
from setuptools import setup

os.system('pip install --no-cache-dir -r requirements.txt')

setup(name='cavelab',
      version='0.0.1',
      description='Python Computer Vision toolkit for rapid prototyping and massive application',
      author='Davit Buniatyan',
      author_email='davit@princeton.edu',
      url='loqsh.com',
      packages=['cavelab']
     )
