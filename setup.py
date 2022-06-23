# -*- coding: utf-8 -*-

import os
from distutils.core import setup

info = {}
with open("tracker/info.py", "r") as finfo:
    exec(finfo.read(), info)

description, long_description = info["__package_name__"], ""

'''scripts_dir = 'scripts'
scripts = []
for fname in os.listdir(scripts_dir):
    if fname.startswith('tracker-'):
        scripts.append("%s/%s" % (scripts_dir, fname))'''

setup(name=info["__package_name__"],
      version=info["__version__"],
      description=description,
      long_description=long_description,
      classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        ],
      packages=['tracker'],
      #scripts=scripts,
      ext_package='tracker'
      )