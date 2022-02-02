# --- built in ---
import os
import re

from setuptools import find_packages, setup

package_name = 'dungeon_maps'
def get_version():
  with open(os.path.join(package_name, '__init__.py'), 'r') as f:
    return re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', f.read(), re.M).group(1)

setup(
  name=package_name,
  version=get_version(),
  description='A tiny PyTorch library for depth map manipulations',
  long_description=open('README.md', encoding='utf8').read(),
  long_description_content_type='text/markdown',
  url='https://github.com/Ending2015a/dungeon_map',
  author='JoeHsiao',
  author_email='joehsiao@gapp.nthu.edu.tw',
  license='MIT',
  python_requires=">=3.6",
  classifiers=[
    # How mature is this project? Common values are
    #   3 - Alpha
    #   4 - Beta
    #   5 - Production/Stable
    'Development Status :: 2 - Pre-Alpha',
    # Indicate who your project is intended for
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Software Development :: Libraries :: Python Modules',
    # Pick your license as you wish (should match "license" above)
    'License :: OSI Approved :: MIT License',
    # Specify the Python versions you support here. In particular, ensure
    # that you indicate whether you support Python 2, Python 3 or both.
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
  ],
  keywords='deep learning, computer vision, machine learning, '
    'robotics, depth map, point cloud, orthographic projection',
  packages=[
    # exclude deprecated module
    package for package in find_packages()
    if package.startswith(package_name)
  ],
  package_data={
    package_name: [
      'sim/data/*.fs', # shaders
      'sim/data/*.vs'
    ]
  },
  install_requires=[
    'numpy',
    'torch>=1.8.0',
    'torch-scatter'
  ],
  extras_require={
    'sim': [
      'moderngl',
      'opencv-python'
    ]
  }
)