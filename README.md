# Dungeon Maps
A tiny PyTorch library for depth map manipulations.

## Features

| | Batching | Multi-channels | GPU acceleration |
|:-:|:-:|:-:|:-:|
| Orthographic projection<br>(Top-down map) |:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
| Egocentric motion flow |:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
| 3D affine transformation<br>(Camera space) |:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
| 2D affine transformation<br>(Top-down view) |:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
| Map builder |:heavy_check_mark:|:heavy_check_mark:|:heavy_minus_sign:|

## Demos

### Orthographic projection

<img src="https://github.com/Ending2015a/dungeon_maps/blob/master/assets/demos_height_map.gif">

[(Watch video in high resolution!)](https://youtu.be/vXpTaCOoH24)

Run this example
```shell
python -m dungeon_maps.demos.height_map.run
```

### Egocentric motion flow



## Installation

Basic requirements:
* Python >= 3.6
* PyTorch >= 1.8.0
* [PyTorch Scatter](https://github.com/rusty1s/pytorch_scatter)

Install from pip
```shell
pip install dungeon_maps
```

Install from GitHub repo
```shell
pip install git+https://github.com.Ending2015a/dungeon_map.git@master
```
