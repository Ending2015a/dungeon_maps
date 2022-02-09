# Dungeon Maps
A tiny PyTorch library for depth map manipulations.

Version: `0.0.2a1`

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

#### Depth maps
<img src="https://github.com/Ending2015a/dungeon_maps/blob/master/assets/demos_height_map.gif">

[(Watch this video in high quality)](https://youtu.be/vXpTaCOoH24)

This example shows how to project depth maps to top-down maps (plan view).
* Top left: RGB
* Top right: Depth map
* Bottom left: top-down maps in local space
* Bottom right: top-down maps in global space


Run this example
```shell
python -m dungeon_maps.demos.height_map.run
```
Control: `W`, `A`, `S`, `D`. `Q` for exit


#### Semantic segmentations

<img src="https://github.com/Ending2015a/dungeon_maps/blob/master/assets/demos_object_map.gif">

[(Watch this video in height quality)](https://youtu.be/QBa3fRzOnHI)

This example shows how to project arbitrary value maps, e.g. semantic segmentation, to top-down maps (plan view).
* Top left: RGB
* Top center: Depth map
* Top right: Semantic segmentation
* Bottom left: top-down maps in local space
* Bottom right: top-down maps in global space

Run this example
```shell
python -m dungeon_maps.demos.object_map.run
```
Control: `W`, `A`, `S`, `D`. `Q` for exit


### Egocentric motion flow

<img src="https://github.com/Ending2015a/dungeon_maps/blob/master/assets/demos_ego_flow.gif">

[(Watch this video in high quality)](https://youtu.be/q6HnNAVr2ps)

This example shows how to calculate the flow fields caused by camera motion.

Run this example
```shell
python -m dungeon_maps.demos.ego_flow.run
```

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

Install demos
```shell
pip install dungeon_maps[sim]
```
