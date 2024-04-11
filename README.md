# Dungeon Maps
*Dungeon Maps* is a powerful, lightweight PyTorch library for depth map manipulations, which is originally developed as a 2D mapping system for solving navigation problems: it can produce accurate 2D semantic top-down views from the depth map observations along with semantic segmentation predictions. See this [Habitat](https://github.com/facebookresearch/habitat-lab) object-goal navigation example:




https://user-images.githubusercontent.com/18180004/181456530-6222fb7d-c15a-4718-b2c0-4620b79feceb.mp4




*Dungeon Maps* also provides other functionalities, e.g. ego-centric motion flow calculation from depth maps. Scroll down to [Demos](#demos) to see more functionalities of this library.

This code is used by:
* HK Yang, TC Hsiao, et al. (2022). Investigation of Factorized Optical Flows as Mid-Level Representations. IROS 2022. [Paper](https://arxiv.org/abs/2203.04927)
* "Kemono", a rule-based object-goal navgiation agent for Habitat Challenge 2022. [Ending2015a/kemono-habitat-2022](https://github.com/Ending2015a/kemono-habitat-2022/tree/master)

Version: `0.0.3a1`

## Features

| | Batching | Multi-channels | GPU acceleration |
|:-:|:-:|:-:|:-:|
| Orthographic projection<br>(Top-down map) |:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
| Egocentric motion flow |:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
| 3D affine transformation<br>(Camera space) |:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
| 2D affine transformation<br>(Top-down view) |:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
| Map builder |:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|

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

## Demos

### Orthographic projection

#### Depth maps
<img src="https://github.com/Ending2015a/dungeon_maps/blob/master/assets/demos_height_map.gif">

[(Watch this video in high quality)](https://youtu.be/vXpTaCOoH24)

This example shows how to project depth maps to top-down maps.
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

This example shows how to project arbitrary value maps, e.g. semantic segmentation, to top-down maps.
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


### Ego-centric motion flow

<img src="https://github.com/Ending2015a/dungeon_maps/blob/master/assets/demos_ego_flow.gif">

[(Watch this video in high quality)](https://youtu.be/q6HnNAVr2ps)

This example shows how to calculate the flow fields caused by camera motion.

Run this example
```shell
python -m dungeon_maps.demos.ego_flow.run
```
