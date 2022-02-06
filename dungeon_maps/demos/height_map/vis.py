# --- built in ---
from typing import Union
# --- 3rd party ---
import numpy as np
import cv2
# --- my module ---
import dungeon_maps as dmap

# Colors [b, g, r]
hex2bgr = lambda hex: [int(hex[i:i+2], 16) for i in (0, 2, 4)][::-1]
FLOOR_COLOR   = hex2bgr('90D5C3')
WALL_COLOR    = hex2bgr('6798D0')
INVALID_COLOR = hex2bgr('F4F7FA')
CAMERA_COLOR  = hex2bgr('EC5565')
ORIGIN_COLOR  = hex2bgr('FFC300')


def draw_map(topdown_map: dmap.TopdownMap):
  occ_map = draw_occlusion_map(topdown_map.height_map, topdown_map.mask)
  occ_map = draw_origin(occ_map, topdown_map)
  occ_map = draw_camera(occ_map, topdown_map)
  return occ_map

def draw_occlusion_map(height_map, mask):
  """Draw occulution map: floor, wall, invalid area

  Args:
      height_map (torch.Tensor, np.ndarray): height map (b, c, h, w).
      mask (torch.Tensor, np.ndarray): mask (b, c, h, w).
  """
  height_map = dmap.utils.to_numpy(height_map[0, 0]) # (h, w)
  mask = dmap.utils.to_numpy(mask[0, 0]) # (h, w)
  height_threshold = 0.2
  floor_area = (height_map <= height_threshold) & mask
  wall_area = (height_map > height_threshold) & mask
  invalid_area = ~mask
  topdown_map = np.full(
    height_map.shape + (3,),
    fill_value=255, dtype=np.uint8
  ) # canvas (h, w, 3)
  topdown_map[invalid_area] = INVALID_COLOR
  topdown_map[floor_area] = FLOOR_COLOR
  topdown_map[wall_area] = WALL_COLOR
  return topdown_map

def draw_origin(
  image: np.ndarray,
  topdown_map: dmap.TopdownMap,
  color: np.ndarray = ORIGIN_COLOR,
  size: int = 4
):
  assert len(image.shape) == 3 # (h, w, 3)
  assert image.dtype == np.uint8
  assert topdown_map.proj is not None
  pos = np.array([
    [0., 0., 0.], # camera position
    [0., 0., 1.], # forward vector
    [0., 0., -1], # backward vector
    [-1, 0., 0.], # left-back vector
    [1., 0., 0.], # right-back vector
  ], dtype=np.float32)
  pos = topdown_map.get_coords(pos, is_global=True) # (b, 5, 2)
  pos = dmap.utils.to_numpy(pos)[0] # (5, 2)
  return draw_diamond(image, pos, color=color, size=size)

def draw_camera(
  image: np.ndarray,
  topdown_map: dmap.TopdownMap,
  color: np.ndarray = CAMERA_COLOR,
  size: int = 4
):
  assert len(image.shape) == 3 # (h, w, 3)
  assert image.dtype == np.uint8
  assert topdown_map.proj is not None
  pos = np.array([
    [0., 0., 0.], # camera position
    [0., 0., 1.], # forward vector
    [-1, 0., -1], # left-back vector
    [1., 0., -1], # right-back vector
  ], dtype=np.float32)
  pos = topdown_map.get_coords(pos, is_global=False) # (b, 4, 2)
  pos = dmap.utils.to_numpy(pos)[0] # (4, 2)
  return draw_arrow(image, pos, color=color, size=size)

def draw_arrow(image, points, color, size=2):
  # points: [center, forward, left, right]
  norm = lambda p: p/np.linalg.norm(p)
  c = points[0]
  f = norm(points[1] - points[0]) * (size*2) + points[0]
  l = norm(points[2] - points[0]) * (size*2) + points[0]
  r = norm(points[3] - points[0]) * (size*2) + points[0]
  pts = np.asarray([f, l, c, r], dtype=np.int32)
  return cv2.fillPoly(image, [pts], color=color)

def draw_diamond(image, points, color, size=2):
  # points [center, forward, back, left, right]
  norm = lambda p: p/np.linalg.norm(p)
  c = points[0]
  f = norm(points[1] - points[0]) * (size*2) + points[0]
  b = norm(points[2] - points[0]) * (size*2) + points[0]
  l = norm(points[3] - points[0]) * (size*2) + points[0]
  r = norm(points[4] - points[0]) * (size*2) + points[0]
  pts = np.asarray([f, l, b, r], dtype=np.int32)
  return cv2.fillPoly(image, [pts], color=color)

def draw_mark(image, point, color, size=2):
  radius = size
  thickness = radius + 2
  image = cv2.circle(image, (int(point[0]), int(point[1])),
      radius=radius, color=color, thickness=thickness)
  return image