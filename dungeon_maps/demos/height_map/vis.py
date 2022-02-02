# --- built in ---
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
ORIGIN_COLOR  = hex2bgr('E788B8')



def draw_occlusion_map(height_map, mask):
  """Draw occulution map

  Args:
      height_map (torch.Tensor, np.ndarray): height map (b, c, h, w).
      mask (torch.Tensor, np.ndarray): mask (b, c, h, w).
  """
  height_map = dmap.utils.to_numpy(height_map[0, 0]) # (h, w)
  mask = dmap.utils.to_numpy(mask[0, 0]) # (h, w)
  floor_area = (height_map <= 0.2) & mask
  wall_area = (height_map > 0.2) & mask
  invalid_area = ~mask
  topdown_map = np.full(
    height_map.shape + (3,),
    fill_value=255, dtype=np.uint8
  ) # canvas (h, w, 3)
  topdown_map[invalid_area] = INVALID_COLOR
  topdown_map[floor_area] = FLOOR_COLOR
  topdown_map[wall_area] = WALL_COLOR
  return topdown_map

def draw_camera(image, map_projector, color=CAMERA_COLOR, size=4):
  assert len(image.shape) == 3
  assert image.dtype == np.uint8
  assert map_projector is not None
  proj = map_projector
  pos = np.array([[
    [0., 0., 0.], # camera position
    [0., 0., 1.], # forward vector
    [-1, 0., -1.], # left-back vector
    [1., 0., -1.], # right-back vector
  ]], dtype=np.float32)
  if proj.to_global:
    pos = proj.local_to_global_space(pos)
  pos_x, pos_z = proj.map_quantize(
    x_coords = pos[..., 0],
    z_coords = pos[..., 2]
  )
  pos_x = dmap.utils.to_numpy(pos_x) # (b, 4)
  pos_z = dmap.utils.to_numpy(pos_z)
  pos = np.stack((pos_x, pos_z), axis=-1)[0] # (4, 2)
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

def draw_mark(image, point, color, size=2):
  radius = size
  thickness = radius + 2
  image = cv2.circle(image, (int(point[0]), int(point[1])),
      radius=radius, color=color, thickness=thickness)
  return image