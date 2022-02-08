# --- built in ---
import os
import math
# --- 3rd party ---
import numpy as np
import torch
import cv2
# --- my module ---
import dungeon_maps as dmap
# import simulators, so that one can use dmap.sim.make()
# to create simulators.
import dungeon_maps.sim
from dungeon_maps.demos.ego_flow import vis

# Some constants
WIDTH, HEIGHT = 800, 600
HFOV = math.radians(70)
CAM_PITCH = math.radians(-10)
CAM_HEIGHT = 0.88 # meter
MIN_DEPTH = 0.1 # meter
MAX_DEPTH = 10.0 # meter

def subtract_pose(p1, p2):
  """Caulate delta pose from p1 -> p2"""
  x1, y1, o1 = p1[...,0], p1[...,1], p1[...,2]
  x2, y2, o2 = p2[...,0], p2[...,1], p2[...,2]

  r = ((x1-x2)**2.0 + (y1-y2)**2.0)**0.5 # distance
  p = np.arctan2(y2-y1, x2-x1) - o1 #

  do = o2 - o1
  do = np.arctan2(np.sin(do), np.cos(do)) # [-pi/2, pi/2]
  dx = r * np.cos(p)
  dy = r * np.sin(p)
  return np.stack([dx, dy, do], axis=-1) # (batch, 3)

def denormalize(depth_map):
  """Denormalize depth map, from [0, 1] to [MIN_DEPTH, MAX_DEPTH]"""
  return depth_map * (MAX_DEPTH - MIN_DEPTH) + MIN_DEPTH

def create_simulator():
  """Returns environment and MapProjector"""
  # Create simulator
  env = dmap.sim.make(
    'forest',
    width = WIDTH,
    height = HEIGHT,
    hfov = HFOV,
    cam_pitch = CAM_PITCH,
    cam_height = CAM_HEIGHT,
    min_depth = MIN_DEPTH,
    max_depth = MAX_DEPTH
  )
  proj = dmap.MapProjector(
    width = WIDTH,
    height = HEIGHT,
    hfov = HFOV,
    vfov = None,
    cam_pose = [0., 0., 0.],
    width_offset = 0.,
    height_offset = 0.,
    cam_pitch = CAM_PITCH,
    cam_height = CAM_HEIGHT,
    map_res = 0.03,
    map_width = 600,
    map_height = 600,
    trunc_depth_min = 0.15,
    trunc_depth_max = 5.05,
    clip_border = 50,
    to_global = True
  )
  return env, proj

def compute_ego_flow(proj, depth, trans_pose):
  # Compute egocentric motion flow
  depth_map = np.transpose(denormalize(depth), (2, 0, 1)) # (1, h, w)
  depth_map = torch.tensor(depth_map, device='cuda')
  grid = proj.camera_affine_grid(depth_map, -trans_pose)
  x, y = dmap.utils.generate_image_coords(
    depth_map.shape,
    dtype = torch.float32,
    device = 'cuda'
  )
  coords = torch.stack((x, y), dim=-1)
  flow = coords - grid
  flow[..., 0] /= grid.shape[1]
  flow[..., 1] /= grid.shape[0]
  flow[..., 1] = -flow[..., 1] # flip y
  return flow[0, 0] # (h, w, 2)

def render_scene(rgb, depth, trans_pose, proj):
  bgr_image = rgb[...,::-1].astype(np.uint8) # (h, w, 3)
  depth_image = np.concatenate((depth,)*3, axis=-1) # (h, w, 3)
  depth_image = (depth_image*255.).astype(np.uint8)
  scene = np.concatenate((bgr_image, depth_image), axis=1)
  # Render egocentric motion flow
  flow = compute_ego_flow(proj, depth, trans_pose)
  flow_bgr = vis.draw_flow(flow)
  # padding to same size
  pad_num = np.abs(flow_bgr.shape[1] - scene.shape[1])
  left_pad = pad_num//2
  right_pad = pad_num - left_pad
  if scene.shape[1] < flow_bgr.shape[1]:
    scene = np.pad(scene, ((0, 0), (left_pad, right_pad), (0, 0)),
        mode='constant', constant_values=0)
  elif scene.shape[1] > flow_bgr.shape[1]:
    flow_bgr = np.pad(flow_bgr, ((0, 0), (left_pad, right_pad), (0, 0)),
        mode='constant', constant_values=0)
  scene = np.concatenate((scene, flow_bgr), axis=0)
  return scene

def run_example():
  env, proj = create_simulator()
  observations = env.reset()
  last_pose = np.array((0., 0., 0.), dtype=np.float32)
  while True:
    # RGB image (h, w, 3), torch.uint8
    rgb = observations['rgb']
    # Depth image (h, w, 1), torch.float32
    depth = observations['depth']
    # Ground truth camera pose [x, z, yaw] in world coordinate
    cam_pose = observations['pose_gt'].astype(np.float32)
    trans_pose = subtract_pose(last_pose, cam_pose)
    last_pose = cam_pose
    # render scene
    scene = render_scene(rgb, depth, trans_pose, proj)
    cv2.imshow('Ego motion flow', scene)
    key = cv2.waitKey(10)
    if key == ord('q'):
      print('Quit')
      exit()
    observations = env.step()


if __name__ == '__main__':
  run_example()