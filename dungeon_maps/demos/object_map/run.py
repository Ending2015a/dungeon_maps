# --- built in ---
import math
# --- 3rd party ---
import numpy as np
import torch
from torch import nn
import cv2
# --- my module ---
import dungeon_maps as dmap
# import simulators, so that one can use dmap.sim.make()
# to create simulators.
import dungeon_maps.sim
from dungeon_maps.demos.object_map import vis

# Some constants
WIDTH, HEIGHT = 800, 600 # pixel
HFOV = math.radians(70)
CAM_PITCH = math.radians(-10)
CAM_HEIGHT = 0.88 # meter
MIN_DEPTH = 0.1 # meter
MAX_DEPTH = 10.0 # meter
NUM_CLASSES = 5

def denormalize(depth_map):
  """Denormalize depth map, from [0, 1] to [MIN_DEPTH, MAX_DEPTH]"""
  return depth_map * (MAX_DEPTH - MIN_DEPTH) + MIN_DEPTH

def create_simulator():
  """Returns environment and MapProjector"""
  # Create simulator
  env = dmap.sim.make(
    'playground',
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
    fill_value = 0,
    to_global = True
  )
  build = dmap.MapBuilder(
    map_projector = proj
  )
  return env, build

def render_scene(rgb, depth, seg, world_map, local_map, cam_pose):
  bgr_image = rgb[...,::-1].astype(np.uint8) # (h, w, 3)
  depth_image = np.concatenate((depth,)*3, axis=-1) # (h, w, 3)
  depth_image = (depth_image*255.).astype(np.uint8)
  seg_image = vis.draw_segmentation(seg) # (h, w, 3)
  scene = np.concatenate((bgr_image, depth_image, seg_image), axis=1)
  # Plot occlusion map
  local_occ_map = vis.draw_map(local_map)
  cam_pos = world_map.get_camera()
  crop_map = world_map.select(cam_pos, 600, 600)
  crop_occ_map = vis.draw_map(crop_map)
  # Concat occlution maps
  local_occ_map = np.pad(local_occ_map, ((0, 0), (25, 25), (0, 0)),
      mode='constant', constant_values=0)
  crop_occ_map = np.pad(crop_occ_map, ((0, 0), (25, 25), (0, 0)),
      mode='constant', constant_values=0)
  occ_map = np.concatenate((local_occ_map, crop_occ_map), axis=1)
  # padding to same size
  pad_num = np.abs(occ_map.shape[1] - scene.shape[1])
  left_pad = pad_num//2
  right_pad = pad_num - left_pad
  if scene.shape[1] < occ_map.shape[1]:
    scene = np.pad(scene, ((0, 0), (left_pad, right_pad), (0, 0)),
        mode='constant', constant_values=0)
  elif scene.shape[1] > occ_map.shape[1]:
    occ_map = np.pad(occ_map, ((0, 0), (left_pad, right_pad), (0, 0)),
        mode='constant', constant_values=0)
  scene = np.concatenate((scene, occ_map), axis=0)
  return scene

def run_example():
  env, build = create_simulator()
  # Reset simulator and map builder
  observations = env.reset()
  build.reset()
  while True:
    # RGB image (h, w, 3), np.uint8
    rgb = observations['rgb']
    # Depth image (h, w, 1), np.float32
    depth = observations['depth']
    # Segmentation image (h, w, 1), np.int64
    seg = observations['segmentation']
    # Ground truth camera pose [x, z, yaw] in world coordinate
    cam_pose = observations['pose_gt'].astype(np.float32)
    # Denormalized depth map to [MIN_DEPTH, MAX_DEPTH]
    depth_map = np.transpose(denormalize(depth), (2, 0, 1)) # (1, h, w)
    # Project height map from depth map
    # One can enable GPU acceleration by converting depth map to
    # torch.Tensor and placing it on cuda devices. For example:
    #   depth_map = torch.tensor(depth_map, device='cuda')
    # other variables will be converted to torch.Tensor automatically.
    depth_map = torch.tensor(depth_map, device='cuda')
    seg_map = torch.tensor(seg, device='cuda')
    # Convert to one-hot encoding
    seg_map = seg_map.squeeze(dim=-1) # (h, w)
    seg_map = nn.functional.one_hot(seg_map, num_classes=NUM_CLASSES) # (h, w, c)
    seg_map = seg_map.permute((2, 0, 1)).to(dtype=torch.float32) # (c, h, w)
    local_map = build.step(
      depth_map = depth_map,
      value_map = seg_map,
      cam_pose = cam_pose,
      to_global = False,
      map_res = 0.015,
      width_offset = build.proj.map_width/2.,
      height_offset = 0.,
      map_width = 600,
      map_height = 600,
      center_mode = dmap.CenterMode.none,
      merge = False
    )
    # Merge local height map to world's height map
    build.merge(local_map, keep_pose=False)
    # render scene
    scene = render_scene(rgb, depth, seg, build.world_map, local_map, cam_pose)
    cv2.imshow('Object map', scene)
    # Taking actions via keyboard inputs
    key = cv2.waitKey(0)
    if key == ord('w'):
      action = env.FORWARD
    elif key == ord('a'):
      action = env.LEFT
    elif key == ord('s'):
      action = env.BACKWARD
    elif key == ord('d'):
      action = env.RIGHT
    elif key == ord('q'):
      print("Quit")
      exit()
    else:
      action = env.NONE
    observations = env.step(action)


if __name__ == '__main__':
    run_example()