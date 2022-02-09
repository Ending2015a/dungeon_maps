# --- built in ---
import enum
from typing import List
# --- 3rd party ---
import numpy as np
import torch
from torch import nn
# --- my module ---
from dungeon_maps import utils

# ======= Utility functions =======

@enum.unique
class CenterMode(str, enum.Enum):
  none = "none"
  origin = "origin"
  camera = "camera"

def get(*args):
  """Return the first non-None argument"""
  for arg in args:
    if arg is not None:
      break
  return arg

def _mask_borders(
  masks: torch.Tensor,
  clip_border: int
):
  """Masking left/right/top/down borders by setting `masks` to False

  Args:
      masks (torch.Tensor): mask in shape (..., h, w). torch.bool
      clip_border (int): number of border pixels to mask.

  Returns:
      torch.Tensor: new mask in shape (..., h, w). torch.bool
  """
  assert torch.is_tensor(masks), \
    f"`masks` should be a torch.Tensor, got {type(masks)}"
  assert len(masks.shape) >= 2, "rank of `masks` should be greater than 2"
  device = masks.device
  *batch_dims, h, w = masks.shape # (..., h, w)
  # clip h
  masks[..., :clip_border, :] = False
  masks[..., -clip_border:, :] = False
  # clip w
  masks[..., :clip_border] = False
  masks[..., -clip_border:] = False
  return masks

def _masked_gather(values, indices, masks, fill_value=-np.inf):
  """Gather `values` with masked indices for some indices may be invalid
  indices, e.g. -1. Those elements are filled with `fill_value`.

  For example:
  ```
  values = [0.1, 0.2, 0.3]
  indices = [-1, 0, 1, 2]
  masks = [False, True, True, False]
  fill_value = -np.inf
  ```
  Then the return values of this method can be
  ```
  [-np.inf, 0.1, 0.2, -np.inf]
  ```
  since -1 and 2 in `indices` are masked.

  Args:
      values (torch.Tensor): value tensor. (b, ..., N)
      indices (torch.Tensor): indices of elements to gather. (b, ..., M)
      masks (torch.Tensor): masks (b, ..., M).
      fill_value (float, optional): default values to fill in. Defaults to -np.inf.

  Returns:
      torch.Tensor: value tensor (b, ..., M).
  """
  # broadcast shapes
  (values, indices, masks) = utils.validate_tensors(
    values, indices, masks, same_device=True
  )
  batch_shapes = (indices.shape[:-1], values.shape[:-1], masks.shape[:-1])
  batch = torch.broadcast_shapes(*batch_shapes)
  indices = torch.broadcast_to(indices, (*batch, indices.shape[-1]))
  values = torch.broadcast_to(values, (*batch, values.shape[-1]))
  masks = torch.broadcast_to(masks, (*batch, masks.shape[-1]))
  # shift indices
  indices[~masks] = -1
  dummy_shift = 1
  dummy_channel = torch.full_like(values[..., 0:1], fill_value)
  indices += dummy_shift
  values = torch.cat((dummy_channel, values), dim=-1)
  return torch.gather(values, dim=-1, index=indices)

# ======== Raw functional APIs =======
# We don't encourage you to call these APIs directly
# because you have to assign many arguments every time
# you call these APIs. Instead, you can use MapProjector
# to set the default arguments.

def orth_project(
  depth_map: torch.Tensor,
  value_map: torch.Tensor,
  cam_pose: torch.Tensor,
  width_offset: torch.Tensor,
  height_offset: torch.Tensor,
  cam_pitch: torch.Tensor,
  cam_height: torch.Tensor,
  map_res: float,
  map_width: int,
  map_height: int,
  focal_x: float,
  focal_y: float,
  center_x: float,
  center_y: float,
  trunc_depth_min: float,
  trunc_depth_max: float,
  clip_border: int,
  to_global: bool,
  flip_h: bool = True,
  get_height_map: bool = False,
  device: torch.device = None,
  _validate_args: bool = True
):
  """Project batch UNNORMALIZED depth maps to top-down maps or project
  value maps to top-down "value" maps (orthographic projection).

  The rank of `depth_map` must be at least 4D (b, c, h, w). If not, it is
  interpreted as (h, w), (c, h, w), (b, c, h, w) and (b, ..., h, w) for higher
  ranks.

  The `value_map` should have the same `h` and `w` dimensions as `depth_map`, but
  not for `c` dimension (channel), as it is the depth of vector values for each
  pixel. For example, `value_map` can be a semantic segmentation in one-hot
  categorical format. Note that each channel is projected separatly, which means
  each channel is independent. The values should be comparable by `max` operator.

  If `value_map` is None, the y-axis of the depth maps are used to project the
  height maps.

  Args:
      depth_map (torch.Tensor): UNNORMALIZED depth map, which means the range of
          values is [min_depth, max_depth]. torch.float32. The rank must be at
          least 4D (b, c, h, w). If not, it is converted automatically.
      value_map (torch.Tensor): value map to project. If this is set to None,
          the height map generated from `depth_map` is used. torch.float32.
      cam_pose (torch.Tensor): camera pose [x, z, yaw] with shape (b, 3), yaw in
          rad. torch.float32
      width_offset (torch.Tensor): batch pixel offset along map width (b,).
          torch.float32
      height_offset (torch.Tensor): batch pixel offset along map height (b,).
          torch.float32
      cam_pitch (torch.Tensor): batch camera pitch (b,). torch.float32
      cam_height (torch.Tensor): batch camera height (b,). torch.float32
      map_res (float): map resolution, unit per cell.
      map_width (int): map width, pixel.
      map_height (int): map height, pixel.
      focal_x (float): focal length on x direction.
      focal_y (float): focal length on y direction.
      center_x (float): center coordinate of depth map.
      center_y (float): center coordinate of depth map.
      trunc_depth_min (float): depth below this value is truncated. None to disable.
      trunc_depth_max (float): depth above this value is truncated. None to disable.
      clip_border (int): number of pixels to crop left/right/top/down borders.
      to_global (bool): convert to global space according to `cam_pose`.
      flip_h (bool, optional): whether to flip the horizontal axis. Note that in
          OpenCV format, the origin (0, 0) of an image is at the upper left corner,
          which should be flipped before converting to point cloud. Defaults
          to True.
      get_height_map (bool, optional): project height map from depth map and return
          it. Defaults to False.

  Returns:
      torch.Tensor: top-down maps. The value is set to -inf for invalid
          regions. One can use `np.isinf` or `masks` to check if it's valid.
          torch.float32
      torch.Tensor: masks, True for valid regions. torch.bool
      torch.Tensor, optional: height maps, indicating the height of each pixels
          on top-down maps. -inf for invalid regions. Returns this only when
          `get_height_map` is set to True.
  """
  if _validate_args:
    # Convert to tensors and ensure they are on the same device
    (depth_map, cam_pose, width_offset, height_offset,
      cam_pitch, cam_height) = utils.validate_tensors(
        depth_map, cam_pose, width_offset, height_offset,
        cam_pitch, cam_height,
        same_device = device or True
    )
    # Ensure tensor shape at least 4D (b, ..., h, w)
    depth_map = utils.to_4D_image(depth_map) # (b, c, h, w)
    # We don't broadcast shapes here since we don't
    # know the actual size of the batch dimensions, so
    # we remain the broadcasting parts to each sub-method.
    cam_pose = cam_pose.view(-1, 3) # (b, 3)
    cam_pitch = cam_pitch.view(-1) # (b,)
    cam_height = cam_height.view(-1) # (b,)
    width_offset = width_offset.view(-1) # (b,)
    height_offset = height_offset.view(-1) # (b,)
    # Ensure dtypes
    depth_map = depth_map.to(dtype=torch.float32)
    cam_pose = cam_pose.to(dtype=torch.float32)
    width_offset = width_offset.to(dtype=torch.float32)
    height_offset = height_offset.to(dtype=torch.float32)
    cam_pitch = cam_pitch.to(dtype=torch.float32)
    cam_height = cam_height.to(dtype=torch.float32)
    # Convert optional value maps
    if value_map is not None:
      device = depth_map.device
      value_map = utils.validate_tensors(
        value_map, same_device=device
      )
      value_map = utils.to_4D_image(value_map) # (b, c, h, w)
  device = depth_map.device
  # Convert depth map to point cloud
  point_cloud, valid_area = depth_map_to_point_cloud(
    depth_map = depth_map,
    focal_x = focal_x,
    focal_y = focal_y,
    center_x = center_x,
    center_y = center_y,
    trunc_depth_min = trunc_depth_min,
    trunc_depth_max = trunc_depth_max,
    flip_h = flip_h,
    _validate_args = False
  )  # (b, ..., h, w, 3)
  # Truncate border pixels of depth maps
  if (clip_border is not None) and (clip_border > 0):
    valid_area = _mask_borders(
      masks = valid_area,
      clip_border = clip_border
    )
  # Transform space from camera space to local space
  point_cloud = camera_to_local_space(
    points = point_cloud,
    cam_pitch = cam_pitch,
    cam_height = cam_height,
    _validate_args = False
  )
  # Transform space from local space to global space
  if to_global:
    point_cloud = local_to_global_space(
      points = point_cloud,
      cam_pose = cam_pose,
      _validate_args = False
    )
  # Flatten point cloud
  # (b, ..., h, w, 3) -> (b, ..., h*w, 3)
  flat_point_cloud = torch.flatten(point_cloud, -3, -2)
  # (b, ..., h, w) -> (b, ..., h*w)
  flat_mask = torch.flatten(valid_area, -2, -1)
  x_bin, z_bin = map_quantize(
    x_coords = flat_point_cloud[..., 0],
    z_coords = flat_point_cloud[..., 2],
    width_offset = width_offset,
    height_offset = height_offset,
    map_res = map_res,
    map_height = map_height,
    flip_h = flip_h,
    _validate_args = False
  )
  if value_map is None:
    # use the y coordinates as value_map
    flat_value_map = flat_point_cloud[..., 1]
  else:
    # (b, ..., h, w) -> (b, ..., h*w)
    flat_value_map = torch.flatten(value_map, -2, -1)
  # Projecting top-down map with orthographic projection
  coords = torch.stack((z_bin, x_bin), dim=-1)
  canvas_shape = (*flat_value_map.shape[:-1], map_height, map_width)
  canvas = torch.zeros(canvas_shape, device=device)
  topdown_map, masks, indices = project(
    coords = coords,
    values = flat_value_map,
    masks = flat_mask,
    canvas = canvas,
    fill_value = -np.inf,
    _validate_args = False
  )
  if get_height_map:
    if value_map is None:
      # See Note below...
      return topdown_map, masks, topdown_map
    # Projeciting height map
    flat_indices = torch.flatten(indices, -2, -1)
    flat_masks = torch.flatten(masks, -2, -1)
    flat_height_map = _masked_gather(
      values = flat_point_cloud[..., 1],
      indices = flat_indices,
      masks = flat_masks,
      fill_value = -np.inf
    )
    height_map = flat_height_map.view(topdown_map.shape)
    # Note: topdown_map should equal to height_map
    #    when value_map is not given.
    # a = utils.to_numpy(topdown_map)
    # a[a == -np.inf] = 0
    # b = utils.to_numpy(height_map)
    # b[b == -np.inf] = 0
    # assert np.allclose(a, b)
    return topdown_map, masks, height_map
  return topdown_map, masks

def camera_affine_grid(
  depth_map: torch.Tensor,
  trans_pose: torch.Tensor,
  cam_pitch: torch.Tensor,
  cam_height: torch.Tensor,
  focal_x: float,
  focal_y: float,
  center_x: float,
  center_y: float,
  flip_h: bool = True,
  device: torch.device = None,
  _validate_args: bool = True
):
  """Generates camera space flow fields with the given `depth_map` at time `t`.
  `trans_pose` indicates the transition of camera pose from time `t` to `t+1`.

  Args:
      depth_map (torch.Tensor): UNNORMALIZED depth map, which means the range of
          values is [min_depth, max_depth]. The rank must be at least 4D
          (b, c, h, w). If not, it is converted automatically as the following
              2D: (h, w)
              3D: (c, h, w)
              4D: (b, c, h, w)
              nD: (b, ..., h, w) for n > 4.
          torch.float32.
      trans_pose (torch.Tensor): camera pose transition [x, z, yaw] from time `t`
          to `t+1`. shape (b, 3). torch.float32
      cam_pitch (torch.Tensor): batch camera pitch (b,). torch.float32
      cam_height (torch.Tensor): batch camera height (b,). torch.float32
      focal_x (float): focal length on x direction.
      focal_y (float): focal length on y direction.
      center_x (float): center coordinate of depth map.
      center_y (float): center coordinate of depth map.
      flip_h (bool, optional): [description]. Defaults to True.
      _validate_args (bool, optional): [description]. Defaults to True.
  """
  if _validate_args:
    # Convert to tensors and ensure they are on the same device
    (depth_map, trans_pose, cam_pitch, cam_height) = utils.validate_tensors(
        depth_map, trans_pose, cam_pitch, cam_height,
        same_device = device or True
    )
    # Ensure tensor shape at least 4D (b, ..., h, w)
    depth_map = utils.to_4D_image(depth_map) # (b, c, h, w)
    # Note that we don't broadcast shapes here since we do
    # not sure the actual size of the batch dimensions, so
    # we remain the broadcasting parts to each sub-method.
    trans_pose = trans_pose.view(-1, 3) # (b, 3)
    cam_pitch = cam_pitch.view(-1) # (b,)
    cam_height = cam_height.view(-1) # (b,)
    # Ensure dtypes
    depth_map = depth_map.to(dtype=torch.float32)
    trans_pose = trans_pose.to(dtype=torch.float32)
    cam_pitch = cam_pitch.to(dtype=torch.float32)
    cam_height = cam_height.to(dtype=torch.float32)
  # Convert depth map to point cloud
  point_cloud, _ = depth_map_to_point_cloud(
    depth_map = depth_map,
    focal_x = focal_x,
    focal_y = focal_y,
    center_x = center_x,
    center_y = center_y,
    trunc_depth_min = None,
    trunc_depth_max = None,
    flip_h = flip_h,
    _validate_args = False
  )  # (b, ..., h, w, 3)
  # Transform space from camera space to local space
  point_cloud = camera_to_local_space(
    points = point_cloud,
    cam_pitch = cam_pitch,
    cam_height = cam_height,
    _validate_args = False
  )
  # Apply transition
  point_cloud = local_to_global_space(
    points = point_cloud,
    cam_pose = trans_pose,
    _validate_args = False
  )
  # Transform points from local to camera space
  point_cloud = local_to_camera_space(
    points = point_cloud,
    cam_pitch = cam_pitch,
    cam_height = cam_height,
    _validate_args = False
  )
  # Transform to image space
  point_cloud = camera_to_image_space(
    points = point_cloud, # (b, ..., h, w, 3)
    focal_x = focal_x,
    focal_y = focal_y,
    center_x = center_x,
    center_y = center_y,
    flip_h = flip_h,
    height = None,
    _validate_args = False
  )
  # Get affine grid (x, y)
  grid = point_cloud[..., 0:2] # (b, ..., h, w, 2)
  return grid

def depth_map_to_point_cloud(
  depth_map: torch.Tensor,
  focal_x: float,
  focal_y: float,
  center_x: float,
  center_y: float,
  trunc_depth_min: float,
  trunc_depth_max: float,
  flip_h: bool = True,
  device: torch.device = None,
  _validate_args: bool = True
):
  """Generate point clouds from `depth_map`. The generated point clouds are
  in the camera space. X: camera right, Y: camera up, Z: forward (depth).

  The rank of `depth_map` must be at least 4D (b, c, h, w). If not, it is
  interpreted as (h, w), (c, h, w), (b, c, h, w) and (b, ..., h, w) for higher
  ranks.

  Args:
      depth_map (torch.Tensor): UNNORMALIZED depth map, which means the range of
          values is [min_depth, max_depth]. torch.float32. The rank must be at
          least 4D (b, c, h, w). If not, it is converted automatically.
      focal_x (float): focal length on x direction.
      focal_y (float): focal length on y direction.
      center_x (float): center coordinate of depth map.
      center_y (float): center coordinate of depth map.
      trunc_depth_min (float): depth below this value is truncated. None to disable.
      trunc_depth_max (float): depth above this value is truncated. None to disable.
      flip_h (bool, optional): whether to flip the horizontal axis. Note that in
          OpenCV format, the origin (0, 0) of an image is at the upper left corner,
          which should be flipped before converting to point cloud. Defaults
          to True.

  Returns:
      torch.Tensor: point cloud in shape (..., 3)
      torch.Tensor: mask in shape (..., h, w) indicating the valid area.
  """
  if _validate_args:
    # Convert to tensors and ensure they are on the same device
    depth_map = utils.validate_tensors(depth_map, same_device=device or True)
    # Ensure tensor shape at least 4D (b, ..., h, w)
    depth_map = utils.to_4D_image(depth_map) # (b, c, h, w)
    # Ensure dtypes
    depth_map = depth_map.to(dtype=torch.float32)
  device = depth_map.device
  x, y = utils.generate_image_coords(
    depth_map.shape,
    dtype = torch.float32,
    device = device
  ) # same shape as depth_map
  z = depth_map # (..., h, w)
  points = torch.stack((x, y, z), dim=-1)
  point_cloud = image_to_camera_space(
    points = points,
    focal_x = focal_x,
    focal_y = focal_y,
    center_x = center_x,
    center_y = center_y,
    flip_h = flip_h,
    height = depth_map.shape[-2],
    _validate_args = False
  ) # (..., h, w, 3)
  valid_area = torch.ones_like(z, dtype=torch.bool) # (..., h, w)
  # Truncate invalid values
  if trunc_depth_max is not None:
    valid_area = torch.logical_and(z <= trunc_depth_max, valid_area)
  if trunc_depth_min is not None:
    valid_area = torch.logical_and(z >= trunc_depth_min, valid_area)
  return point_cloud, valid_area

def height_map_to_point_cloud(
  height_map: torch.Tensor,
  width_offset: torch.Tensor,
  height_offset: torch.Tensor,
  map_res: float,
  map_width: int,
  map_height: int,
  flip_h: bool = True,
  device: torch.device = None,
  _validate_args: bool = True
):
  """Generate point cloud from `height_map`.

  The rank of `height_map` must be at least 4D (b, c, h, w). If not, it is
  interpreted as (h, w), (c, h, w), (b, c, h, w) and (b, ..., h, w) for higher
  ranks.

  Args:
      height_map (torch.Tensor): UNNORMALIZED height map. torch.float32. The rank
          must be at least 4D (b, c, h, w). If not, it is converted automatically.
      width_offset (torch.Tensor): batch pixel offset along map width (b,).
          torch.float32
      height_offset (torch.Tensor): batch pixel offset along map height (b,).
          torch.float32
      map_res (float): map resolution, unit per cell.
      map_width (int): map width, pixel.
      map_height (int): map height, pixel.
      flip_h (bool, optional): whether to flip the horizontal axis. Note that
          in OpenCV format, the origin (0, 0) of an image is at the upper left
          corner, which should be flipped after converting to image space.
          Defaults to True.

  Returns:
      torch.Tensor: converted point cloud (b, c, h, w, 3). torch.float32
  """
  if _validate_args:
    # Convert to tensors and ensure they are on the same device
    (height_map, width_offset, height_offset) = utils.validate_tensors(
      height_map, width_offset, height_offset,
      same_device = device or True
    )
    # Ensure tensor shape at least 4D
    height_map = utils.to_4D_image(height_map) # (b, c, h, w)
    width_offset = width_offset.view(-1) # (b,)
    height_offset = height_offset.view(-1) # (b,)
    # Ensure dtypes
    height_map = height_map.to(dtype=torch.float32)
    width_offset = width_offset.to(dtype=torch.float32)
    height_offset = height_offset.to(dtype=torch.float32)
  device = height_map.device
  x_bin, z_bin = utils.generate_image_coords(
    height_map.shape,
    dtype = torch.float32,
    device = device
  )
  x, z = map_dequantize(
    x_coords = x_bin,
    z_coords = z_bin,
    width_offset = width_offset,
    height_offset = height_offset,
    map_res = map_res,
    map_height = map_height,
    flip_h = flip_h,
    _validate_args = False
  )
  y = height_map
  return torch.stack((x, y, z), dim=-1)

# ====== Space transform APIs =====

def image_to_camera_space(
  points: torch.Tensor,
  focal_x: float,
  focal_y: float,
  center_x: float,
  center_y: float,
  flip_h: bool = True,
  height: int = None,
  device: torch.device = None,
  _validate_args: bool = True
):
  """Transform 3D points in image space to camera space.
  The x, y coordinates of 3D points are the horizontal and vertical
  coordinates of pixels. z is the depth value.

  Args:
      points (torch.Tensor): points in shape (..., 3). torch.float32.
      focal_x (float): focal length on x direction.
      focal_y (float): focal length on y direction.
      center_x (float): center coordinate of depth map.
      center_y (float): center coordinate of depth map.
      flip_h (bool, optional): whether to flip the horizontal axis. Note that in
          OpenCV format, the origin (0, 0) of an image is at the upper left corner,
          which should be flipped after converting to image space. Defaults
          to True.
      height (int, optional): height of the image. This argument is used to flip
          the horizontal axis of 3D points. If the input shape of `points` are not
          (..., h, w, 3), this argument should be provided. Otheriwse, throw a
          RuntimeError. Defaults to None.

  Returns:
      torch.Tensor: transformed 3D points. (..., 3), torch.float32.
  """
  if _validate_args:
    # Convert to tensors and ensure they are on the same device
    points = utils.validate_tensors(points, same_device=device or True)
    # Ensure tensor shapes
    orig_shape = points.shape
    orig_ndims = len(orig_shape)
    if orig_ndims < 2:
      # pad batch dim, the rank of points should be at least 2
      points = points.view(-1, 3)
    # Ensure dtypes
    points = points.to(dtype=torch.float32)
  # Get height from the shape of `points`
  if flip_h and height is None:
    if len(points.shape) < 3:
      raise RuntimeError("The rank of `points` must be at least 3D (..., h, w, 3) "
          "or `height` should be provided if `flip_h` is enabled.")
    height = points.shape[-3] # h
  x = points[..., 0] # (...,)
  y = points[..., 1]
  z = points[..., 2]
  if flip_h:
    y = (height-1) - y
  cx = utils.to_tensor_like(center_x, points)
  cy = utils.to_tensor_like(center_y, points)
  fx = utils.to_tensor_like(focal_x, points)
  fy = utils.to_tensor_like(focal_y, points)
  # Convert to camera space
  x = (x-cx)/fx * z
  y = (y-cy)/fy * z
  points = torch.stack((x, y, z), dim=-1) # (..., 3)
  if _validate_args:
    points = points.view(orig_shape)
  return points

def camera_to_image_space(
  points: torch.Tensor,
  focal_x: float,
  focal_y: float,
  center_x: float,
  center_y: float,
  flip_h: bool = True,
  height: int = None,
  device: torch.device = None,
  _validate_args: bool = True
):
  """Transform 3D points in camera space to image space.
  The x, y coordinates of 3D points in image space are the horizontal and
  vertical coordinates of pixels. z is the depth value.

  Args:
      points (torch.Tensor): points in shape (..., 3). torch.float32
      focal_x (float): focal length on x direction.
      focal_y (float): focal length on y direction.
      center_x (float): center coordinate of depth map.
      center_y (float): center coordinate of depth map.
      flip_h (bool, optional): whether to flip the horizontal axis. Note that in
          OpenCV format, the origin (0, 0) of an image is at the upper left corner,
          which should be flipped after converting to image space. Defaults
          to True.
      height (int, optional): height of the image. This argument is used to flip
          the horizontal axis of 3D points. If the input shape of `points` are not
          (..., h, w, 3), this argument should be provided. Otheriwse, throw a
          RuntimeError. Defaults to None.

  Returns:
      torch.Tensor: point cloud in image space (..., h, w, 3). torch.float32
  """
  if _validate_args:
    # Convert to tensors and ensure they are on the same device
    points = utils.validate_tensors(points, same_device=device or True)
    # Ensure tensor shapes
    orig_shape = points.shape
    orig_ndims = len(orig_shape)
    if orig_ndims < 2:
      # pad batch dim, the rank of points should be at least 2
      points = points.view(-1, 3)
    # Ensure dtypes
    points = points.to(dtype=torch.float32)
  # Get height from the shape of `points`
  if flip_h and height is None:
    if len(points.shape) < 3:
      raise RuntimeError("The rank of `points` must be at least 3D (..., h, w, 3) "
          "or `height` should be provided if `flip_h` is enabled.")
    height = points.shape[-3] # h
  x = points[..., 0] # (...,)
  y = points[..., 1]
  z = points[..., 2]
  cx = utils.to_tensor_like(center_x, points)
  cy = utils.to_tensor_like(center_y, points)
  fx = utils.to_tensor_like(focal_x, points)
  fy = utils.to_tensor_like(focal_y, points)
  # Convert to image space
  z_eps = z + 1e-7
  x = x / z_eps * fx + cx
  y = y / z_eps * fy + cy
  if flip_h:
    y = (height-1) - y
  points = torch.stack((x, y, z), dim=-1) # (..., 3)
  if _validate_args:
    points = points.view(orig_shape)
  return points

def camera_to_local_space(
  points: torch.Tensor,
  cam_pitch: torch.Tensor,
  cam_height: torch.Tensor,
  device: torch.device = None,
  _validate_args: bool = True
):
  """Transform 3D points from camera space to local space

  Args:
      points (torch.Tensor): points in shape (b, ..., 3), torch.float32
      cam_pitch (torch.Tensor): batch camera pitch (b,). torch.float32
      cam_height (torch.Tensor): batch camera height (b,). torch.float32
      
  Returns:
      torch.Tensor: transformed points in shape (b, ..., 3), torch.float32
  """
  if _validate_args:
    (points, cam_pitch, cam_height) = utils.validate_tensors(
      points, cam_pitch, cam_height, same_device=device or True
    )
    # Ensure tensor shapes
    orig_shape = points.shape
    orig_ndims = len(orig_shape)
    if orig_ndims < 2:
      # pad batch dim, the rank of points should be at least 2
      points = points.view(-1, 3)
    batch = points.shape[0]
    points = points.view(batch, -1, 3) # (b, ..., 3)
    cam_pitch = cam_pitch.view(-1) # (b,)
    cam_height = cam_height.view(-1) # (b,)
    # Ensure dtypes
    points = points.to(dtype=torch.float32)
    cam_pitch = cam_pitch.to(dtype=torch.float32)
    cam_height = cam_height.to(dtype=torch.float32)
  # Rotate `cam_pitch` angle along x-axis
  points = utils.rotate(points, [1., 0., 0.], cam_pitch)
  zeros = torch.zeros_like(cam_height)
  x = zeros
  y = cam_height
  z = zeros # (b,)
  # Apply translations
  pos = torch.stack((x, y, z), dim=-1) # (b, 3)
  points = utils.translate(points, pos)
  if _validate_args:
    points = points.view(orig_shape)
  return points

def local_to_camera_space(
  points: torch.Tensor,
  cam_pitch: torch.Tensor,
  cam_height: torch.Tensor,
  device: torch.device = None,
  _validate_args: bool = True
):
  """Transform points from local space to camera space

  Args:
      points (torch.Tensor): points in shape (b, ..., 3), torch.float32
      cam_pitch (torch.Tensor): batch camera pitch (b,). torch.float32
      cam_height (torch.Tensor): batch camera height (b,). torch.float32

  Returns:
      torch.Tensor: transformed points in shape (b, ..., 3), torch.float32
  """
  if _validate_args:
    (points, cam_pitch, cam_height) = utils.validate_tensors(
      points, cam_pitch, cam_height, same_device=device or True
    )
    # Ensure tensor shapes
    orig_shape = points.shape
    orig_ndims = len(orig_shape)
    if orig_ndims < 2:
      # pad batch dim, the rank of points should be at least 2
      points = points.view(-1, 3)
    batch = points.shape[0]
    points = points.view(batch, -1, 3) # (b, ..., 3)
    cam_pitch = cam_pitch.view(-1)
    cam_height = cam_height.view(-1)
    # Ensure tensor dtypes
    points = points.to(dtype=torch.float32)
    cam_pitch = cam_pitch.to(dtype=torch.float32)
    cam_height = cam_height.to(dtype=torch.float32)
  zeros = torch.zeros_like(cam_height)
  x = zeros
  y = -cam_height
  z = zeros
  # Apply translations
  pos = torch.stack((x, y, z), dim=-1) # (b, 3)
  points = utils.translate(points, pos)
  points = utils.rotate(points, [1., 0., 0.], -cam_pitch)
  if _validate_args:
    points = points.view(orig_shape)
  return points

def local_to_global_space(
  points: torch.Tensor,
  cam_pose: torch.Tensor,
  device: torch.device = None,
  _validate_args: bool = True
):
  """Transform points from local space to global space

  Args:
      points (torch.Tensor): points in shape (b, ..., 3), torch.float32
      cam_pose (torch.Tensor): camera pose [x, z, yaw] with shape (b, 3), yaw in
          rad. torch.float32
  
  Returns:
      torch.Tensor: transformed points (b, ..., 3), torch.float32
  """
  if _validate_args:
    (points, cam_pose) = utils.validate_tensors(
      points, cam_pose, same_device=device or True
    )
    # Ensure tensor shapes
    orig_shape = points.shape
    orig_ndims = len(orig_shape)
    if orig_ndims < 2:
      # pad batch dim, the rank of point cloud should be at least 2
      points = points.view(-1, 3)
    batch = points.shape[0]
    points = points.view(batch, -1, 3) # (b, ..., 3)
    cam_pose = cam_pose.view(-1, 3)
    # Ensure tensor dtypes
    points = points.to(dtype=torch.float32)
    cam_pose = cam_pose.to(dtype=torch.float32)
  # Rotate yaw along y-axis
  yaw = cam_pose[..., 2] # (b,)
  points = utils.rotate(points, [0., 1., 0.], yaw)
  zeros = torch.zeros_like(yaw)
  x = cam_pose[..., 0]
  y = zeros
  z = cam_pose[..., 1] # (b,)
  # Apply translations
  pos = torch.stack((x, y, z), dim=-1) # (b, 3)
  points = utils.translate(points, pos)
  if _validate_args:
    points = points.view(orig_shape)
  return points

def global_to_local_space(
  points: torch.Tensor,
  cam_pose: torch.Tensor,
  device: torch.device = None,
  _validate_args: bool = True
):
  """Transform points from global space to local space

  Args:
      points (torch.tensor): points in shape (b, ..., 3), torch.float32
      cam_pose (torch.Tensor): camera pose [x, z, yaw] with shape (b, 3), yaw in
          rad. torch.float32

  Returns:
      torch.tensor: transformed points (b, ..., 3), torch.float32
  """
  if _validate_args:
    (points, cam_pose) = utils.validate_tensors(
      points, cam_pose, same_device=device or True
    )
    # Ensure tensor shapes
    orig_shape = points.shape
    orig_ndims = len(orig_shape)
    if orig_ndims < 2:
      # pad batch dim, the rank of points should be at least 2
      points = points.view(-1, 3)
    batch = points.shape[0]
    points = points.view(batch, -1, 3) # (b, ..., 3)
    cam_pose = cam_pose.view(-1, 3) # (b, 3)
    # Ensure tensor dtypes
    points = points.to(dtype=torch.float32)
    cam_pose = cam_pose.to(dtype=torch.float32)
  yaw = cam_pose[..., 2] # (b,)
  zeros = torch.zeros_like(yaw)
  x = cam_pose[..., 0]
  y = zeros
  z = cam_pose[..., 1] # (b,)
  # Apply translations
  pos = torch.stack((x, y, z), dim=-1) # (b, 3)
  points = utils.translate(points, -pos)
  # Rotate `yaw` angle along y-aixs
  points = utils.rotate(points, [0., 1., 0.], -yaw)
  if _validate_args:
    points = points.view(orig_shape)
  return points

def map_quantize(
  x_coords: torch.Tensor,
  z_coords: torch.Tensor,
  width_offset: torch.Tensor,
  height_offset: torch.Tensor,
  map_res: float,
  map_height: int = None,
  flip_h: bool = True,
  device: torch.device = None,
  _validate_args: bool = True
):
  """Quantize x, z coordinates of points into bins

  Args:
      x_coords (torch.Tensor): x coordinates. (b, ...) torch.float32
      z_coords (torch.Tensor): z coordinates. (b, ...) torch.float32
      width_offset (torch.Tensor): batch pixel offset along map width (b,).
          torch.float32
      height_offset (torch.Tensor): batch pixel offset along map height (b,).
          torch.float32
      map_res (float): map resolution, unit per cell.
      map_height (int, optional): map height, pixel. Defaults to None.
      flip_h (bool, optional): whether to flip the horizontal axis. Note that
          in OpenCV format, the origin (0, 0) of an image is at the upper left
          corner, which should be flipped after converting to image space.
          Defaults to True.

  Returns:
      torch.Tensor: indices of bins on x-axis for each point, torch.int64
      torch.Tensor: indices of bins on z-axis for each point, torch.int64
  """
  if _validate_args:
    (x_coords, z_coords, width_offset, height_offset) = utils.validate_tensors(
      x_coords, z_coords, width_offset, height_offset,
      same_device = device or True
    )
    # Ensure tensor shapes
    x_coords, z_coords = torch.broadcast_tensors(x_coords, z_coords)
    orig_shape = x_coords.shape
    orig_ndims = len(orig_shape)
    if orig_ndims < 2:
      # pad batch dim, the rank of point cloud should be at least 2
      x_coords = x_coords.view(1, -1) # (b, N)
      z_coords = z_coords.view(1, -1)
    width_offset = width_offset.view(-1) # (b,)
    height_offset = height_offset.view(-1) # (b,)
    # Ensure tensor dtypes
    x_coords = x_coords.to(dtype=torch.float32)
    z_coords = z_coords.to(dtype=torch.float32)
    width_offset = width_offset.to(dtype=torch.float32)
    height_offset = height_offset.to(dtype=torch.float32)
  device = x_coords.device
  ndims = len(x_coords.shape)
  # broadcast offsets to match the rank of coords
  width_offset = width_offset.view((-1,)+(1,)*(ndims-1))
  height_offset = height_offset.view((-1,)+(1,)*(ndims-1))
  x = x_coords
  z = z_coords
  # Z: (0, z) -> (far, near) + shift, X: (0, x) -> (left, right) + shift
  x_bin = x / map_res + width_offset
  z_bin = z / map_res + height_offset
  if flip_h:
    assert map_height is not None
    map_height = torch.tensor(map_height, device=device)
    z_bin = (map_height-1) - z_bin
  # Note that torch.round use half-to-even rounding
  # instead, we use up rounding.
  x_bin = torch.floor(x_bin+0.5).to(dtype=torch.int64)
  z_bin = torch.floor(z_bin+0.5).to(dtype=torch.int64)
  #x_bin = torch.round(x_bin).to(dtype=torch.int64)
  #z_bin = torch.round(z_bin).to(dtype=torch.int64)
  if _validate_args:
    x_coords = x_coords.view(orig_shape)
    z_coords = z_coords.view(orig_shape)
  return x_bin, z_bin

def map_dequantize(
  x_coords: torch.Tensor,
  z_coords: torch.Tensor,
  width_offset: torch.Tensor,
  height_offset: torch.Tensor,
  map_res: float,
  map_height: int = None,
  flip_h: bool = True,
  device: torch.device = None,
  _validate_args: bool = True
):
  """The inverse operation of `map_quantize`

  Args:
      x_coords (torch.Tensor): x coordinates. (b, ...) torch.float32
      z_coords (torch.Tensor): z coordinates. (b, ...) torch.float32
      width_offset (torch.Tensor): batch pixel offset along map width (b,).
          torch.float32
      height_offset (torch.Tensor): batch pixel offset along map height (b,).
          torch.float32
      map_res (float): map resolution, unit per cell.
      map_height (int, optional): map height, pixel. Defaults to None.
      flip_h (bool, optional): whether to flip the horizontal axis. Note that
          in OpenCV format, the origin (0, 0) of an image is at the upper left
          corner, which should be flipped before converting to points.
          Defaults to True.

  Returns:
      torch.Tensor: x coordinate. torch.float32
      torch.Tensor: z coordinate. torch.float32
  """
  if _validate_args:
    (x_coords, z_coords, width_offset, height_offset) = utils.validate_tensors(
      x_coords, z_coords, width_offset, height_offset,
      same_device = device or True
    )
    # Ensure tensor shapes
    x_coords, z_coords = torch.broadcast_tensors(x_coords, z_coords)
    orig_shape = x_coords.shape
    orig_ndims = len(orig_shape)
    if orig_ndims < 2:
      # pad batch dim, the rank of point cloud should be at least 2
      x_coords = x_coords.view(1, -1) # (b, N)
      z_coords = z_coords.view(1, -1)
    width_offset = width_offset.view(-1) # (b,)
    height_offset = height_offset.view(-1) # (b,)
    # Ensure tensor dtypes
    x_coords = x_coords.to(dtype=torch.float32)
    z_coords = z_coords.to(dtype=torch.float32)
    width_offset = width_offset.to(dtype=torch.float32)
    height_offset = height_offset.to(dtype=torch.float32)
  device = x_coords.device
  ndims = len(x_coords.shape)
  # broadcast offsets to match the rank of coords
  width_offset = width_offset.view((-1,)+(1,)*(ndims-1))
  height_offset = height_offset.view((-1,)+(1,)*(ndims-1))
  x_bin = x_coords
  z_bin = z_coords
  if flip_h:
    assert map_height is not None
    map_height = torch.tensor(map_height, device=device)
    z_bin = (map_height - 1) - z_bin
  z = (z_bin - height_offset) * map_res
  x = (x_bin - width_offset) * map_res
  return x, z

def project(
  coords: torch.Tensor,
  values: torch.Tensor,
  masks: torch.Tensor,
  canvas: torch.Tensor,
  canvas_masks: torch.Tensor = None,
  fill_value: float = -np.inf,
  device: torch.device = None,
  _validate_args: bool = True
):
  """Project `values` onto `canvas` with the specified `coords`.

  The shape of `coords` is (b, ..., n, 2), which stores the coordinates of
  `canvas`'s last two dimensions, usually are `h` and `w`, where `n` is the number
  of points. `...` is the arbitrary batch dimensions. The shape of `canvas` is
  (b, ..., mh, mw).

  Args:
      coords (torch.Tensor): flattened coordinates, shape (b, ..., n, 2).
          torch.float32.
      values (torch.Tensor): flattened values for each coordinate. shape
          (b, ..., n). torch.float32.
      masks (torch.Tensor): flattened mask for each coordinate, True for valid
          points. shape (b, ..., n). torch.float32.
      canvas (torch.Tensor): canvas to project onto, (b, ..., mh, mw).
          torch.float32.
      canvas_masks (torch.Tensor, optional): masks for canvas indicating which
          regions are valid or invalid. (b, ..., mh, mw). torch.bool
      fill_value (float, optional): default values to fill in the canvas. Set None
          to disable filling. Defaults to -np.inf.
  
  Returns:
      torch.Tensor: projected canvas in shape (b, ..., mh, mw)
      torch.Tensor: masks in shape (b, ..., mh, mw), False if the value is invalid,
          True otherwise.
      torch.Tensor: scattered indices (`coords`' indices) in shape (b, ..., mh, mw).
          -1 if there is no value scattered.
  """
  if _validate_args:
    (coords, values, masks, canvas, canvas_masks) = utils.validate_tensors(
      coords, values, masks, canvas, canvas_masks,
      same_device = device or True
    )
    if len(coords.shape) < 3:
      coords = coords.view(1, -1, 2) # at least 3 dims (b, n, 2)
    batch = (values.shape, masks.shape, coords.shape[:-1], canvas.shape[:-2])
    batch = torch.broadcast_shapes(*batch)
    coords = torch.broadcast_to(coords, batch+(2,)) # (b, ..., n, 2)
    values = torch.broadcast_to(values, batch) # (b, ..., n)
    masks = torch.broadcast_to(masks, batch) # (b, ..., n)
    canvas = torch.broadcast_to(canvas, batch+canvas.shape[-2:])
    # Ensure dtypes
    coords = coords.to(dtype=torch.int64)
    values = values.to(dtype=torch.float32)
    masks = masks.to(dtype=torch.bool)
    canvas = canvas.to(dtype=torch.float32)
  device = coords.device
  indices = coords
  # index bounds [0, 0] ~ [map_height, map_width]
  low_inds = 0
  up_inds = torch.tensor(
    canvas.shape[-2:], dtype=torch.int64, device=device
  )
  # Filtering invalid area (b, ..., n)
  valid_area = torch.cat((
    indices >= low_inds, indices < up_inds,
    masks.unsqueeze(dim=-1)
  ), dim=-1).all(dim=-1)
  maps, scattered_indices = utils.scatter_max(
    canvas = canvas,
    indices = indices,
    values = values,
    masks = valid_area,
    fill_value = fill_value,
    _validate_args = False
  ) # (b, ...., mh, mw)
  masks = ~(scattered_indices == -1) # (b, ..., mh, mw)
  # Merge canvas
  if canvas_masks is not None:
    canvas_masks = torch.broadcast_to(canvas_masks, masks.shape)
    canvas_masks = canvas_masks.to(dtype=torch.bool)
    masks = torch.logical_or(canvas_masks, masks)
  return maps, masks, scattered_indices

def compute_center_offsets(
  cam_pose: torch.Tensor,
  width_offset: torch.Tensor,
  height_offset: torch.Tensor,
  map_res: float,
  map_width: float,
  map_height: int,
  to_global: bool,
  center_mode: CenterMode = CenterMode.none,
  device: torch.device = None,
  _validate_args: bool = True
):
  """Compute offsets to the center of maps in different centering mode.

  Args:
      cam_pose (torch.Tensor): camera pose [x, z, yaw] with shape (b, 3), yaw in
          rad. torch.float32
      width_offset (torch.Tensor): additional pixel offsets along map width (b,).
          torch.float32
      height_offset (torch.Tensor): additional pixel offsets along map height (b,).
          torch.float32
      map_res (float): map resolution, unit per cell.
      map_width (int): map width, pixel.
      map_height (int): map height, pixel.
      to_global (bool): convert to global space according to `cam_pose`.
      center_mode (CenterMode, optional): centering mode. Defaults to
          CenterMode.none.

  Returns:
      torch.Tensor: pixel offsets along map width
      torch.Tensor: pixel offsets along map height
  """
  if center_mode is None:
    center_mode = CenterMode.none
  center_mode = CenterMode(center_mode)
  if _validate_args:
    if cam_pose is None:
      cam_pose = torch.zeros((3,))
    if width_offset is None:
      width_offset = 0.
    if height_offset is None:
      height_offset = 0.
    (cam_pose, width_offset, height_offset) = utils.validate_tensors(
      cam_pose, width_offset, height_offset,
      same_device = device or True,
      same_dtype = torch.float32
    )
  if center_mode is CenterMode.none:
    w_offset = 0.
    h_offset = 0.
  else:
    center_pos = torch.zeros_like(cam_pose)
    if center_mode is CenterMode.camera:
      if to_global:
        cam_pos = local_to_global_space(
          points = center_pos,
          cam_pose = cam_pose,
          _validate_args = True
        )
        center_pos = cam_pos
    # Compute center position in topdown map space
    center_pos_x, center_pos_z = map_quantize(
      x_coords = center_pos[..., 0],
      z_coords = center_pos[..., 2],
      width_offset = 0.,
      height_offset = 0.,
      map_res = map_res,
      map_height = map_height,
      flip_h = False
    )
    w_offset = map_width/2. - center_pos_x
    h_offset = map_height/2. - center_pos_z
  width_offset = width_offset + w_offset
  height_offset = height_offset + h_offset
  return width_offset, height_offset

# ========= Functional APIs =========

class MapProjector():
  def __init__(
    self,
    width: int,
    height: int,
    hfov: float,
    vfov: float = None,
    cam_pose: tuple = None,
    width_offset: float = None,
    height_offset: float = None,
    cam_pitch: float = None,
    cam_height: float = None,
    map_res: float = None,
    map_width: int = None,
    map_height: int = None,
    trunc_depth_min: float = None,
    trunc_depth_max: float = None,
    clip_border: int = None,
    to_global: bool = False,
    flip_h: bool = True,
    fill_value: float = -np.inf,
    device: torch.device = None
  ):
    """Map Projector
    This helper class provides a simple interface to access the raw functional APIs
    defined above. Some common arguments, e.g. camera intrinsics, are computed and
    stored as the default arguments, so that one can ignore those arguments when
    calling those APIs.

    Args:
        width (int): width of depth map. (pixel)
        height (int): height of depth map. (pixel)
        hfov (float): camera's horizontal field of view. (rad)
        vfov (float, optional): camera's verticle field of view (rad).
            This argument usually can be ignored since `hfov`=`vfov`.
            Defaults to None.
        cam_pose (tuple, optional): camera pose [x, z, yaw]. Defaults to None.
        width_offset (float, optional): offsets of image coordinates. Defaults to
            None.
        height_offset (float, optional): offsets of image coordinates. Defaults to
            None.
        cam_pitch (float, optional): camera pitch (rad). Defaults to None.
        cam_height (float, optional): camera height (unit). Defaults to None.
        map_res (float, optional): topdown map resolution (unit per cell).
            Defaults to None.
        map_width (int, optional): topdown map width (pixel). Defaults to None.
        map_height (int, optional): topdown map height (pixel). Defaults to None.
        trunc_depth_min (float, optional): minimum depth values to truncate (unit).
            Defaults to None.
        trunc_depth_max (float, optional): maximum depth values to truncate (unit).
            Defaults to None.
        clip_border (int, optional): number of border pixels to clip. Defaults to
            None.
        to_global (bool, optional): transform to global space when projecting.
            Defaults to False.
        flip_h (bool, optional): whether to flip the horizontal axis. Note that in
            OpenCV format, the origin (0, 0) of an image is at the upper left
            corner, which should be flipped before converting to point cloud.
            Defaults to True.
        fill_value (float, optional): default values to fill in the map when
            projecting. Defaults to -np.inf.
        device (torch.device, optional): torch device. Defaults to None.
    """
    self.width = width
    self.height = height
    self.hfov = hfov
    self.vfov = vfov
    self.cam_pose = cam_pose
    self.width_offset = width_offset
    self.height_offset = height_offset
    self.cam_pitch = cam_pitch
    self.cam_height = cam_height
    self.map_res = map_res
    self.map_width = map_width
    self.map_height = map_height
    self.trunc_depth_min = trunc_depth_min
    self.trunc_depth_max = trunc_depth_max
    self.clip_border = clip_border
    self.to_global = to_global
    self.flip_h = flip_h
    self.fill_value = fill_value
    self.device = device
    # compute camera intrinsic
    self.cam_params = utils.get_camera_intrinsics(
      width = self.width,
      height = self.height,
      hfov = self.hfov,
      vfov = self.vfov
    )

  def clone(
    self,
    width: int = None,
    height: int = None,
    hfov: float = None,
    vfov: float = None,
    cam_pose: tuple = None,
    width_offset: float = None,
    height_offset: float = None,
    cam_pitch: float = None,
    cam_height: float = None,
    map_res: float = None,
    map_width: int = None,
    map_height: int = None,
    trunc_depth_min: float = None,
    trunc_depth_max: float = None,
    clip_border: int = None,
    to_global: bool = None,
    flip_h: bool = None,
    device: torch.device = None
  ):
    """Clone MapProjector and override some arguments. Note that
    this method only do a shallow copy for each argument.

    Args:
        See MapProjector.__init__

    Returns:
        MapProjector: 
    """
    return MapProjector(
      width = get(width, self.width),
      height = get(height, self.height),
      hfov = get(hfov, self.hfov),
      vfov = get(vfov, self.vfov),
      cam_pose = get(cam_pose, self.cam_pose),
      width_offset = get(width_offset, self.width_offset),
      height_offset = get(height_offset, self.height_offset),
      cam_pitch = get(cam_pitch, self.cam_pitch),
      cam_height = get(cam_height, self.cam_height),
      map_res = get(map_res, self.map_res),
      map_width = get(map_width, self.map_width),
      map_height = get(map_height, self.map_height),
      trunc_depth_min = get(trunc_depth_min, self.trunc_depth_min),
      trunc_depth_max = get(trunc_depth_max, self.trunc_depth_max),
      clip_border = get(clip_border, self.clip_border),
      to_global = get(to_global, self.to_global),
      flip_h = get(flip_h, self.flip_h),
      device = get(device, self.device)
    )

  def orth_project(
    self,
    depth_map: torch.Tensor,
    value_map: torch.Tensor = None,
    cam_pose: torch.Tensor = None,
    width_offset: torch.Tensor = None,
    height_offset: torch.Tensor = None,
    cam_pitch: torch.Tensor = None,
    cam_height: torch.Tensor = None,
    map_res: float = None,
    map_width: int = None,
    map_height: int = None,
    focal_x: float = None,
    focal_y: float = None,
    center_x: float = None,
    center_y: float = None,
    trunc_depth_min: float = None,
    trunc_depth_max: float = None,
    clip_border: int = None,
    to_global: bool = None,
    flip_h: bool = None,
    get_height_map: bool = False,
    device: torch.device = None,
    _validate_args: bool = True
  ):
    return orth_project(
      depth_map = depth_map,
      value_map = value_map,
      cam_pose = get(cam_pose, self.cam_pose),
      width_offset = get(width_offset, self.width_offset),
      height_offset = get(height_offset, self.height_offset),
      cam_pitch = get(cam_pitch, self.cam_pitch),
      cam_height = get(cam_height, self.cam_height),
      map_res = get(map_res, self.map_res),
      map_width = get(map_width, self.map_width),
      map_height = get(map_height, self.map_height),
      focal_x = get(focal_x, self.cam_params.fx),
      focal_y = get(focal_y, self.cam_params.fy),
      center_x = get(center_x, self.cam_params.cx),
      center_y = get(center_y, self.cam_params.cy),
      trunc_depth_min = get(trunc_depth_min, self.trunc_depth_min),
      trunc_depth_max = get(trunc_depth_max, self.trunc_depth_max),
      clip_border = get(clip_border, self.clip_border),
      to_global = get(to_global, self.to_global),
      flip_h = get(flip_h, self.flip_h),
      get_height_map = get_height_map,
      device = get(device, self.device),
      _validate_args = _validate_args
    )

  def camera_affine_grid(
    self,
    depth_map: torch.Tensor,
    trans_pose: torch.Tensor,
    cam_pitch: torch.Tensor = None,
    cam_height: torch.Tensor = None,
    focal_x: float = None,
    focal_y: float = None,
    center_x: float = None,
    center_y: float = None,
    flip_h: bool = None,
    device: torch.device = None,
    _validate_args: bool = True
  ):
    return camera_affine_grid(
      depth_map = depth_map,
      trans_pose = trans_pose,
      cam_pitch = get(cam_pitch, self.cam_pitch),
      cam_height = get(cam_height, self.cam_height),
      focal_x = get(focal_x, self.cam_params.fx),
      focal_y = get(focal_y, self.cam_params.fy),
      center_x = get(center_x, self.cam_params.cx),
      center_y = get(center_y, self.cam_params.cy),
      flip_h = get(flip_h, self.flip_h),
      device = get(device, self.device),
      _validate_args = _validate_args
    )

  def depth_map_to_point_cloud(
    self,
    depth_map: torch.Tensor,
    focal_x: float = None,
    focal_y: float = None,
    center_x: float = None,
    center_y: float = None,
    trunc_depth_min: float = None,
    trunc_depth_max: float = None,
    flip_h: bool = None,
    device: torch.device = None,
    _validate_args: bool = True
  ):
    return depth_map_to_point_cloud(
      depth_map = depth_map,
      focal_x = get(focal_x, self.cam_params.fx),
      focal_y = get(focal_y, self.cam_params.fy),
      center_x = get(center_x, self.cam_params.cx),
      center_y = get(center_y, self.cam_params.cy),
      trunc_depth_min = get(trunc_depth_min, self.trunc_depth_min),
      trunc_depth_max = get(trunc_depth_max, self.trunc_depth_max),
      flip_h = get(flip_h, self.flip_h),
      device = get(device, self.device),
      _validate_args = _validate_args
    )

  def height_map_to_point_cloud(
    self,
    height_map: torch.Tensor,
    width_offset: torch.Tensor = None,
    height_offset: torch.Tensor = None,
    map_res: float = None,
    map_width: int = None,
    map_height: int = None,
    flip_h: bool = None,
    device: torch.device = None,
    _validate_args: bool = True
  ):
    return height_map_to_point_cloud(
      height_map = height_map,
      width_offset = get(width_offset, self.width_offset),
      height_offset = get(height_offset, self.height_offset),
      map_res = get(map_res, self.map_res),
      map_width = get(map_width, self.map_width),
      map_height = get(map_height, self.map_height),
      flip_h = get(flip_h, self.flip_h),
      device = get(device, self.device),
      _validate_args = _validate_args
    )

  def image_to_camera_space(
    self,
    points: torch.Tensor,
    focal_x: float = None,
    focal_y: float = None,
    center_x: float = None,
    center_y: float = None,
    flip_h: bool = None,
    height: int = None,
    device: torch.device = None,
    _validate_args: bool = True
  ):
    return image_to_camera_space(
      points = points,
      focal_x = get(focal_x, self.cam_params.fx),
      focal_y = get(focal_y, self.cam_params.fy),
      center_x = get(center_x, self.cam_params.cx),
      center_y = get(center_y, self.cam_params.cy),
      flip_h = get(flip_h, self.flip_h),
      height = get(height, self.height),
      device = get(device, self.device),
      _validate_args = _validate_args
    )

  def camera_to_image_space(
    self,
    points: torch.Tensor,
    focal_x: float = None,
    focal_y: float = None,
    center_x: float = None,
    center_y: float = None,
    flip_h: bool = None,
    height: int = None,
    device: torch.device = None,
    _validate_args: bool = True
  ):
    return camera_to_image_space(
      points = points,
      focal_x = get(focal_x, self.cam_params.fx),
      focal_y = get(focal_y, self.cam_params.fy),
      center_x = get(center_x, self.cam_params.cx),
      center_y = get(center_y, self.cam_params.cy),
      flip_h = get(flip_h, self.flip_h),
      height = get(height, self.height),
      device = get(device, self.device),
      _validate_args = _validate_args
    )

  def camera_to_local_space(
    self,
    points: torch.Tensor,
    cam_pitch: torch.Tensor = None,
    cam_height: torch.Tensor = None,
    device: torch.device = None,
    _validate_args: bool = True
  ):
    return camera_to_local_space(
      points = points,
      cam_pitch = get(cam_pitch, self.cam_pitch),
      cam_height = get(cam_height, self.cam_height),
      device = get(device, self.device),
      _validate_args = _validate_args
    )

  def local_to_camera_space(
    self,
    points: torch.Tensor,
    cam_pitch: torch.Tensor = None,
    cam_height: torch.Tensor = None,
    device: torch.device = None,
    _validate_args: bool = True
  ):
    return local_to_camera_space(
      points = points,
      cam_pitch = get(cam_pitch, self.cam_pitch),
      cam_height = get(cam_height, self.cam_height),
      device = get(device, self.device),
      _validate_args = _validate_args
    )

  def local_to_global_space(
    self,
    points: torch.Tensor,
    cam_pose: torch.Tensor = None,
    device: torch.device = None,
    _validate_args: bool = True
  ):
    return local_to_global_space(
      points = points,
      cam_pose = get(cam_pose, self.cam_pose),
      device = get(device, self.device),
      _validate_args = _validate_args
    )

  def global_to_local_space(
    self,
    points: torch.Tensor,
    cam_pose: torch.Tensor = None,
    device: torch.device = None,
    _validate_args: bool = True
  ):
    return global_to_local_space(
      points = points,
      cam_pose = get(cam_pose, self.cam_pose),
      device = get(device, self.device),
      _validate_args = _validate_args
    )

  def map_quantize(
    self,
    x_coords: torch.Tensor,
    z_coords: torch.Tensor,
    width_offset: torch.Tensor = None,
    height_offset: torch.Tensor = None,
    map_res: float = None,
    map_height: int = None,
    flip_h: bool = None,
    device: torch.device = None,
    _validate_args: bool = True
  ):
    return map_quantize(
      x_coords = x_coords,
      z_coords = z_coords,
      width_offset = get(width_offset, self.width_offset),
      height_offset = get(height_offset, self.height_offset),
      map_res = get(map_res, self.map_res),
      map_height = get(map_height, self.map_height),
      flip_h = get(flip_h, self.flip_h),
      device = get(device, self.device),
      _validate_args = _validate_args
    )

  def map_dequantize(
    self,
    x_coords: torch.Tensor,
    z_coords: torch.Tensor,
    width_offset: torch.Tensor = None,
    height_offset: torch.Tensor = None,
    map_res: float = None,
    map_height: int = None,
    flip_h: bool = None,
    device: torch.device = None,
    _validate_args: bool = True
  ):
    return map_dequantize(
      x_coords = x_coords,
      z_coords = z_coords,
      width_offset = get(width_offset, self.width_offset),
      height_offset = get(height_offset, self.height_offset),
      map_res = get(map_res, self.map_res),
      map_height = get(map_height, self.map_height),
      flip_h = get(flip_h, self.flip_h),
      device = get(device, self.device),
      _validate_args = _validate_args
    )

  def project(
    self,
    coords: torch.Tensor,
    values: torch.Tensor,
    masks: torch.Tensor,
    canvas: torch.Tensor,
    canvas_masks: torch.Tensor = None,
    fill_value: float = None,
    device: torch.device = None,
    _validate_args: bool = True
  ):
    return project(
      coords = coords,
      values = values,
      masks = masks,
      canvas = canvas,
      canvas_masks = canvas_masks,
      fill_value = get(fill_value, self.fill_value),
      device = get(device, self.device),
      _validate_args = _validate_args
    )

  def compute_center_offsets(
    self,
    cam_pose: torch.Tensor = None,
    width_offset: torch.Tensor = None,
    height_offset: torch.Tensor = None,
    map_res: float = None,
    map_width: int = None,
    map_height: int = None,
    to_global: bool = None,
    center_mode: CenterMode = CenterMode.none,
    device: torch.device = None,
    _validate_args: bool = True
  ):
    return compute_center_offsets(
      cam_pose = get(cam_pose, self.cam_pose),
      width_offset = get(width_offset, self.width_offset),
      height_offset = get(height_offset, self.height_offset),
      map_res = get(map_res, self.map_res),
      map_width = get(map_width, self.map_width),
      map_height = get(map_height, self.map_height),
      to_global = get(to_global, self.to_global),
      center_mode = center_mode,
      device = get(device, self.device),
      _validate_args = _validate_args
    )

# ===== Object-oriented APIs =====

class TopdownMap():
  """A simple top-down map wrapper provides common top-down map
  manipulations, e.g. crop and merge.
  """
  def __init__(
    self,
    topdown_map: torch.Tensor = None,
    mask: torch.Tensor = None,
    height_map: torch.Tensor = None,
    map_projector: MapProjector = None,
    is_height_map: bool = None
  ):
    """Create top-down map.

    Args:
        topdown_map (torch.Tensor, optional): top-down map. Defaults to None.
        mask (torch.Tensor, optional): mask. Defaults to None.
        height_map (torch.Tensor, optional): height map. Defaults to None.
        map_projector (MapProjector, optional): map projector. Defaults to None.
        is_height_map (bool, optional): determine if this top-down map is a height
            map. Defaults to None.
    """
    self._proj = map_projector
    self._topdown_map = topdown_map
    self._mask = mask
    self._height_map = height_map
    # If is_height_map is None, check if topdown_map and
    # height_map are the same instance.
    if is_height_map is None:
      is_height_map = (not self.is_empty
        and topdown_map is height_map)
    self._is_height_map = is_height_map

  @property
  def is_empty(self) -> bool:
    """Return True if `topdown_map` is not set"""
    return self.topdown_map is None

  @property
  def is_height_map(self) -> bool:
    """Whether this top-down map is a height map"""
    return self._is_height_map

  @property
  def map(self) -> torch.Tensor:
    """Shortcut to self.topdown_map"""
    return self._topdown_map
  
  @property
  def topdown_map(self) -> torch.Tensor:
    """Return topdown map"""
    return self._topdown_map

  @property
  def height_map(self) -> torch.Tensor:
    """Return height map"""
    if self.is_height_map:
      return self.topdown_map
    else:
      return self._height_map

  @property
  def mask(self) -> torch.Tensor:
    """Return mask"""
    return self._mask

  @property
  def proj(self) -> MapProjector:
    """Return map projector of this map"""
    return self._proj

  def get_camera(
    self
  ):
    """Get the image coordinates of camera

    Returns:
        torch.Tensor: camera position in image coordinates.
            (b, 2). torch.int64
    """
    # The camera a.k.a. local origin is at (0, 0, 0)
    cam_pos = torch.zeros((3,), dtype=torch.float32)
    cam_pos = self.get_coords(
      points = cam_pos,
      is_global = False
    ) # (b, 1, 2)
    return cam_pos.squeeze(dim=-2) # (b, 2)

  def get_origin(
    self,
  ):
    """Get the image coordinates of global origin

    Returns:
        torch.Tensor: camera position in image coordinates.
            (b, 2). torch.int64
    """
    # The global origin is at (0, 0, 0)
    origin_pos = torch.zeros((3,), dtype=torch.float32)
    origin_pos = self.get_coords(
      points = origin_pos,
      is_global = True
    ) # (b, 1, 2)
    return origin_pos.squeeze(dim=-2) # (b, 2)

  def get_coords(
    self,
    points: torch.Tensor,
    is_global: bool = True
  ):
    """Get the image coordinates of `points` in global space or local space

    Args:
        points (torch.Tensor): points in shape (3,), (n, 3), (b, n, 3),
            (b, ..., n, 3), where `n` is the number of points. torch.float32.
        is_global (bool, optional): whether this points is in global space.
            Defaults to True.

    Returns:
        torch.Tensor: image coordinates of the points. (b, n, 2) or (b, ..., n, 2).
            torch.int64.
    """
    # Ensure tensor shape (b, n, 3)
    points = utils.to_tensor(points)
    if len(points.shape) < 3:
      points = points.view(1, -1, 3)
    if self.proj.to_global:
      if not is_global:
        # Convert to global coordinates
        points = self.proj.local_to_global_space(
          points = points
        )
    else:
      if is_global:
        # Convert to local coordinates
        points = self.proj.global_to_local_space(
          points = points
        )
    # Convert to image coordinates
    pos_x, pos_z = self.proj.map_quantize(
      x_coords = points[..., 0],
      z_coords = points[..., 2]
    )
    pos = torch.stack((pos_x, pos_z), dim=-1) # (b, n, 2)
    return pos

  def get_points(
    self,
    coords: torch.Tensor
  ):
    """Get x, z coordinates of `coords` in image coordinates

    Args:
        coords (torch.Tensor): image coordinates. (2,), (n, 2), (b, n, 2),
            (b, ..., n, 2). torch.int64
    
    Returns:
        torch.Tensor: x, z coordinates. (b, n, 2) or (b, ..., n, 2).
            torch.float32
    """
    coords = utils.to_tensor(coords)
    if len(coords.shape) < 3:
      coords = coords.view(1, -1, 2)
    pos_x, pos_z = self.proj.map_dequantize(
      x_coords = coords[..., 0],
      z_coords = coords[..., 1]
    )
    pos = torch.stack((pos_x, pos_z), dim=-1) # (b, n, 2)
    return pos

  def select(
    self,
    center: torch.Tensor,
    crop_width: int,
    crop_height: int,
    fill_value: float = None,
  ):
    """Select map region, a.k.a. "resize with crop or pad"

    Args:
        center (torch.Tensor): center coordinates of the region. (b, 2).
            torch.int64
        crop_width (int): width to crop.
        crop_height (int): height to crop.
        fill_value (float, optional): default values to fill in. Defaults to None.
    
    Returns:
      TopdownMap: cropped top-down map.
    """
    return crop_topdown_map(
      self,
      center = center,
      crop_width = crop_width,
      crop_height = crop_height,
      fill_value = fill_value,
      _validate_args = True
    )

  def merge(
    self,
    *sources: List['TopdownMap'],
  ):
    raise NotImplementedError

# ===== TopdownMap functional API =====

def crop_topdown_map(
  source: TopdownMap,
  center: torch.Tensor,
  crop_width: int,
  crop_height: int,
  fill_value: float = None,
  mode: str = 'nearest',
  _validate_args: bool = True
):
  """Differentiable crop top-down map

  Args:
      source (TopdownMap): top-down map to crop.
      center (torch.Tensor): center coordinates. (b, 2).
      crop_width (int): image width.
      crop_height (int): image height.
      fill_value (float, optional): default values to fill in. Defaults to None.
      mode (str, optional): interpolation mode. Defaults to 'nearest'.

  Returns:
      TopdownMap: cropped top-down map.
  """
  proj = source.proj
  # Convert center to image coordinates
  if _validate_args:
    (center, width_offset, height_offset) = utils.validate_tensors(
      center, proj.width_offset, proj.height_offset,
      same_device = True
    )
    center = center.view(-1, 2)
  # Compute affine grid for cropping image
  grid = utils.generate_crop_grid(
    center = center,
    image_width = source.proj.map_width,
    image_height = source.proj.map_height,
    crop_width = crop_width,
    crop_height = crop_height
  )
  # crop height maps
  height_map = utils.image_sample(
    image = source.height_map,
    grid = grid,
    fill_value = -np.inf,
    mode = mode
  )
  # crop masks
  mask = utils.image_sample(
    image = source.mask,
    grid = grid,
    fill_value = False,
    mode = mode
  )
  # crop top-down map
  topdown_map = height_map
  if not source.is_height_map:
    topdown_map = utils.image_sample(
      image = source.topdown_map,
      grid = grid,
      fill_value = get(fill_value, proj.fill_value),
      mode = mode
    )
  # Update offsets
  if proj.flip_h:
    center[..., 1] = (proj.map_height-1) - center[..., 1]
  width_offset = width_offset + crop_width/2 - center[..., 0]
  height_offset = height_offset + crop_height/2 - center[..., 1]
  map_projector = proj.clone(
    width_offset = width_offset,
    height_offset = height_offset,
    map_width = crop_width,
    map_height = crop_height,
  )
  return TopdownMap(
    topdown_map = topdown_map,
    mask = mask,
    height_map = height_map,
    is_height_map = source.is_height_map,
    map_projector = map_projector
  )

def _flattened_topdown_map(
  source: TopdownMap
):
  assert not source.is_empty
  assert source.proj is not None
  height_map, mask = utils.validate_tensors(
    source.height_map, source.mask, same_device=True
  )
  # Convert height map to point clouds
  point_cloud = source.proj.height_map_to_point_cloud(height_map)
  # Flatten point clouds, masks, topdown maps
  # (b, ..., h, w) -> (b, ..., h*w)
  mask = torch.flatten(mask, -2, -1)
  # (b, ..., h, w, 3) -> (b, ..., h*w, 3)
  points = torch.flatten(point_cloud, -3, -2)
  # Convert from local to global space
  if source.proj.to_global is False:
    points = source.proj.local_to_global_space(points)
  if not source.is_height_map:
    (points, topdown_map) = utils.validate_tensors(
      points, source.topdown_map, same_device=True
    )
    # (b, ..., h, w) -> (b, ..., h*w)
    values = torch.flatten(topdown_map, -2, -1)
  else:
    values = None
  return points, mask, values

def _merge_point_clouds(
  *maps: List[TopdownMap],
  map_projector: MapProjector
):
  # Return flattened point clouds, masks, values
  assert len(maps) > 0
  assert map_projector is not None
  proj = map_projector
  assert proj is not None, "map_projector is not provided"
  points = []
  masks = []
  values = []
  # Convert all topdown map into flat point clouds
  for index, _map in enumerate(maps):
    if not _map.is_empty:
      (_map_points, _map_masks, _map_values) = \
          _flattened_topdown_map(_map)
      points.append(_map_points)
      masks.append(_map_masks)
      if _map_values is not None:
        values.append(_map_values)
      # check if all maps are height maps or not
      assert (len(values) == 0) or (len(values) == len(masks)), \
        (f"All maps must be the same type of maps, but the {index}-th map "
        "is not consistent with others.")
  if len(points) == 0:
    return None, None, None
  # Concat all flattened point clouds
  is_height_map = (len(values) == 0)
  points = utils.validate_tensors(
    *points, same_device=True, same_dtype=torch.float32,
    keep_tuple = True
  )
  masks = utils.validate_tensors(
    *masks, same_device=points[0].device, same_dtype=torch.bool,
    keep_tuple = True
  )
  # (b, ..., h*w, 3)
  points = torch.cat(points, dim=-2)
  # (b, ..., h*w)
  masks = torch.cat(masks, dim=-1)
  if proj.to_global is False:
    points = proj.global_to_local_space(points)
  if not is_height_map:
    values = utils.validate_tensors(
      *values, same_device=points[0].device, same_dtype=torch.float32,
      keep_tuple = True
    )
    # (b, ..., h*w)
    values = torch.cat(values, dim=-1)
  else:
    values = None
  return points, masks, values

def _compute_bounding_box(
  x_coords: torch.Tensor,
  z_coords: torch.Tensor,
  keepdim: bool = False
):
  dims = list(range(1, len(x_coords.shape)))
  min_x = torch.amin(x_coords, dim=dims, keepdim=keepdim)
  max_x = torch.amax(x_coords, dim=dims, keepdim=keepdim)
  min_z = torch.amin(z_coords, dim=dims, keepdim=keepdim)
  max_z = torch.amax(z_coords, dim=dims, keepdim=keepdim)
  return min_x, max_x, min_z, max_z

def _compute_new_shape_and_offsets(
  points: torch.Tensor,
  map_projector: MapProjector
):
  assert points is not None
  assert map_projector is not None
  proj = map_projector
  # Compute bounding box for each batch
  x_coords, z_coords = proj.map_quantize(
    x_coords = points[..., 0],
    z_coords = points[..., 2],
    width_offset = 0.,
    height_offset = 0.,
    flip_h = False
  ) # (b, n)
  min_x, max_x, min_z, max_z = _compute_bounding_box(
    x_coords = x_coords,
    z_coords = z_coords
  ) # (b,)
  # Find the largest bounding box
  padding = 2
  map_width = torch.max((max_x - min_x)).item() + padding
  map_height = torch.max((max_z - min_z)).item() + padding
  # Compute offsets
  center_pos_x = (max_x + min_x) / 2.
  center_pos_z = (max_z + min_z) / 2.
  width_offset = map_width/2. - center_pos_x
  height_offset = map_height/2. - center_pos_z
  return map_width, map_height, width_offset, height_offset

def fuse_topdown_maps(
  *maps: List[TopdownMap],
  map_projector: MapProjector = None,
  fill_value: float = None
) -> TopdownMap:
  """Reproject topdown maps

  Args:
      maps (List[TopdownMap], optional): maps to be reprojected by `map_projector`.
      map_projector (MapProjector, optional): projector to reproject maps. If None,
          use the projector from the first element of `maps`. Defaults to None.
      fill_value (float, optional): default values to fill in the map. Defaults
          to None.

  Returns:
      TopdownMap: reprojected topdown map. If failed to reproject the maps, return
          an empty TopdownMap.
  """
  if len(maps) == 0:
    # Return empty topdown map
    return TopdownMap(map_projector=map_projector)
  # Use the first topdown map's map projector
  if map_projector is None:
    map_projector = maps[0].proj
  proj = map_projector
  # merge point clouds
  points, masks, values = _merge_point_clouds(
    *maps, map_projector=map_projector
  )
  if points is None:
    # Return empty topdown map
    return TopdownMap(map_projector=map_projector)
  is_height_map = (values == None)
  if is_height_map:
    values = points[..., 1]
  # Compute new shape and offsets
  (map_width, map_height, width_offset, height_offset) = \
      _compute_new_shape_and_offsets(
    points = points[masks],
    map_projector = proj
  )
  x_bin, z_bin = proj.map_quantize(
    x_coords = points[..., 0],
    z_coords = points[..., 2],
    width_offset = width_offset,
    height_offset = height_offset,
    map_height = map_height
  )
  coords = torch.stack((z_bin, x_bin), dim=-1)
  canvas = torch.zeros(
    (*coords.shape[:-2], map_height, map_width),
    device = coords.device
  )
  fill_value = get(fill_value, proj.fill_value, -np.inf)
  topdown_map, masks, indices = proj.project(
    coords = coords,
    values = values,
    masks = masks,
    canvas = canvas,
    fill_value = fill_value,
    _validate_args = False
  )
  if is_height_map:
    height_map = topdown_map
  else:
    # project height map
    flat_indices = torch.flatten(indices, -2, -1)
    flat_masks = torch.flatten(masks, -2, -1)
    flat_height_map = _masked_gather(
      values = points[..., 1],
      indices = flat_indices,
      masks = flat_masks,
      fill_value = -np.inf
    )
    height_map = flat_height_map.view(topdown_map.shape)
  map_projector = proj.clone(
    width_offset = width_offset,
    height_offset = height_offset,
    map_width = map_width,
    map_height = map_height,
  )
  # Create top-down map
  topdown_map = TopdownMap(
    topdown_map = topdown_map,
    mask = masks,
    height_map = height_map,
    map_projector = map_projector,
    is_height_map = is_height_map
  )
  return topdown_map

class MapBuilder():
  def __init__(
    self,
    map_projector: MapProjector,
    world_map: TopdownMap = None,
  ):
    self._proj = map_projector
    self._world_map = world_map
    if self._world_map is None:
      # Create empty world map
      self._world_map = TopdownMap(
        map_projector = self.proj.clone()
      )
  
  @property
  def proj(self) -> MapProjector:
    return self._proj

  @property
  def world_map(self) -> TopdownMap:
    """Return the world map"""
    return self._world_map

  def reset(
    self,
    depth_map: np.ndarray = None,
    value_map: np.ndarray = None,
    cam_pose: np.ndarray = None,
    center_mode: CenterMode = CenterMode.none,
    **kwargs
  ):
    """Reset map builder, clear cached world maps

    Args:
        depth_map (np.ndarray, optional): initial 2/3/4D depth map. It should be in
            (h, w), (c, h, w), (b, c, h, w), np.float32. Defaults to None.
        value_map (np.ndarray, optional): optional value map to project. For
            example, one can assign an "one-hot encoded segmentation image" to
            project to a categorical map. If this is None, the default height map
            is projected. Defaults to None. np.float32
        cam_pose (np.ndarray, optional): initial camera pose (3,) or (b, 3). If
            it's None, [0., 0., 0.] is used. Defaults to None.
        center_mode (CenterMode, optional): centering mode can be [None,
            'origin', 'camera']. Defaults to None.
      
    Returns:
        TopdownMap, optional: the projected top-down map if `depth_map` is
            provided.
    """
    # Reset world map, create empty top-down map
    self._world_map = TopdownMap(
      map_projector = self.proj.clone()
    )
    topdown_map = None
    # if `depth_map` and `cam_pose` is provided
    # then plot and merge to world map
    if depth_map is not None:
      topdown_map = self.step(
        depth_map = depth_map,
        value_map = value_map,
        cam_pose = cam_pose,
        center_mode = center_mode,
        **kwargs
      )
    return topdown_map

  def step(
    self,
    depth_map: np.ndarray,
    value_map: np.ndarray = None,
    cam_pose: np.ndarray = None,
    center_mode: CenterMode = CenterMode.none,
    merge: bool = True,
    keep_pose: bool = False,
    **kwargs
  ):
    """Plot the new top-down map, and merge it to the world map.

    Args:
        depth_map (np.ndarray): new 2/3/4D depth map. It should be in (h, w),
            (c, h, w), (b, c, h, w), np.float32. Defaults to None.
        value_map (np.ndarray, optional): optional 2/3/4D value map to project.
            It should have the same b, h, w dims as `depth_map`. If this is None,
            the default height map is projected. Defaults to None.
        cam_pose (np.ndarray, optional): camera pose of the new depth map. This is
            used to match the coordinates of the new top-down map with the world
            map. Defaults to None.
        center_mode (CenterMode, optional): centering mode can be [None,
            'origin', 'camera']. Defaults to None.
        merge (bool, optional): [description]. Defaults to True.
        keep_pose (bool, optional): [description]. Defaults to False.

    Returns:
        TopdownMap: topdown map
    """
    # plot top-down map
    topdown_map = self.plot(
      depth_map = depth_map,
      value_map = value_map,
      cam_pose = cam_pose,
      center_mode = center_mode,
      **kwargs
    )
    # whether to merge the new top-down map to the world map
    if merge:
      self.merge(
        topdown_map,
        keep_pose = keep_pose
      )
    return topdown_map

  def plot(
    self,
    depth_map: np.ndarray,
    value_map: np.ndarray = None,
    cam_pose: np.ndarray = None,
    center_mode: CenterMode = CenterMode.none,
    **kwargs
  ):
    """Plot top-down map TODO

    Args:
        depth_map (np.ndarray): depth map.
        value_map (np.ndarray, optional): value map. Defaults to None.
        cam_pose (np.ndarray, optional): camera pose. Defaults to None.
        center_mode (CenterMode, optional): center mode. Defaults to None.

    Returns:
        TopdownMap: topdown map
    """
    is_height_map = (value_map is None)
    # get default arguments
    cam_pose = get(
      cam_pose,
      self.proj.cam_pose,
      np.array([0., 0., 0.], dtype=np.float32)
    )
    # compute offsets
    width_offset, height_offset = self._compute_offsets(
      cam_pose = cam_pose,
      center_mode = center_mode,
      **kwargs
    )
    # update offsets
    kwargs['width_offset'] = width_offset
    kwargs['height_offset'] = height_offset
    # discard argument
    kwargs.pop('get_height_map', None)
    topdown_map, mask, height_map = self.proj.orth_project(
      depth_map = depth_map,
      value_map = value_map,
      cam_pose = cam_pose,
      get_height_map = True,
      **kwargs
    )
    # Create map projector to store the projection settings
    map_projector = self.proj.clone(
      cam_pose = cam_pose,
      **kwargs
    )
    # Create top-down map
    topdown_map = TopdownMap(
      topdown_map = topdown_map,
      mask = mask,
      height_map = height_map,
      map_projector = map_projector,
      is_height_map = is_height_map
    )
    return topdown_map

  def merge(
    self,
    topdown_map: TopdownMap,
    keep_pose: bool = False,
  ):
    """Merge input top-down map to the world map

    Args:
        topdown_map (TopdownMap): input top-down map.
        keep_pose (bool, optional): keep the world map's pose. If False,
            `topdown_map`'s pose is used. Defaults to False.
    
    Returns:
        TopdownMap: world map.
    """
    if self._world_map is None:
      # Create empty world map
      self._world_map = TopdownMap(
        map_projector = self.proj.clone()
      )
    if keep_pose:
      cam_pose = self._world_map.proj.cam_pose
    else:
      cam_pose = topdown_map.proj.cam_pose
    self._world_map = fuse_topdown_maps(
      self._world_map, topdown_map,
      map_projector = self.proj.clone(
        cam_pose = cam_pose
      )
    )
    return self._world_map

  def _compute_offsets(
    self,
    cam_pose: np.ndarray,
    width_offset: np.ndarray = None,
    height_offset: np.ndarray = None,
    map_res: float = None,
    map_width: int = None,
    map_height: int = None,
    to_global: bool = None,
    center_mode: CenterMode = None,
    **kwargs_
  ):
    width_offset, height_offset = self.proj.compute_center_offsets(
      cam_pose = cam_pose,
      width_offset = width_offset,
      height_offset = height_offset,
      map_res = map_res,
      map_width = map_width,
      map_height = map_height,
      to_global = to_global,
      center_mode = center_mode
    )
    # Returns: torch.Tensor, torch.Tensor
    return width_offset, height_offset
