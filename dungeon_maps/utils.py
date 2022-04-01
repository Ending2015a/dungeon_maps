# --- built in ---
import enum
from dataclasses import dataclass
from typing import (
  Any,
  Callable,
  Dict,
  Tuple,
  Union,
  Optional
)
# --- 3rd party ---
import numpy as np
import torch
from torch import nn
import torch_scatter
# --- my module ---

# ======== Utils =========

__all__ = [
  'NINF',
  'Reduction',
  'CameraIntrinsics',
  'get_camera_intrinsics',
  # --- PyTorch utils ----
  'to_numpy',
  'to_tensor',
  'to_tensor_like',
  'validate_tensors',
  # --- Transformations ----
  'translate',
  'rotate',
  'ravel_index',
  'scatter_tensor',
  # --- Image utils ----
  'to_4D_image',
  'from_4D_image',
  'generate_image_coords',
  'generate_crop_grid',
  'image_sample'
]

# === CONSTANTS ===

NINF = -np.inf
ANGLE_EPS = 0.001

# === TYPING ===
Float3D = Tuple[float, float, float]

@enum.unique
class Reduction(str, enum.Enum):
  max = 'max'
  min = 'min'
  sum = 'sum'
  mean = 'mean'
  prod = 'prod'

  @classmethod
  def _missing_(cls, value):
    """In default we use Reduction.max
    for example:
      assert Reduction(None) == Reduction.max
    """
    if value is None:
      return cls.max

# Reduction/scatter mapping
_ScatterMethods: Dict[Reduction, Callable] = {
  Reduction.max: torch_scatter.scatter_max,
  Reduction.min: torch_scatter.scatter_min,
  Reduction.sum: torch_scatter.scatter_add,
  Reduction.mean: torch_scatter.scatter_mean,
  Reduction.prod: torch_scatter.scatter_mul
}


@dataclass
class CameraIntrinsics:
  """Camera intrinsic parameters
  
  Properties:
    cx: principle point a.k.a. optical center
    cy: principle point a.k.a. optical center
    fx: focal length
    fy: focal length
  """
  cx: float
  cy: float
  fx: float
  fy: float

def get_camera_intrinsics(
  width: float,
  height: float,
  hfov: float,
  vfov: Optional[float] = None
) -> CameraIntrinsics:
  """Return camera's intrinsic parameters
  
  Args:
    width (float): image width (px)
    height (float): image height (px)
    hfov (float): horizontal field of view (rad)
    vfov (float, optional): vertical field of view (red). Automaically
      inferred from `hfov` if set this to None.

  Returns:
    CameraIntrinsics: camera intrnsic parameters
  """
  cx = width / 2.
  cy = height / 2.
  fx = cx / np.tan(hfov / 2.)
  fy = cy / np.tan(vfov / 2.) if vfov is not None else fx
  return CameraIntrinsics(cx=cx, cy=cy, fx=fx, fy=fy)

# ======== PyTorch utils =======
def to_numpy(
  inputs: Any,
  dtype: Optional[np.dtype] = None
) -> np.ndarray:
  """Convert inputs to numpy array

  Args:
    inputs (Any): input tensor
    dtype (np.dtype, optional): numpy data type. Defaults to None.

  Returns:
    np.ndarray: numpy array
  """
  if torch.is_tensor(inputs):
    t = inputs.detach().cpu().numpy()
  else:
    t = np.asarray(inputs)
  dtype = dtype or t.dtype
  return t.astype(dtype=dtype)

def to_tensor(
  inputs: Any,
  dtype: Optional[torch.dtype] = None,
  device: Optional[torch.device] = None,
  **kwargs
) -> torch.Tensor:
  """Convert `inputs` to torch.Tensor with the specified `dtype`
  and place on the specified `device`

  Args:
    inputs (Any): input tensor.
    dtype (torch.dtype, optional): data type. Defaults to None.
    device (torch.device, optional): device. Defaults to None.

  Returns:
    torch.Tensor: tensor
  """
  if torch.is_tensor(inputs):
    t = inputs
  elif isinstance(inputs, np.ndarray):
    t = torch.from_numpy(inputs)
  else:
    t = torch.tensor(inputs, dtype=dtype)
  return t.to(device=device, dtype=dtype, **kwargs)

def to_tensor_like(
  inputs: Any,
  tensor: torch.Tensor
) -> torch.Tensor:
  """Convert `inputs` to torch.Tensor with the same dtype and
  device as `tensor`

  Args:
    inputs (Any): input tensor.
    tensor (torch.Tensor): target tensor.
  
  Returns:
    torch.Tensor: tensor.
  """
  assert torch.is_tensor(tensor), \
      f"`tensor` must be a torch.Tensor, got {type(tensor)}"
  return to_tensor(inputs, dtype=tensor.dtype, device=tensor.device)

def validate_tensors(
  *args: Any,
  same_device: Optional[Union[bool, torch.device]] = None,
  same_dtype: Optional[Union[bool, torch.dtype]] = None,
  keep_tuple: bool = False
) -> Union[Tuple[torch.Tensor, ...], torch.Tensor]:
  """Validate tensors, convert all args into torch.tensor

  Args:
    same_device (bool, torch.device, optional): device that all tensors being
      placed on. If it is True, all the tensors are placed on the same
      device as the first torch.tensor. Defaults to None.
    same_dtype (bool, torch.dtype, optional): tensor types. If it is True,
      all the tensors are converted to the same data type as the first
      torch.tensor. Defaults to None.
    keep_tuple (bool, optional): if False, the returns are automatically
      unpacked when only one `args` is provided. Otherwise, return a tuple.

  Returns:
    Union[Tuple[torch.Tensor], torch.tensor]: tuple of tensors or a tensor if
      only one `args` is provided and `keep_tuple` is set to False.
  """
  if len(args) == 0:
    return
  # Convert the first args to torch.tensor
  first_tensor = to_tensor(args[0])
  # Get device from the first tensor
  if isinstance(same_device, bool):
    if same_device:
      same_device = first_tensor.device
    else:
      same_device = None
  # Get dtype from the first tensor
  if isinstance(same_dtype, bool):
    if same_dtype:
      same_dtype = first_tensor.device
    else:
      same_dtype = None
  tensors = []
  for arg in args:
    tensors.append(
      to_tensor(arg, device=same_device, dtype=same_dtype)
    )
  if len(tensors) == 1 and not keep_tuple:
    return tensors[0]
  return tuple(tensors)

def translate(
  points: torch.Tensor,
  offsets: torch.Tensor
) -> torch.Tensor:
  """Translating 3D points (move along XYZ)

  Note that the coordinate system follows the rule
    X: right
    Y: up
    Z: forward

  Args:
    points (torch.Tensor): points to translate, shape (b, ..., 3). torch.float32
    offsets (torch.Tensor): XYZ offsets (b, 3). torch.float32
  
  Returns:
    torch.Tensor: translated points
  """
  points, offsets = validate_tensors(
    points, offsets,
    same_device = True,
    same_dtype = torch.float32
  )
  offsets = offsets.view(-1, 1, 3) # (b, 1, 3)
  batch = points.shape[0]
  # apply translation
  points = torch.reshape(
    points.view(batch, -1, 3) + offsets,
    points.shape
  )
  return points

def rotate(
  points: torch.Tensor,
  axis: torch.Tensor,
  angle: torch.Tensor,
  angle_eps: float = ANGLE_EPS
) -> torch.Tensor:
  """Rotating 3D points along `axis` with Rodringues' Rotation formula:
    R = I + S * sin(angle) + S^2 * (1-cos(angle))
  where
    R: rotation metrics
    S: [
            0, -axis_x,  axis_y,
       axis_z,       0, -axis_x,
      -axis_y,  axis_z,       0
    ]
    S^2: matmul(S, S)
  
  Note that the coordinate system follows the rule
    X: right
    Y: up
    Z: forward

  Args:
    points (torch.Tensor): points to rotate, in shape (b, ..., 3). torch.float32
    axis (torch.Tensor): axis the points rotated along, in shape (b, 3).
      torch.float32
    angle (torch.Tensor): rotated angle in radian with shape (b,) or (b, 1).
      torch.float32
    angle_eps (float, optional): angle precision. `abs(angle)` smaller than this
      value is clamped to zero. Defaults to 0.001.

  Returns:
    torch.tensor: rotated points
  """
  points, axis, angle = validate_tensors(
    points, axis, angle,
    same_device = True,
    same_dtype = torch.float32
  )
  device = points.device
  batch = points.shape[0]
  # Shaping axis and angle to match the batch shape
  axis = axis.view(-1, 3)
  angle = angle.view(-1, 1)
  # Creating batch rotation matrics from axis
  #  Normalize axis
  ax = axis / torch.linalg.norm(axis, dim=-1, keepdim=True) # (b, 3)
  ax_x = ax[..., 0] # (b,)
  ax_y = ax[..., 1]
  ax_z = ax[..., 2]
  zeros = torch.zeros((batch,), dtype=torch.float32, device=device)
  S_flat = torch.stack((
    zeros, -ax_z, ax_y,
    ax_z, zeros, -ax_x,
    -ax_y, ax_x, zeros
  ), dim=-1) # (b, 9)
  S = S_flat.view(-1, 3, 3)
  S2 = torch.einsum('bij,bjk->bik', S, S) # matmul(S, S)
  S2_flat = S2.view(-1, 9)
  eye_flat = torch.eye(3, device=device).view(-1, 9) # flat eye matrices
  # Clamp angle if it is near to 0.0
  # torch bug: condition y must be a tensor with type float32
  zr = torch.tensor(0.0, device=device)
  angle = torch.where(torch.abs(angle) > angle_eps, angle, zr)
  # Create rotation matrices
  R_flat = eye_flat + torch.sin(angle) * S_flat + (1-torch.cos(angle)) * S2_flat
  R = R_flat.view(-1, 3, 3) # (b, 3, 3)
  # apply rotation
  points = torch.einsum('bji,b...j->b...i', R, points)
  return points

def ravel_index(
  index: torch.Tensor,
  shape: torch.Size,
  keepdim: bool = False
) -> torch.Tensor:
  """Ravel multi-dimensional indices to 1D index
  similar to np.ravel_multi_index

  For example:
  ```python
  indices = [[3, 2, 3], [0, 2, 1]]
  shape = (6, 5, 4)
  print(ravel_index(indices, shape))
  ```
  will output:
  ```
  tensor([71,  9])
  # 71 = 3 * (5*4) + 2 * (4) + 3
  # 9 = 0 * (5*4) + 2 * (4) + 1
  ```

  Args:
    index (torch.Tensor): indices in reversed order dn, ..., d1,
      with shape (..., n). torch.int64
    shape (torch.Size): shape of each dimension dn, ..., d1
    keepdim (bool, optional): keep dimensions. Defaults to False.

  Returns:
    torch.Tensor: Raveled indices in shape (...,), if `keepdim` is False,
      otherwise in shape (..., 1).
  """
  index, shape = validate_tensors(
    index, (1,)+shape[::-1], # [1, d1, ..., dn]
    same_device = True,
    same_dtype = torch.int64
  )
  shape = torch.cumprod(shape, dim=0)[:-1].flip(0) # [d1*...*dn-1, ..., d1*d2, d1, 1]
  index = (index * shape).sum(dim=-1, keepdim=keepdim) # (..., 1) or (...,)
  return index

def _get_scatter_method(
  reduction: Union[str, Reduction]
) -> Callable:
  """Get scatter method by reduction type. Note that
  if reduction is set to None, Reduction.max is used in default.

  Args:
    reduction (Union[str, Reduction]): reduction enum.

  Returns:
    Callable: corresponding scatter method
  """
  reduction = Reduction(reduction)
  if reduction not in _ScatterMethods:
    raise ValueError(f"Invalid reduction method: {reduction}")
  return _ScatterMethods[reduction]

def scatter_tensor(
  canvas: torch.Tensor,
  indices: torch.Tensor,
  values: torch.Tensor,
  masks: Optional[torch.Tensor] = None,
  fill_value: Optional[float] = None,
  reduction: Optional[Union[str, Reduction]] = None,
  _validate_args: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
  """Scattering values over an `n`-dimensional canvas

  In the case of projecting values to an image-type canvas (`n`=2), i.e. projecting
  height values to a top-down height map, the shape of the canvas is (b..., d1, d2)
  or we say (b..., h, w), where `b...` is the batch dimensions, `d1`, `d2` the data
  dimensions. `values`, in this case, is the height values and has the shape
  (b..., N) for a point cloud that containing `N` points. `indices` is the
  coordinates of the top-down map that `N` points being projected to and has the
  shape (b..., N, 2). The last dimension of `indices` stores the indices of each
  data dimensions, i.e. [d1, d2] or [h, w]. For `n`-dimensional canvas
  (b..., d1, ..., dn), it stores [d1, ..., dn].

  Args:
    canvas (torch.Tensor): canvas in shape (b..., d1, ..., dn). torch.float32
    indices (torch.Tensor): [d1, ..., dn] coordinates in shape (b..., N, n).
      torch.float32
    values (torch.Tensor): values to scatter in shape (b..., N). torch.float32
    masks (torch.Tensor, optional): boolean masks. True for valid values,
      False for invalid values. shape (b..., N). torch.bool. Defaults to None.
    fill_value (float, optional): default values to fill in with. Defaults
      to None.
    reduction (Union[str, Reduction], optional): method to reduce the values
      scattered into the same cells. Defaults to Reduction.max.

  Returns:
    torch.Tensor: resuling canvas, (b..., d1, ..., dn). torch.float32
    torch.Tensor: binary masks. False for unchanged values, True for changed
      values. shape (b..., d1, ..., dn) torch.bool
  """
  if _validate_args:
    # Create default masks
    if masks is None:
      masks = torch.ones(values.shape)
    (canvas, indices, values, masks) = validate_tensors(
      canvas, indices, values, masks, same_device=True
    )
    # Ensure dtypes
    canvas = canvas.to(dtype=torch.float32)
    indices = indices.to(dtype=torch.int64)
    values = values.to(dtype=torch.float32)
    masks = masks.to(dtype=torch.bool)
  assert masks is not None
  # Get dimensions
  n = indices.shape[-1]
  N = values.shape[-1]
  assert len(canvas.shape) > n, \
    f"The rank of `canvas` must be greater than {n}, got {len(canvas.shape)}"
  dn_d1 = canvas.shape[-n:] # (d1, ..., dn)
  batch_dims = canvas.shape[:-n] # (b...,)
  # Mark the out-of-bound points as invalid
  valid_areas = [masks]
  for i in reversed(range(n)):
    di = indices[..., i] # (b..., N)
    valid_areas.extend((di < dn_d1[i], di >= 0))
  valid_areas = torch.broadcast_tensors(*valid_areas)
  masks = torch.stack(valid_areas, dim=0).all(dim=0) # (b..., N)
  # Set dummy indices for invalid points (0, ..., 0, -1)
  indices[..., :][~masks] = 0
  indices[..., -1][~masks] = -1
  # Flatten all things to 1D
  flat_canvas = canvas.view(*batch_dims, -1) # (b..., d1*...*dn)
  # convert nd indices to 1d indices (b..., N)
  flat_indices = ravel_index(indices, dn_d1)
  flat_values = values # (b..., N)
  # Create dummy channel to store invalid values
  dummy_channel = torch.zeros_like(flat_canvas[..., 0:1]) # (b..., 1)
  dummy_shift = 1
  # Shifting dummy index from (0, ..., 0, -1) to (0, ..., 0, 0)
  # (b..., 1 + d1*...*dn)
  flat_canvas = torch.cat((dummy_channel, flat_canvas), dim=-1)
  flat_indices = flat_indices + dummy_shift
  # Get reduction method
  reduction = Reduction(reduction)
  scatter_method = _get_scatter_method(reduction)
  if fill_value is not None:
    flat_canvas.fill_(fill_value)
  flat_canvas_copy = flat_canvas.clone()
  scatter_method(
    flat_values, flat_indices, dim=-1, out=flat_canvas
  )
  # slice off dummy channel
  flat_canvas = flat_canvas[..., 1:]
  canvas = flat_canvas.view(canvas.shape)
  flat_canvas_copy = flat_canvas_copy[..., 1:]
  canvas_copy = flat_canvas_copy.view(canvas.shape)
  # here we use subtraction to detect whether the canvas values are
  # changed after scattering.
  # note that if `fill_value` is set to inf or -inf. It has the following
  # rules, where x denotes an arbitrary floating value:
  #   inf - inf = nan  => non-changed
  #   inf - x = inf    => changed
  masks = torch.abs(canvas - canvas_copy)
  masks = torch.nan_to_num(masks, nan=0.0, posinf=1.0, neginf=1.0)
  masks = ~(masks == 0.0).to(dtype=torch.bool)
  return canvas, masks

def to_4D_image(image: torch.Tensor) -> torch.Tensor:
  """Convert `image` to 4D tensors (b, c, h, w)

  Args:
    image (torch.Tensor): 2/3/4D image tensor
      2D: (h, w)
      3D: (c, h, w)
      4D: (b, c, h, w)
  
  Returns:
    torch.Tensor: 4D image tensor
  """
  ndims = len(image.shape)
  assert ndims in [2, 3, 4], \
    f"only supports 2/3/4D images while {ndims}-D are given."
  if ndims == 2:
    return image[None, None, :, :] # b, c
  elif ndims == 3:
    return image[None, :, :, :] # b
  else:
    return image

def from_4D_image(image: torch.Tensor, ndims: int) -> torch.Tensor:
  """Convert `image` to `ndims`-D tensors

  Args:
    image (torch.Tensor): 4D image tensors in shape (b, c, h, w).
    ndims (int): the original rank of the image.

  Returns:
    torch.Tensor: `ndims`-D tensors
  """
  _ndims = len(image.shape)
  assert _ndims == 4, f"`image` must be a 4D tensor, while {_ndims}-D are given."
  if ndims == 2:
    return image[0, 0, :, :] # -b, -c
  elif ndims == 3:
    return image[0, :, :, :] # -b
  else:
    return image

def generate_image_coords(
  image_shape: torch.Size,
  dtype: Optional[torch.dtype] = None,
  device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
  """Generate image horizontal, vertical coordinates

  Args:
    image_shape (torch.Size): shape of image. Expected 2/3/4D or higher
    dimensions, where the last two dimensions are (h, w).
    flip_h (bool, optional): whether to flip horizontal axis. Defaults to True.
    dtype (torch.dtype, optional): data type, defaults to torch.float32.
      Defaults to None.
    device (torch.device, optional): torch device. Defaults to None.

  Returns:
    torch.tensor: horizontal coordinates in `image_shape`.
    torhc.tensor: vertical coordinates in `image_shape`.
  """
  dtype = dtype or torch.float32
  ndims = len(image_shape)
  if ndims < 2:
    raise ValueError("rank of `image_shape` must be at east 2D, "
      f"got {ndims}")
  h = image_shape[-2]
  w = image_shape[-1]
  # Generate x, y coordinates
  x = torch.arange(w, dtype=dtype, device=device)
  y = torch.arange(h, dtype=dtype, device=device)
  # Expand dims to match depth map
  x = x.view((1,)*(ndims-2) + (1, -1)) # (..., 1, w)
  y = y.view((1,)*(ndims-2) + (-1, 1)) # (..., h, 1)
  x = torch.broadcast_to(x, image_shape) # (..., h, w)
  y = torch.broadcast_to(y, image_shape)
  return x, y

def generate_crop_grid(
  center: torch.Tensor,
  image_width: int,
  image_height: int,
  crop_width: int,
  crop_height: int,
  device: Optional[torch.device] = None
) -> torch.Tensor:
  """Generate affine grid for cropping.

  Args:
    center (torch.Tensor): cropping center coordinates (b, 2)
    image_width (int): image width.
    image_height (int): image height.
    crop_width (int): width to crop.
    crop_height (int): height to crop.
    device (torch.device, optional): torch device. Defaults to None.

  Returns:
    torch.Tensor: crop grid (b, crop_height, crop_width, 2). torch.float32
  """
  center = to_tensor(center, device=device)
  center = center.view(-1, 2).to(dtype=torch.float32)
  batch = center.shape[0]
  h, w = image_height, image_width
  # padding left right top bottom
  center = center + 1
  h = h + 2
  w = w + 2
  x, y = generate_image_coords(
    (batch, crop_height, crop_width),
    dtype = torch.float32,
    device = center.device
  )
  ndims = len(x.shape)
  center_x = (center[..., 0] - w/2.).view((-1,)+(1,)*(ndims-1)) # (b, ...)
  center_y = (center[..., 1] - h/2.).view((-1,)+(1,)*(ndims-1)) # (b, ...)
  x = (x - crop_width/2. + center_x) / (w/2.)
  y = (y - crop_height/2. + center_y) / (h/2.)
  grid = torch.stack((x, y), dim=-1)
  return grid

def image_sample(
  image: torch.Tensor,
  grid: torch.Tensor,
  fill_value: Optional[float] = None,
  mode: str = 'nearest',
  _validate_args: bool = True
) -> torch.Tensor:
  """Sample image by affine grid

  Args:
    image (torch.Tensor): image tensor (b, c, h, w). torch.float32
    grid (torch.Tensor): affine grid (b, mh, mw, 2), usually generated from
      `generate_crop_grid`. torch.float32
    fill_value (Any, optional): default values to fill in. Defaults to None.
    mode (str, optional): sampling method can be either 'nearest', 'bilinear'
      or 'bicubic'. See [1]. Defaults to 'nearest'.

  [1] https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html

  Returns:
    torch.Tensor: sampled image tensor (b, c, mh, mw), torch.float32.
  """
  if _validate_args:
    image, grid = validate_tensors(image, grid, same_device=True)
    # Ensure tensor shapes
    image = to_4D_image(image)
  padding_mode = 'border'
  if fill_value is None:
    fill_value = 0.0 # pad zeros
    padding_mode = 'zeros'
  # pad default values
  pad = [1, 1, 1, 1]
  image = nn.functional.pad(image, pad, mode='constant', value=fill_value)
  orig_dtype = image.dtype
  # grid_sample restrict the image type to be same as grid
  image = image.to(dtype=grid.dtype)
  image = nn.functional.grid_sample(image, grid, mode=mode,
      padding_mode=padding_mode, align_corners=True)
  image = image.to(dtype=orig_dtype)
  return image

# ======= Deprecated utilities ========
# these utilities will be removed soon

def gather_nd(
  params: torch.Tensor,
  indices: np.ndarray,
  batch_dims: int = 0,
  _validate_args: bool = True
):
  """torch version of tf.gather_nd
  see: https://discuss.pytorch.org/t/how-to-do-the-tf-gather-nd-in-pytorch/6445/36

  Args:
    params (torch.Tensor): shape (b..., d1, ..., dn), where `b...` is arbitrary
      number of batch dimensions. `d1`~`dn` is the `n` data dimensions.
    indices (np.ndarray): stores the indices of data to pick. shape (b..., m),
      where `m` is the rank of data dimensions to pick, m <= n. np.int64
    batch_dims (int, optional): number of leading batch dimensions of `params`,
      i.e. len(b...). Defaults to 0.

  Returns:
    torch.Tensor: shape (b..., d(n-m), ..., dn)
  """
  if _validate_args:
    params = validate_tensors(params, same_device=True)
    indices = to_numpy(indices, dtype=np.int64)
  if batch_dims == 0:
    orig_shape = list(indices.shape)
    b_dims = int(np.prod(orig_shape[:-1]))
    m = orig_shape[-1]
    n = len(params.shape)
    if m <= n:
      out_shape = orig_shape[:-1] + list(params.shape[m:])
    else:
      raise ValueError(
        "The last dimension of indices must less or equal to the rank of params. "
        f"Got indices: {indices.shape}, params: {params.shape}. {m} > {n}"
      )
    indices = indices.reshape((b_dims, m)).transpose().tolist()
    return params[indices].reshape(out_shape).contiguous()
  else:
    batch_shape = params.shape[:batch_dims] # b...
    m = indices.shape[-1]
    if batch_shape != indices.shape[:batch_dims]:
      raise ValueError(
        "The leading batch dimensions of `params` and `indices` does not match."
      )
    out_shape = indices.shape[:-1] + params.shape[batch_dims+m:]
    b_dims = int(np.prod(batch_shape))
    if b_dims != 1:
      params = params.reshape(b_dims, *params.shape[batch_dims:])
      indices = indices.reshape(b_dims, *indices.shape[batch_dims:])
    out = []
    for i in range(b_dims):
      out.append(gather_nd(params[i], indices[i], batch_dims=0))
    out = torch.stack(out, dim=0)
    return out.reshape(out_shape).contiguous()

def advance_indexing(inputs, *indices):
  batch = inputs.shape[0]
  ndims = len(inputs.shape)
  ind_ndims = len(indices)
  assert ind_ndims < ndims
  batch_inds = torch.arange(batch, dtype=torch.int64)
  batch_inds = batch_inds.view((batch,) + (1,)*len(indices))
  batch_inds = batch_inds.tile((1,) + inputs.shape[1:ind_ndims+1])
  batch_inds = torch.stack((batch_inds,) + indices, dim=-1)
  return gather_nd(inputs, batch_inds)

def remap(
  image: torch.Tensor,
  grid: torch.Tensor,
  method: str = 'bilinear',
  _validate_args: bool = True
):
  """Re-sample image tensors

  Args:
    image (torch.Tensor): 2/3/4D image tensoes to re-sample. shape (b, c, h, w).
    grid (torch.Tensor): affine grid (b, h, w, 2). torch.float32
    method (str, optional): sampling method. Must be one of ['bilinear',
      'nearest']. Defaults to 'bilinear'.

  Returns:
    torch.Tensor: Re-sampled image tensors.
  """
  if _validate_args:
    (image, grid) = validate_tensors(image, grid, same_device=True)
    # Ensure 4D tensor (b, c, h, w)
    orig_ndims = len(image.shape)
    image = to_4D_image(image)
    if len(grid.shape) < 4:
      grid = grid.view(-1, *grid.shape) # (b, h, w, 2)
  orig_dtype = image.dtype
  image = image.to(dtype=torch.float32)
  h = image.shape[-2]
  w = image.shape[-1]
  x = grid[..., 0] # (b, h, w)
  y = grid[..., 1] # (b, h, w)
  x0 = x.floor().clamp(0, w-1)
  x1 = (x0 + 1).clamp(0, w-1)
  y0 = y.floor().clamp(0, h-1)
  y1 = (y0 + 1).clamp(0, h-1)
  # Sample pixel values
  x0i = x0.long()
  x1i = x1.long()
  y0i = y0.long()
  y1i = y1.long()
  i1 = advance_indexing(image, x0i, y0i) #TODO
  i2 = advance_indexing(image, x0i, y1i)
  i3 = advance_indexing(image, x1i, y0i)
  i4 = advance_indexing(image, x1i, y1i)
  # Re-sampling
  if method == 'nearest':
    xp = (x1-x > x-x0).float()
    yp = (y1-y > y-y0).float()
    xf = 1 - (x1 == x0).float()
    yf = 1 - (y1 == y0).float()
    w1 = xp * yp
    w2 = xp * (1-yp)
    w3 = (1-xp) * yp
    w4 = (1-xp) * (1-yp)
    mask = xf * yf
  elif method == 'bilinear':
    w1 = (x1-x) * (y1-y)
    w2 = (x1-x) * (y-y0)
    w3 = (x-x0) * (y1-y)
    w4 = (x-x0) * (y-y0)
    mask = torch.ones_like(w1)
  else:
    raise NotImplementedError(f"Sampling method not implemented: {method}")
  mask = mask.unsqueeze(dim=-1)
  w1 = w1.unsqueeze(dim=-1)
  w2 = w2.unsqueeze(dim=-1)
  w3 = w3.unsqueeze(dim=-1)
  w4 = w4.unsqueeze(dim=-1)
  new_image = (w1*i1 + w2*i2 + w3*i3 + w4*i4) * mask
  new_image = from_4D_image(new_image, orig_ndims)
  new_image = new_image.to(dtpye=orig_dtype)
  return new_image