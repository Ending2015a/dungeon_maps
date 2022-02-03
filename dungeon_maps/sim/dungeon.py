# --- built in ---
import os
import math
# --- 3rd party ---
import numpy as np
import cv2
import moderngl
# --- my module ---
import dungeon_maps as dmap
from dungeon_maps.sim import RESOURCE_ROOT

def read_resource(filename):
  """Read text files"""
  return open(os.path.join(RESOURCE_ROOT, filename), 'r').read()

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

class Dungeon():
  # Action definitions
  NONE = 0
  FORWARD = 1
  LEFT = 2
  RIGHT = 3
  BACKWARD = 4
  STOP = 5

  def __init__(
    self,
    width: int = 800,
    height: int = 600,
    hfov: float = 1.2217304,
    cam_pitch: float = -0.3490659,
    cam_height: float = 0.88,
    # --- simulator macros ---
    min_depth: float = 0.1,
    max_depth: float = 10.0,
    ray_iter: int = 250,
    ray_mult: float = 0.95,
    shadow_iter: int = 48,
    shadow_max_step: float = 0.05,
    maze_scale: float = 2.0,
    wall_height: float = 1.0,
    wall_width: float = 0.25,
    ctx = None
  ):
    """Create dungeon simulator

    Args:
        width (int, optional): image width. Defaults to 800.
        height (int, optional): image height. Defaults to 600.
        hfov (float, optional): horizontal field of view (rad). Defaults to
            1.2217304. ~70 deg
        cam_pitch (float, optional): pitch angle of camera (rad). Defaults to
            -0.3490659. ~-20 deg
        cam_height (float, optional): vertical distance from ground to camera.
            (meter) Defaults to 0.88.
        min_depth (float, optional): minimum depth of depth sensor (meter).
            Defaults to 0.1.
        max_depth (float, optional): maximum depth of depth sensor (meter).
            Defaults to 10.0.
        ray_iter (int, optional): maximum number of ray marching steps. Defaults
            to 250.
        ray_mult (float, optional): size multiplier of each ray marching step.
            Defaults to 0.95.
        shadow_iter (int, optional): maximum number of steps to casting shadow.
            Defaults to 48.
        shadow_max_step (float, optional): maximum distance of steps when casting
            shadow. Defaults to 0.05.
        maze_scale (float, optional): size multiplier of the maze. Defaults to 2.0.
        wall_height (float, optional): height of maze's wall (meter). Defaults to 1.0.
        wall_width (float, optional): width of maze's wall (meter). Defaults to 0.25.
        ctx (optional): moderngl context object. Defaults to None.
    """
    self.width = width
    self.height = height
    self.window_size = (width, height)
    self.min_depth = min_depth
    self.max_depth = max_depth
    self.ray_iter = ray_iter
    self.ray_mult = ray_mult
    self.shadow_iter = shadow_iter
    self.shadow_max_step = shadow_max_step
    self.maze_scale = maze_scale
    self.wall_height = wall_height
    self.wall_width = wall_width
    # Cretae context if it's not given
    if ctx is None:
      ctx = moderngl.create_context(standalone=True, backend='egl')
    self.ctx = ctx
    # Create framebuffer for offline rendering
    self.fbo = ctx.simple_framebuffer((width, height), components=4)
    self.fbo.use()
    self.fbo.clear()
    self.ctx.enable(moderngl.DEPTH_TEST)
    # Create shader program
    self.program = self.ctx.program(
      vertex_shader = read_resource('dungeon.vs'),
      fragment_shader = read_resource('dungeon.fs')
    )
    self.vao = self.create_screen_vao(self.program)
    # Uniforms
    program = self.program
    self.iTime = program.get("iTime", None)
    self.iResolution = program.get("iResolution", None)
    self.iPosition = program.get("iPosition", None)
    self.iTarget = program.get("iTarget", None)
    self.iHFOV = program.get("iHFOV", None)
    self.set_macros()
    # Player initial states
    self.hfov = hfov
    self.init_pos = np.array((0., cam_height, 0.), dtype=np.float64)
    # pitch, yaw, roll (not used)
    self.init_rot = np.array((cam_pitch, math.radians(-135.), 0.), dtype=np.float64)
    self.step_size = 0.5
    # Player current states
    self.cur_pos = self.init_pos.copy()
    self.cur_rot = self.init_rot.copy()
    self.cur_dir = None
    self.cur_steps = 0
    self.delta_time = 0.25
    self.update_player_states()

  def set_macros(self):
    def set_if_not_none(key, value):
      uniform = self.program.get(key, None)
      if uniform is not None:
        uniform.value = value
    set_if_not_none('MIN_DEPTH', self.min_depth)
    set_if_not_none('MAX_DEPTH', self.max_depth)
    set_if_not_none('RAY_ITER', self.ray_iter)
    set_if_not_none('RAY_MULT', self.ray_mult)
    set_if_not_none('SHADOW_ITER', self.shadow_iter)
    set_if_not_none('SHADOW_MAX_STEP', self.shadow_max_step)
    set_if_not_none('MAZE_SCALE', self.maze_scale)
    set_if_not_none('WALL_HEIGHT', self.wall_height)
    set_if_not_none('WALL_WIDTH', self.wall_width)

  def create_screen_vao(self, program):
    vertex_data = np.array([
      # x,    y,   z,    u,   v
      -1.0, -1.0, 0.0,  0.0, 0.0,
      +1.0, -1.0, 0.0,  1.0, 0.0,
      -1.0, +1.0, 0.0,  0.0, 1.0,
      +1.0, +1.0, 0.0,  1.0, 1.0,
    ]).astype(np.float32)
    content = [(
      self.ctx.buffer(vertex_data),
      '3f 2f',
      'in_vert', 'in_uv'
    )]
    idx_data = np.array([
      0, 1, 2,
      1, 2, 3
    ]).astype(np.int32)
    idx_buffer = self.ctx.buffer(idx_data)
    return self.ctx.vertex_array(program, content, idx_buffer)

  def update_player_states(self):
    pitch, yaw, _ = self.cur_rot
    self.cur_dir = np.array((
      math.cos(pitch) * (-math.sin(yaw)),
      math.sin(pitch),
      math.cos(pitch) * math.cos(yaw)
    ), dtype=np.float64)

  def render(self, mode: str='rgb_array'):
    if self.iTime is not None:
      self.iTime.value = self.delta_time * self.cur_steps
    if self.iResolution is not None:
      self.iResolution.value = self.window_size
    if self.iPosition is not None:
      self.iPosition.value = tuple(
        (self.cur_pos * [1., 1., -1.]).tolist() # flip z-axis
      )
    if self.iTarget is not None:
      self.iTarget.value = tuple(
        ((self.cur_pos + self.cur_dir)* [1., 1., -1.]).tolist() # flip z-axis
      )
    if self.iHFOV is not None:
      self.iHFOV.value = self.hfov
    self.fbo.use()
    self.fbo.clear()
    self.vao.render()
    # Get RGB image
    raw_bytes = np.frombuffer(self.fbo.read(components=4), dtype=np.uint8)
    image = raw_bytes.reshape(self.height, self.width, 4)
    image = image[::-1][..., 0:3] # flip horizontal, RGB
    # Get Depth image
    raw_bytes = np.frombuffer(
      self.fbo.read(attachment=-1, dtype='f4'),
      dtype=np.float32
    )
    depth = raw_bytes.reshape(self.height, self.width, 1)
    depth = depth[::-1]
    return {
      'rgb': image,
      'depth': depth
    }

  def calc_related_pose(self):
    pose1 = np.array([
      self.init_pos[0],
      self.init_pos[2],
      self.init_rot[1]],
      dtype=np.float64
    )
    pose2 = np.array([
      self.cur_pos[0],
      self.cur_pos[2],
      self.cur_rot[1]
    ], dtype=np.float64)
    return subtract_pose(pose1, pose2)

  def _get_observations(self):
    observations = self.render()
    pose = self.calc_related_pose()
    observations['pose_gt'] = pose
    return observations

  def reset(self):
    # Reset player states
    self.cur_pos = self.init_pos.copy()
    self.cur_rot = self.init_rot.copy()
    self.cur_dir = None
    self.cur_steps = 0
    self.update_player_states()
    return self._get_observations()

  def step(self, action):
    if action == self.NONE:
      pass
    elif action == self.FORWARD:
      d = np.array((self.cur_dir[0], 0., self.cur_dir[2]), dtype=np.float64)
      d = d / np.linalg.norm(d)
      self.cur_pos += d * self.step_size
    elif action == self.LEFT:
      self.cur_rot[1] += math.radians(30)
    elif action == self.RIGHT:
      self.cur_rot[1] -= math.radians(30)
    elif action == self.BACKWARD:
      d = np.array((self.cur_dir[0], 0., self.cur_dir[2]), dtype=np.float64)
      d = d / np.linalg.norm(d)
      self.cur_pos -= d * self.step_size
    elif action == self.STOP:
      print("Call STOP!")
      pass
    else:
      raise ValueError(f"Undefined action: {action}")
    self.cur_steps += 1
    self.update_player_states()
    return self._get_observations()