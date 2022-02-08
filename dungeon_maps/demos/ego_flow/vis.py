# --- built in ---
import math
# --- 3rd party ---
import numpy as np
import torch
import dungeon_maps as dmap

def draw_flow(flow):
  flow_scale = 0.2
  ang = torch.atan2(flow[..., 1], flow[..., 0])
  hue = ang / (math.pi * 2.0) + 0.5
  value = torch.norm(flow, dim=-1) * flow_scale
  # Coloring flow
  hsv = torch.stack((hue, torch.ones_like(hue), value), dim=-1)
  h = hsv[..., 0]
  r = torch.abs(h*6-3)-1
  g = 2 - torch.abs(h*6-2)
  b = 2 - torch.abs(h*6-4)
  mv_rgb = torch.clamp(torch.stack((r,g,b), axis=-1), 0.0, 1.0)
  mv_rgb = ((mv_rgb - 1.0) * hsv[..., 1:2] + 1) * hsv[..., 2:3]
  mv_rgb = (torch.clamp(mv_rgb, 0.0, 1.0) * 255).to(dtype=torch.uint8)
  return dmap.utils.to_numpy(mv_rgb.flip(-1))

