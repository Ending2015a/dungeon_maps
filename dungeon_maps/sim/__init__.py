# --- built in ---
import os
import sys
# --- 3rd party ---
try:
  import cv2
  import moderngl
except ImportError as e:
  raise ImportError(
    "Failed to import some libraries. "
    "You may try to install them via 'pip install dungeon_maps[sim]'. "
    f"Message: {str(e)}"
  )

# default: dungeon_maps/sim/
SIM_ROOT = os.path.dirname(os.path.abspath(__file__))
# default: dungeon_maps/
LIB_ROOT = os.path.dirname(SIM_ROOT)
# default: dungeon_maps/sim/data/
RESOURCE_ROOT = os.path.join(SIM_ROOT, 'data')

# --- my module ---
from .dungeon import Dungeon
from .forest import Forest
from .playground import Playground

def make(sim_name, **kwargs):
  if sim_name.lower() == 'dungeon':
    return Dungeon(**kwargs)
  elif sim_name.lower() == 'forest':
    return Forest(**kwargs)
  elif sim_name.lower() == 'playground':
    return Playground(**kwargs)
  else:
    raise NotImplementedError(
      f"Unknown sim_name received: {sim_name}"
    )
