"""----------------------------------------------------------------------------
Author: N. Zaunders. 2026
The University of Queensland

3D real-time renderer system for physics simulations in Python using PyGame's 
rendering engine.
----------------------------------------------------------------------------"""

import numpy as np
import pygame as pg
from pygame import gfxdraw as gfx

rng = np.random.default_rng()

import warnings
warnings.filterwarnings('ignore') 

#%% Initialisation

WINDOW_SIZE     = (1600, 900)
FRAMERATE       = 500
window          = pg.display.set_mode(WINDOW_SIZE)                              
clock           = pg.time.Clock()

# Set refresh rate to 60 Hz
clock.tick(FRAMERATE)

OFFSET          = (WINDOW_SIZE[0] / 2, WINDOW_SIZE[1] / 2)