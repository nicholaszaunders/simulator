"""----------------------------------------------------------------------------
Author: N. Zaunders. 2026
The University of Queensland

3D real-time renderer system for physics simulations in Python using PyGame's 
rendering engine.

TO DO:
    - Fix slight warping on axes when viewing at 45 degree angles
    - Speedup evaluation
    - Add method for showing / not showing axes and origin points
    - Streamline rendering by introducing a buffer for rendered points
    - Add camera position globe and text for reference
    - Generalise camera to point in arbitrary directions
----------------------------------------------------------------------------"""

import numpy as np
import pygame as pg
from pygame import gfxdraw as gfx

rng = np.random.default_rng()

import warnings
warnings.filterwarnings('ignore') 

#%% Initialisation

# Initialise renderer window and clock
WINDOW_SIZE     = (1600, 900)
FRAMERATE       = 60
window          = pg.display.set_mode(WINDOW_SIZE)                              
clock           = pg.time.Clock()

# Set refresh rate to 60 Hz
clock.tick(FRAMERATE)

# Set origin point of render window
OFFSET          = (WINDOW_SIZE[0] / 2, WINDOW_SIZE[1] / 2)

#%% Camera functions

# Converts spherical point to Cartesian point.
def SPHERE_TO_CART(
    point
):
    (r, theta, phi) = point
    x       = r * np.sin(theta) * np.cos(phi)
    y       = r * np.sin(theta) * np.sin(phi)
    z       = r * np.cos(theta)
    return np.array([x, y, z])

# Converts Cartesian point to spherical point.
def CART_TO_SPHERE(
    point
):
    (x, y, z) = point
    r       = np.sqrt(x**2 + y**2 + z**2)
    theta   = np.arctan2(np.sqrt(x**2 + y**2), z)
    phi     = np.arctan2(y, x)
    return np.array([r, theta, phi])

# Takes a Cartesian point 'point' and rotates by the spherical zenith angle 
# theta and azimuthal angle phi. Outputs rotated point in Cartesian coords.
def SPHERICAL_ROTATE_CART_POINT(
    point,
    theta,
    phi
):
    point_spherical = CART_TO_SPHERE(point)
    point_spherical[1] += theta
    point_spherical[2] += phi
    return SPHERE_TO_CART(point_spherical)

# Defines the 3D Rodrigues transform; takes initial vector 'vector' and returns
# the vector given by rotating vector by theta about the unit vector k.
def rodrigues_transform(
    k, 
    theta, 
    vector
):
    return vector * np.cos(theta) + np.cross(k, vector) * np.sin(theta) + k * np.dot(k, vector) * (1 - np.cos(theta))

# Takes in the position of the camera vector and returns the Rodrigues unit
# vector and angle theta required to rotate it to the Cartesian position (0,0,1).
def camera_return_rotation(
    camera_vector
):
    target_vector = np.array([0, 0, 1]) * np.linalg.norm(camera_vector)
    
    # If the camera vector is in the right place, break and return.
    if np.array_equal(camera_vector, target_vector):
        return (np.array([0, 0, 0]), 0)
    
    # Using Rodrigues' 3d rotation formula
    k       = np.cross(camera_vector, target_vector) / np.linalg.norm(np.cross(camera_vector, target_vector))
    theta   = np.arccos(np.dot(camera_vector, target_vector) / (np.linalg.norm(camera_vector) * np.linalg.norm(target_vector)))
    
    return k, theta   

# Returns 2D projected point for input 3D Cartesian point. 
def PERSPECTIVE_PROJECTION(
    point_vector,
    focal_distance,
    camera_vector,
    OFFSET      
):
    (k, theta)              = camera_return_rotation(camera_vector)
    camera_vector_rotated   = rodrigues_transform(k, theta, camera_vector)
    point_vector_rotated    = rodrigues_transform(k, theta, point_vector)
    
    projected_point =  (focal_distance * (point_vector_rotated[0]-camera_vector_rotated[0]) / (point_vector_rotated[2]-camera_vector_rotated[2]),
                        focal_distance * (point_vector_rotated[1]-camera_vector_rotated[1]) / (point_vector_rotated[2]-camera_vector_rotated[2]))
    
    camera_twist = - np.pi/2 - CART_TO_SPHERE(camera_vector)[2]
    rotation_matrix_2d = np.array([[np.cos(camera_twist), -np.sin(camera_twist)],
                                   [np.sin(camera_twist),  np.cos(camera_twist)]])
    
    return rotation_matrix_2d @ projected_point + np.array([OFFSET[0], OFFSET[1]])

#%% Main loop initialisation

# Set initial camera position
camera_radius       = 750 #1000
camera_theta        = 1.4764285714285723
camera_phi          = -0.415673265173984
field_of_vision     = 20

# Buffer array for mouse position delta
mouse_pos_buffer    = np.zeros((2,2), dtype = float)
mouse_sens          = 3

# Sensitivity of camera zoom scroll
mouse_scroll_sens   = 50

# Set particle initial speed
particle_speed      = 10.0

# Set time dilation factor for evolution of chaotic particle
time_dilation       = 1

# Set particle tail time
tail_time           = 0.2

# Int flag to set whether axes and context cube appears (initial value False)
axis_cubes_visible  = False

# Define a particle class governed by a Lorentz attractor.
class particle_chaotic:
    def __init__(self, initial_position, parameters, tail_time):
        self.pos_arr       = np.zeros((int(tail_time * FRAMERATE), 3))
        self.pos_arr[0, :] = initial_position
        self.params        = parameters #[σ, ρ, β]
        self.colour        = np.array([np.random.rand() * 255, np.random.rand() * 255, np.random.rand() * 255])
        
    # Gives the velocity values of the particle based on the Lorentz attractor ODE.
    def particle_ode(self, pos):
        dx = (self.params[0] * (pos[1] - pos[0]))
        dy = (pos[0] * (self.params[1] - pos[2]) - pos[1])
        dz = (pos[0] * pos[1] - self.params[2] * pos[2])
        return np.array([dx, dy, dz])
        
    def tick_forward(self, tickrate):
        dt = 1/tickrate
        k1 = self.particle_ode(self.pos_arr[0, :])
        k2 = self.particle_ode(self.pos_arr[0, :] + k1 * dt / 2)
        k3 = self.particle_ode(self.pos_arr[0, :] + k2 * dt / 2)
        k4 = self.particle_ode(self.pos_arr[0, :] + k3 * dt)
        
        newpos = self.pos_arr[0, :] + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
        
        self.pos_arr[1:, :] = self.pos_arr[0:-1, :]
        self.pos_arr[0, :]  = newpos
    
# Initialise an empty particle array
PARTICLE_ARRAY = []



# Define a new class which takes in static objects and renders them in a
# sequential pipeline.
# Essentially should be a list of `objects' defined by vertices and edge
# connections joining pairs of vertices.

# Initialise an empty queue for static objects
STATIC_OBJECT_QUEUE = []

# Define a generic static object with arbitrary vertices and edges.
# 'vertices' is a 3xN array defining N points in 3D space; each column is x,y,z.
# 'edges' is a 2xM array where each column holds the indices of the two points
# (relative to the array vertices) that should be connected by a line.
# Also takes an optional name 'ident'.
class STATIC_OBJECT:
    def __init__(self, vert, edge, iden):
        self.vert = vert
        self.edge = edge
        self.iden = iden

# Define a class enumerating edges and vertices of a static cube of side length d.
class CUBE:
    def __init__(self, iden, d):
        self.vert = np.array([[ 1.,  1.,  1.,  1., -1., -1., -1., -1.],
                              [ 1.,  1., -1., -1.,  1.,  1., -1., -1.],
                              [ 1., -1.,  1., -1.,  1., -1.,  1., -1.]],
                             dtype = float) * d/2
        
        self.edge = np.array([[ 0,   1,   3,   2,   4,   5,   7,   6,   0,   1,   2,   3],
                              [ 1,   3,   2,   0,   5,   7,   6,   4,   4,   5,   6,   7]],
                             dtype = int)
        
        self.iden = iden

# Define a class for drawing traditional Cartesian x,y,z axes. Takes a flag "x",
# "y" or "z" to specify which axis is being drawn.
class AXIS:
    def __init__(self, iden, axis):
        
        if axis not in ["x", "y", "z"]:
            raise Exception("Improper axis flag specified.")
        
        axisDict = {"x":0, "y":1, "z":2}
        self.vert = np.zeros((3, 2), dtype = float)
        self.vert[axisDict[axis], 0] = -1000
        self.vert[axisDict[axis], 1] =  1000
        
        self.edge = np.array([[0],
                              [1]],
                             dtype = int)
        
        self.iden = iden


#%% Main loop

while True:
    
    # Wipe the screen by filling with a set colour
    window.fill((0, 0, 0))
    
    mouseX, mouseY = pg.mouse.get_pos()
    mouse_pos_buffer[0, :] = mouse_pos_buffer[1, :]
    mouse_pos_buffer[1, :] = np.array([mouseX, mouseY])
    # Note: the pg.mouse.get_rel function might do this automatically.
    
    for event in pg.event.get():
        
        # If window x is pressed, close the window.
        if event.type == pg.QUIT:
            pg.quit()
        
        # Spawn Lorentz attractor particle with randomised parameters
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_SPACE:
                PARTICLE_ARRAY.append(
                    particle_chaotic(
                        np.array([1, 0, 0], dtype = float),
                        np.array([10, 28, 7/3], dtype = float) * rng.normal(1, 0.1, 3),
                        tail_time
                    )
                )
                
            if event.key == pg.K_RALT:
                print(f"R is {camera_radius}, theta is {camera_theta}, phi is {camera_phi}")
                
            # Wipe all spawned attractors    
            if event.key == pg.K_ESCAPE:
                PARTICLE_ARRAY = []
                
            # Toggle cube and axis lines.
            # Note: an efficiency hack to stop us having to search for the context
            # cube and axes whenever RSHIFT is pressed is to put them at the top 
            # of the static object queue. To stop showing them, we just pop the 
            # top four items. But I can't spawn an item before these or it'll break.
            if event.key == pg.K_RSHIFT:
                axis_cubes_visible = not axis_cubes_visible
                if axis_cubes_visible == True:
                    STATIC_OBJECT_QUEUE.insert(0, CUBE("contextcube", 100))
                    STATIC_OBJECT_QUEUE.insert(0, AXIS("xaxis", "x"))
                    STATIC_OBJECT_QUEUE.insert(0, AXIS("yaxis", "y"))
                    STATIC_OBJECT_QUEUE.insert(0, AXIS("zaxis", "z"))
                else:
                    del STATIC_OBJECT_QUEUE[:4]
                    
                    
        if event.type == pg.MOUSEWHEEL:
                camera_radius += event.y * mouse_scroll_sens
     
    camera_position_sphere = np.array([camera_radius, camera_theta, camera_phi])
    camera_position_render = SPHERE_TO_CART(camera_position_sphere)
    camera_focalplane      = 2400
    
    # Update rotation direction using click and drag
    if pg.mouse.get_pressed()[0]:
        camera_theta += mouse_sens * (mouse_pos_buffer[1, 1] - mouse_pos_buffer[0, 1]) / (np.linalg.norm(camera_position_render)  - camera_focalplane)
        camera_phi   += mouse_sens * (mouse_pos_buffer[1, 0] - mouse_pos_buffer[0, 0]) / (np.linalg.norm(camera_position_render)  - camera_focalplane)
    
    
    # Draw static objects
    for static_object in STATIC_OBJECT_QUEUE:
        for i in range(static_object.vert.shape[1]):
            pg.draw.circle(
                window, 
                (255, 0, 0), 
                PERSPECTIVE_PROJECTION(static_object.vert[:, i], camera_focalplane, camera_position_render, OFFSET), 
                5
            )
        for i in range(static_object.edge.shape[1]):
            pg.draw.line(
                window, 
                (100, 100, 100), 
                PERSPECTIVE_PROJECTION(static_object.vert[:, static_object.edge[0, i]], camera_focalplane, camera_position_render, OFFSET),
                PERSPECTIVE_PROJECTION(static_object.vert[:, static_object.edge[1, i]], camera_focalplane, camera_position_render, OFFSET) 
            )
            
        
    
    # Tick forward all particles and draw
    for particle_instance in PARTICLE_ARRAY:
        for j, position in enumerate(particle_instance.pos_arr[:-1]):
                
            colour = (1 - j/(tail_time * FRAMERATE)) * particle_instance.colour + j/(tail_time * FRAMERATE) * np.array([255, 255, 255])
            
            pg.draw.circle(
                window,
                (255, 255, 255),
                PERSPECTIVE_PROJECTION(particle_instance.pos_arr[0], camera_focalplane, camera_position_render, OFFSET),
                3
            )
            pg.draw.aalines(
                window,
                tuple(colour),
                False,
                [PERSPECTIVE_PROJECTION(particle_instance.pos_arr[j], camera_focalplane, camera_position_render, OFFSET),
                 PERSPECTIVE_PROJECTION(particle_instance.pos_arr[j + 1], camera_focalplane, camera_position_render, OFFSET)]
            )
                
        particle_instance.tick_forward(FRAMERATE * time_dilation)
        
    
    # Update window.
    pg.display.update()