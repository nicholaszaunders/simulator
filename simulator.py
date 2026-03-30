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
camera_radius       = 600 #1000
camera_theta        = 1.4314285714285735#0
camera_phi          = 0.8518267348260169#np.pi/4
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

# Define a particle class goverend by a Lorentz attractor.
class particle_chaotic:
    def __init__(self, initial_position, parameters, tail_time):
        self.pos_arr       = np.zeros((int(tail_time * FRAMERATE), 3))
        self.pos_arr[0, :] = initial_position
        self.params        = parameters #[σ, ρ, β]
        self.colour = (np.random.rand() * 255, np.random.rand() * 255, np.random.rand() * 255)
        
    # Method which when called gives the velocity values of the particle based 
    # on the Lorentz attractor ODE.
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
        
        self.pos_arr[1:, :]   = self.pos_arr[0:-1, :]
        self.pos_arr[0, :]      = newpos
    
particle_array = []

obj_cube = np.ones((3, 8))
n = 0
for i in [0,1]:
    for j in [0,1]:
        for k in [0,1]:
            obj_cube[:, n] = np.array([(-1)**i, (-1)**j, (-1)**k]) * 50
            n += 1

obj_cube_edges = np.array([    
    [1, 1, 1],
    [-1, 1, 1],
    [-1, -1, 1],
    [1, -1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, -1],
    [1, 1, 1],
    [-1, 1, 1],
    [-1, 1, -1],
    [-1, 1, 1],
    [-1, -1, 1],
    [-1, -1, -1],
    [-1, -1, 1],
    [1, -1, 1],
    [1, -1, -1],
    [1, 1, -1],
    [-1, 1, -1],
    [-1, -1, -1],
    [1, -1, -1],
]) * 50

#%% Main loop

while True:
    
    # Wipe the screen
    window.fill((255,255,255))
    
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
                particle_array.append(
                    particle_chaotic(
                        np.array([1, 0, 0], dtype = float),
                        np.array([10, 28, 7/3], dtype = float) * rng.normal(1, 0.1, 3),
                        2.0
                    )
                )
                
            if event.key == pg.K_RALT:
                print(f"R is {camera_radius}, theta is {camera_theta}, phi is {camera_phi}")
                    
        if event.type == pg.MOUSEWHEEL:
                camera_radius += event.y * mouse_scroll_sens
     
    camera_position_sphere = np.array([camera_radius, camera_theta, camera_phi])
    camera_position_render = SPHERE_TO_CART(camera_position_sphere)
    camera_focalplane      = 2400
    
    # Update rotation direction using click and drag
    if pg.mouse.get_pressed()[0]:
        camera_theta += mouse_sens * (mouse_pos_buffer[1, 1] - mouse_pos_buffer[0, 1]) / (np.linalg.norm(camera_position_render)  - camera_focalplane)
        camera_phi   += mouse_sens * (mouse_pos_buffer[1, 0] - mouse_pos_buffer[0, 0]) / (np.linalg.norm(camera_position_render)  - camera_focalplane)

    
    # Draw origin and x,y,z lines relative to camera
    # origin
    origin_point = PERSPECTIVE_PROJECTION(np.array([0,0,0]), camera_focalplane, camera_position_render, OFFSET)
    pg.draw.circle(window, (0, 0, 0), origin_point, 2)
    
    # x-axis
    x_axis_point = PERSPECTIVE_PROJECTION(np.array([100,0,0]), camera_focalplane, camera_position_render, OFFSET)
    x_axis_2d_angle = np.arctan2(x_axis_point[1] - origin_point[1], x_axis_point[0] - origin_point[0])
    pg.draw.line(
        window,
        (250, 0, 0),
        origin_point,
        (origin_point[0] + WINDOW_SIZE[0] * np.cos(x_axis_2d_angle), origin_point[1] + WINDOW_SIZE[1] * np.sin(x_axis_2d_angle))
    )
    x_axis_point = PERSPECTIVE_PROJECTION(np.array([-100,0,0]), camera_focalplane, camera_position_render, OFFSET)
    x_axis_2d_angle = np.arctan2(x_axis_point[1] - origin_point[1], x_axis_point[0] - origin_point[0])
    pg.draw.line(
        window,
        (150, 0, 0),
        origin_point,
        (origin_point[0] + WINDOW_SIZE[0] * np.cos(x_axis_2d_angle), origin_point[1] + WINDOW_SIZE[1] * np.sin(x_axis_2d_angle))
    )
    
    # y-axis
    y_axis_point = PERSPECTIVE_PROJECTION(np.array([0,100,0]), camera_focalplane, camera_position_render, OFFSET)
    y_axis_2d_angle = np.arctan2(y_axis_point[1] - origin_point[1], y_axis_point[0] - origin_point[0])
    pg.draw.line(
        window,
        (0, 250, 0),
        origin_point,
        (origin_point[0] + WINDOW_SIZE[0] * np.cos(y_axis_2d_angle), origin_point[1] + WINDOW_SIZE[1] * np.sin(y_axis_2d_angle))
    )
    y_axis_point = PERSPECTIVE_PROJECTION(np.array([0,-100,0]), camera_focalplane, camera_position_render, OFFSET)
    y_axis_2d_angle = np.arctan2(y_axis_point[1] - origin_point[1], y_axis_point[0] - origin_point[0])
    pg.draw.line(
        window,
        (0, 150, 0),
        origin_point,
        (origin_point[0] + WINDOW_SIZE[0] * np.cos(y_axis_2d_angle), origin_point[1] + WINDOW_SIZE[1] * np.sin(y_axis_2d_angle))
    )
    
    # z-axis
    z_axis_point = PERSPECTIVE_PROJECTION(np.array([0,0,100]), camera_focalplane, camera_position_render, OFFSET)
    z_axis_2d_angle = np.arctan2(z_axis_point[1] - origin_point[1], z_axis_point[0] - origin_point[0])
    pg.draw.line(
        window,
        (0, 0, 250),
        origin_point,
        (origin_point[0] + WINDOW_SIZE[0] * np.cos(z_axis_2d_angle), origin_point[1] + WINDOW_SIZE[1] * np.sin(z_axis_2d_angle))
    )
    z_axis_point = PERSPECTIVE_PROJECTION(np.array([0,0,-100]), camera_focalplane, camera_position_render, OFFSET)
    z_axis_2d_angle = np.arctan2(z_axis_point[1] - origin_point[1], z_axis_point[0] - origin_point[0])
    pg.draw.line(
        window,
        (0, 0, 50),
        origin_point,
        (origin_point[0] + WINDOW_SIZE[0] * np.cos(z_axis_2d_angle), origin_point[1] + WINDOW_SIZE[1] * np.sin(z_axis_2d_angle))
    )
    
    # Tick forward all particles and draw
    for i, particle_instance in enumerate(particle_array):
        j = 0
        while j < len(particle_instance.pos_arr):
            position = particle_instance.pos_arr[j]
                
            pg.draw.circle(
                window,
                particle_instance.colour,
                PERSPECTIVE_PROJECTION(position, camera_focalplane, camera_position_render, OFFSET),
                3
            )
            
            j += 1
                
        particle_instance.tick_forward(FRAMERATE * time_dilation)
        
     
    
    # # Draw unit cube vertices
    for i in range(np.shape(obj_cube)[1]):
        projected_point = PERSPECTIVE_PROJECTION(obj_cube[:, i], camera_focalplane, camera_position_render, OFFSET)
        pg.draw.circle(
            window, 
            (255, 0, 0), 
            projected_point, 
            5
        )
   
    # # Draw unit cube edges
    i = 1
    while i < np.shape(obj_cube_edges)[0]:
        input_point_1       = obj_cube_edges[i - 1, :]
        input_point_2       = obj_cube_edges[i, :]
        projected_point_1   = PERSPECTIVE_PROJECTION(input_point_1, camera_focalplane, camera_position_render, OFFSET)
        projected_point_2   = PERSPECTIVE_PROJECTION(input_point_2, camera_focalplane, camera_position_render, OFFSET)
        pg.draw.line(
            window, 
            (100, 100, 100), 
            projected_point_1, 
            projected_point_2
        )
        i += 1
    
    # Update window.
    pg.display.update()