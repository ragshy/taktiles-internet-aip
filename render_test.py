from ursina import *
from ursina.prefabs.first_person_controller import FirstPersonController
import pyautogui
import random
import math

app = Ursina()

window.title = 'Room'
window.borderless = False
window.exit_button.visible = True
window.fps_counter.enabled = False

mouse.visible = False
mouse.locked = True


ground = Entity(model="plane", color = color.white, scale=(100, 1, 100), collider="box", position=(0, 0, 0))
cube = Entity(model='cube',position = (0,2,2), color = color.red)
player = FirstPersonController()

def update():
    cube.rotation_y += time.dt * 10                 
    if held_keys['up arrow']:                           
        player.world_position += (0, 0, time.dt*10)           
    if held_keys['down arrow']:                            
        player.world_position -= (0, 0, time.dt*10) 
    if held_keys['left arrow']:
        player.world_rotation_y -=time.dt*50
    if held_keys['right arrow']:
        player.world_rotation_y +=time.dt*50
app.run()

