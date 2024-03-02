import pyautogui as pg
import time

x,y = pg.position()
width, height = pg.size()
# pg.leftClick(x,y)

pg.moveRel(-200,0, tween=pg.easeInOutQuad, duration=0.2)