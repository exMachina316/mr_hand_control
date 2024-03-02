import pyautogui as pg

x,y = pg.position()
pg.leftClick(x,y)

# for x in range(200):
#     print(x)
#     pg.moveRel(1,0)
#     time.sleep(0.1)