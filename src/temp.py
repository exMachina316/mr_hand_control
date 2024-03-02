import pyautogui
import time

while True:
    x,y = pyautogui.position()
    print(x,y)
    time.sleep(0.1)