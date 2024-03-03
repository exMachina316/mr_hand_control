import websockets
import asyncio

import pyautogui as pg

r_down = False
l_down = False

async def handle_connection(websocket, path):
    global r_down, l_down
    async for message in websocket:
        width, height = pg.size()

        message = message.split(",")
        print(message)

        x,y = pg.position()

        ix = float(message[0])
        iy = float(message[1])

        if ix>0 and iy>0:
            x = ix*width
            y = iy*height
            pg.moveTo(x, y, duration=0.001)

        if "r_touch" in message:
            print("right_button_down")
            r_down = True
            pg.mouseDown(button="right")  
        elif r_down:
            print("right_button_up")
            pg.mouseUp(button="right")
            r_down = False
        
        if "l_touch" in message and not l_down:
            print("left_button_down")
            l_down = True
            pg.keyDown('shift')
            pg.mouseDown(button="middle")
        elif l_down:
            print("left_button_up")
            pg.keyUp('shift')
            pg.mouseUp(button="middle")
            l_down = False

        await websocket.send("Message received")

async def main():
    IP = input("Enter the IP address of the server: ")
    PORT = 8765
    async with websockets.serve(handle_connection, IP, PORT):
        print(f"Server started at {IP}:{PORT}")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Server stopped.")
