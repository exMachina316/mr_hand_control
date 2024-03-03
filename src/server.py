import websockets
import asyncio

import pyautogui as pg

async def handle_connection(websocket, path):
    async for message in websocket:
        width, height = pg.size()

        message = message.split(",")

        x,y = pg.position()

        ix = float(message[0])
        iy = float(message[1])

        if ix>0 and iy>0:
            x = ix*width
            y = iy*height
            pg.moveTo(x, y, duration=0.001)

        if "r_touch" in message:
            pg.mouseDown(button="right")
        else:
            pg.moseuUp(button="right")
        
        if "l_touch" in message:
            pg.mouseDown(button="left")
        else:
            pg.mouseUp(button="left")

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
