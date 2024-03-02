import websockets
import asyncio

import pyautogui as pg

async def handle_connection(websocket, path):
    async for message in websocket:
        x,y = pg.position()
        print(f"Received message: {message}")

        if message == "r_touch":
            pg.rightClick(x,y)
        elif message == "l_touch":
            pg.leftClick(x,y)

        await websocket.send("Message received")

async def main():
    async with websockets.serve(handle_connection, "localhost", 8765):
        print("Server started at localhost:8765")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
