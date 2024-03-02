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
    IP = input("Enter the IP address of the server: ")
    PORT = 8765
    async with websockets.serve(handle_connection, IP, PORT):
        print(f"Server started at {IP}:{PORT}")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
