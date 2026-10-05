import asyncio
import websockets
import sys

async def test():
    try:
        async with websockets.connect('ws://localhost:8000/api/v1/vision/stream') as ws:
            print("Connected successfully!")
            await ws.send("test")
            res = await ws.recv()
            print("Received:", res)
    except websockets.exceptions.ConnectionClosed as e:
        print(f"Connection closed: Code={e.code}, Reason={e.reason}")
    except Exception as e:
        print(f"Exception: {e}")

asyncio.run(test())
