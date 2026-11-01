import asyncio
import websockets

async def test():
    try:
        async with websockets.connect('ws://localhost:8000/api/v1/vision/stream') as ws:
            print("Connected!")
            await ws.send('test')
            print(await ws.recv())
    except Exception as e:
        print(f"Error: {e}")

asyncio.run(test())
