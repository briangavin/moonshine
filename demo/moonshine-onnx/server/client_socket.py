import asyncio
import websockets

# Define color codes
RED = '\033[91m'
GREEN = '\033[92m'
BLUE = '\033[94m'
END_COLOR = '\033[0m'

async def handle_disconnect(websocket):
    print("Disconnected from server")

async def main():
    async with websockets.connect("ws://localhost:8765") as websocket:
        print("Connected to server")
        try:
            # Wait for messages from server
            async for message in websocket:
                print(f"{RED}Received message:{END_COLOR}", message)
        except websockets.ConnectionClosed:
            await handle_disconnect(websocket)

asyncio.run(main())