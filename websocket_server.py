import asyncio
import json
from threading import Thread

import websockets.server as ws_server
from websockets.typing import Data

from message_queue import client_request_queue, message_queue
from mnist import run_model
from util.logger import app_logger

client_connected = False


async def consumer(message: Data):
    if message == "cancel":
        print(message)
        client_request_queue.put("cancel")
    elif message == "run":
        print(message)
        thread = Thread(
            target=run_model, daemon=True, args=(message_queue, client_request_queue)
        )
        thread.start()


async def consumer_handler(websocket: ws_server.WebSocketServerProtocol):
    async for message in websocket:
        await consumer(message)


async def producer_handler(websocket: ws_server.WebSocketServerProtocol):
    global client_connected
    while True:
        # Await a connected client before dumping messages
        if message_queue.qsize() > 0 and client_connected:
            message = message_queue.get()
            if not isinstance(message, str):
                message = json.dumps(message.as_dict())
            await websocket.send(message)
        await asyncio.sleep(0.01)


async def handler(websocket: ws_server.WebSocketServerProtocol):
    global client_connected
    try:
        app_logger.info("Client connected")
        client_connected = True
        consumer_task = asyncio.create_task(consumer_handler(websocket))
        producer_task = asyncio.create_task(producer_handler(websocket))
        _done, pending = await asyncio.wait(
            [consumer_task, producer_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
    finally:
        client_connected = False
        app_logger.info("Client disconnected")


async def main():
    async with ws_server.serve(handler, "localhost", 8765):
        app_logger.info("Websocket open")
        message_queue.put("testing!")
        message_queue.put("another!!")
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
