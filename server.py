from fastapi import FastAPI
import uvicorn
import websockets
from fastapi import WebSocket
import asyncio
import json

from llm_feedback import get_exercise_feedback_async
from test import process_video

app = FastAPI()

import base64
import numpy as np
import cv2

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_json()
        if "frame" in data:
            # Decode base64 to image
            frame_data = base64.b64decode(data["frame"])
            np_arr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            # Optionally, you could save/process a sequence of frames.
            # Here we process a single frame by wrapping it in a list and calling process_video.
            # But process_video currently expects a video path.
            # You'd likely need to refactor process_video to allow processing frames directly.
            # We'll assume you create a new function called process_frame for this example:
            # result = await process_frame(frame)

            # For demonstration, let's just acknowledge receipt
            await websocket.send_json({"status": "frame received"})
        else:
            await websocket.send_json({"error": "No 'frame' key in received data"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)