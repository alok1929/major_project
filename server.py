import cv2
import numpy as np
import base64
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from detect import process_frame
from utils.rep_tracking import SessionState
from utils.feedback import build_session_summary

# ============== FASTAPI APP ==============
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== WEBSOCKET ENDPOINT ==============
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ WebSocket connection established")
    
    session = SessionState()
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data.get('type') == 'frame':
                # Decode base64 frame
                frame_data = base64.b64decode(data['data'])
                np_arr = np.frombuffer(frame_data, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    await process_frame(frame, session, websocket)
                else:
                    await websocket.send_json({'type': 'error', 'message': 'Failed to decode frame'})
            
            elif data.get('type') == 'reset':
                session.reset()
                await websocket.send_json({'type': 'session_reset'})
                print("🔄 Session reset")
            
            elif data.get('type') == 'get_summary':
                summary = build_session_summary(session)
                await websocket.send_json(summary)
            
            elif data.get('type') == 'stop':
                summary = build_session_summary(session)
                await websocket.send_json(summary)
                break
                
    except WebSocketDisconnect:
        print("❌ WebSocket disconnected")
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        try:
            await websocket.send_json({'type': 'error', 'message': str(e)})
        except:
            pass


# ============== HEALTH CHECK ==============
@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Exercise tracker server running"}


# ============== RUN ==============
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
