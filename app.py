from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import pickle
import json
import os
import numpy as np
import uvicorn

app = FastAPI(title="Hand Gesture AI API")

# Mount static files for the frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load model on startup
MODEL_PATH = "models/gesture_model.pkl"
model = None

@app.on_event("startup")
async def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}")
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    else:
        print(f"Warning: Model not found at {MODEL_PATH}. Prediction will fail.")

@app.get("/")
async def get_root():
    # Serve the main frontend page
    with open(os.path.join("static", "index.html"), "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            
            if model is None:
                await websocket.send_json({"error": "Model not loaded"})
                continue
                
            try:
                # Expecting a list of 42 floats
                landmarks = json.loads(data)
                
                if len(landmarks) != 42:
                    await websocket.send_json({"error": "Expected 42 landmark coordinates"})
                    continue
                
                # Reshape for prediction
                features = np.array(landmarks).reshape(1, -1)
                
                # Predict
                prediction = model.predict(features)[0]
                
                # Send back the result
                await websocket.send_json({"gesture": prediction})
                
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid format"})
            except Exception as e:
                await websocket.send_json({"error": str(e)})

    except WebSocketDisconnect:
        print("Client disconnected.")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
