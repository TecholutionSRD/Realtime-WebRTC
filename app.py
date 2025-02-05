from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import random
from ultralytics import YOLO
import cv2
from typing import List

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables
load_dotenv()

# Get API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Load YOLO model
model = YOLO('yolov8n.pt')
token: str

class WeatherResponse(BaseModel):
    temperature: float
    unit: str

class DetectionResponse(BaseModel):
    objects: List[str]

@app.get("/session")
async def get_session():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            'https://api.openai.com/v1/realtime/sessions',
            headers={
                'Authorization': f'Bearer {OPENAI_API_KEY}',
                'Content-Type': 'application/json'
            },
            json={
                "model": "gpt-4o-realtime-preview-2024-12-17",
                "voice": "echo"
            }
        )
        return response.json()

@app.get("/weather/{location}")
async def get_weather(location: str):
    try:
        async with httpx.AsyncClient() as client:
            geocoding_response = await client.get(
                f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1"
            )
            geocoding_data = geocoding_response.json()
            
            if not geocoding_data.get("results"):
                return {"error": f"Could not find coordinates for {location}"}
            
            lat = geocoding_data["results"][0]["latitude"]
            lon = geocoding_data["results"][0]["longitude"]
            
            weather_response = await client.get(
                f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m"
            )
            weather_data = weather_response.json()
            
            temperature = weather_data["current"]["temperature_2m"]
            return WeatherResponse(temperature=temperature, unit="celsius")
            
    except Exception as e:
        return {"error": f"Could not get weather data: {str(e)}"}

import time


# @app.get("/detect")
# async def detect_objects():
#     cap = None
#     try:
#         # Initialize video capture
#         cap = cv2.VideoCapture(0)
        
#         if not cap.isOpened():
#             raise HTTPException(status_code=500, detail="Could not access camera")

#         # Wait for camera initialization
#         time.sleep(1)  # 1 second wait for camera to initialize
        
#         # Capture frame
#         ret, frame = cap.read()
#         if not ret:
#             raise HTTPException(status_code=500, detail="Could not capture frame")

#         # Release camera immediately after capturing
#         cap.release()
#         cap = None

#         # Run inference
#         results = model(frame)[0]
        
#         # Process results
#         detected_objects = []
#         for box in results.boxes:
#             if box.conf.item() > 0.3:  # confidence threshold
#                 class_id = int(box.cls.item())
#                 class_name = model.names[class_id]
#                 if class_name not in detected_objects:
#                     detected_objects.append(class_name)

#         return DetectionResponse(objects=detected_objects)

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error during object detection: {str(e)}")
        
#     finally:
#         # Ensure camera is released
#         if cap is not None:
#             cap.release()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)