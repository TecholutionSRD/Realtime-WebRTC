"""This is the main file for the FastAPI application."""
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import httpx
from pydantic import BaseModel
import os
import sys
from dotenv import load_dotenv
import random
import cv2
from typing import List
from RAIT.cameras.recevier import CameraReceiver
from RAIT.functions.gemini_inference import Gemini_Inference
from RAIT.functions.video_recorder import VideoRecorder
from RAIT.functions.utils_main import *
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import load_config

#----------------- Load Configurations -----------------#
config = load_config()
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

#------------------ Create Application -----------------#
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
#------------------------------------------------------#
# Session : Realtime Session
#------------------------------------------------------#
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
    


#-----------------------------------------------------#
# Function Calling : Functions to do specific function calling.
#-----------------------------------------------------#
# Gemini Object Detection

class ObjectDetectionResponse(BaseModel):
    """
    Response model for Object Detection API.
    """
    detected_objects = List[str]

@app.get("/detect")
async def gemini_detection():
        config = load_config("config/config.yaml")
        gemini_config = config.get("Gemini", {})
        camera = CameraReceiver(config)
        gemini = Gemini_Inference(config)
        recording_dir = gemini_config.get("recording_dir")
        save_path = f"{recording_dir}/{int(time.time())}"
        print(save_path)
        
        frames = await camera.capture_frames(save_path)
        color_frame_path = frames.get('rgb')
        
        if color_frame_path:
            color_image = Image.open(color_frame_path)
            detected_objects = gemini.detect_objects(color_image)
            print(f"Detected objects: {detected_objects}")
        else:
            print("No frames received or saved.")

        return ObjectDetectionResponse(objects=detected_objects)
#-----------------------------------------------------#
# Record 8 Second Video

class VideoRecordingResponse(BaseModel):
    """
    Response model for Video Recording API.
    """
    comment = str

@app.get("/record_videos")
async def record_videos():
    config = load_config("config/config.yaml")
    gemini_config = config.get("Gemini", {})
    camera = CameraReceiver(config)
    gemini = Gemini_Inference(config)
    recording_dir = gemini_config.get("recording_dir")
    save_path = f"{recording_dir}/{int(time.time())}"
    video_recorder = VideoRecorder(config)
    video_recorder.record_video(save_path)
    return VideoRecordingResponse(comment="Video recording complete.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)
