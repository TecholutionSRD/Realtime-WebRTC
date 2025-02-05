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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import load_config
from RAIT.cameras.recevier import CameraReceiver
from RAIT.functions.gemini_inference import Gemini_Inference
from RAIT.functions.video_recorder import VideoRecorder
from RAIT.functions.utils_main import *
#----------------- Load Configurations -----------------#
config = load_config("../config/config.yaml")
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
    Response model for the object detection API.

    This model is used to structure the response from the object detection endpoint.
    It contains a list of detected objects as strings.

    Attributes:
        objects (List[str]): A list of object names identified in the image.
    """
    objects: List[str]

@app.get("/detect")
async def gemini_detection():
    """
    Endpoint to perform object detection using the Gemini inference model.

    This function receives an image from a camera, passes it through the Gemini
    inference model to detect objects, and returns a response with the detected
    objects.

    It captures frames from a camera, processes the image to detect objects, and
    responds with a list of detected objects.

    Returns:
        ObjectDetectionResponse: A response object containing a list of detected objects.
    """
    gemini_config = config.get("Gemini", {})
    camera = CameraReceiver(config)
    gemini = Gemini_Inference(config)
    recording_dir = gemini_config.get("recording_dir")
    save_path = f"{recording_dir}/{int(time.time())}"

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
    Response model for the video recording API.

    This model is used to structure the response from the video recording endpoint.
    It contains a comment indicating the status of the video recording.

    Attributes:
        comment (str): A message describing the result or status of the video recording.
    """
    comment: str

@app.get("/record_video")
async def record_videos(num_recordings: int = 1, action_name: str = "pouring", objects: List[str] = ["red soda can", "white cup"]):
    """
    Endpoint to initiate video recording using the provided parameters.

    This function starts video recording for the specified number of recordings and stores them
    in the corresponding directory. After the recording is complete, it returns a response indicating 
    the status of the video recording.

    Args:
        num_recordings (int): The number of video recordings to perform.
        action_name (str): The name of the action being performed (for directory naming).
        objects (List[str]): List of objects relevant to the recording.

    Returns:
        VideoRecordingResponse: A response with a message indicating the completion of the video recording.
    """
    receiver = CameraReceiver(config)
    await receiver.connect()
    recorder = VideoRecorder(receiver, config, num_recordings, action_name, objects)
    await recorder.record_video()
    return VideoRecordingResponse(comment="Video recording complete.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)
