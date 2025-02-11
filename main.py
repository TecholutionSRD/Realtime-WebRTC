"""This is the main file for the FastAPI application."""
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from PIL import Image
import httpx
from pydantic import BaseModel
import os
import sys
from dotenv import load_dotenv
import random
import cv2
from typing import List, Optional
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import load_config
from RAIT.cameras.recevier import CameraReceiver
from RAIT.functions.geminiInference import Gemini_Inference
from RAIT.functions.videoRecorder import VideoRecorder
from RAIT.functions.onboardingInformation import OnboardingInformation
from RAIT.functions.utilFunctions import *
from RAIT.functions.dataBase import check_knowledgebase, specific_knowledge_check
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
# Docs
#------------------------------------------------------#
@app.get("/", response_class=HTMLResponse)
async def render_docs():
    with open("../RAIT/index.html", "r") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)

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

#-----------------------------------------------------#
# Check Database
class DatabaseCheckResponse(BaseModel):
    """
    Response model for the database check API.

    This model is used to structure the response from the database check endpoint.
    It contains the list of known grasps and actions from the database.

    Attributes:
        grasps (List[str]): A list of known grasps stored in the database.
        actions (List[str]): A list of known actions stored in the database.
    """
    grasps: List[str]
    actions: List[str]

@app.get("/check_knowledge")
async def check_knowledge():
    """
    Checks all the databases and returns all the details.
    """
    known_grasps, known_actions = check_knowledgebase(config.get("DataBase", {}))
    return DatabaseCheckResponse(grasps=known_grasps, actions=known_actions)
#-----------------------------------------------------#
# Check Specific Knowledge
class SpecificKnowledgeCheckResponse(BaseModel):
    """
    Response model for the specific knowledge check API.

    This model is used to structure the response from the specific knowledge check endpoint.
    It contains the details of the specified knowledge item.

    Attributes:
        name (str): The name of the knowledge item.
        grasp (bool): Indicates if the item exists in the grasp database.
        action (bool): Indicates if the item exists in the action database.
    """
    name: str
    grasp: bool
    action: bool

@app.get("/check_specific_knowledge", response_model=SpecificKnowledgeCheckResponse)
async def check_specific_knowledge(name: str):
    """
    Checks the specific knowledge item and returns the details.
    
    Args:
        name (str): The name of the knowledge item to check.

    Returns:
        SpecificKnowledgeCheckResponse: A response object containing the details of the specified knowledge item.
    """
    result = specific_knowledge_check(config.get("DataBase", {}), name)
    if result is None:
        raise HTTPException(status_code=500, detail="Database configuration is incomplete or incorrect.")
    return SpecificKnowledgeCheckResponse(**result)

#-----------------------------------------------------#
# Onboarding Mode Confirmation
class OnboardingResponse(BaseModel):
    """
    Response model for the onboarding information API.

    This model is used to structure the response from the onboarding information endpoint.
    It contains the status, message, task ID, and details of the onboarding information.

    Attributes:
        status (str): The status of the onboarding information.
        message (str): A message indicating the status of the onboarding information.
        task_id (str): The ID of the task created in RLEF.
        details (dict): Additional details of the onboarding information.
    """
    status: str
    message: str
    task_id: Optional[str] = None
    details: Optional[dict] = None

class OnboardingRequest(BaseModel):
    """
    Request model for the onboarding information API.
    
    This model is used to structure the request for the onboarding information endpoint.
    It contains the action name, description, and list of objects.
    
    Attributes:
        action_name (str): The name of the action.
        description (str): The description of the action.
        objects (List[str]): A list of object names mentioned in the description.
        confirm (bool): A flag indicating whether the information should be confirmed.

    """
    action_name: str
    description: str
    objects: List[str]
    confirm: bool = False

@app.post("/onboarding_information", response_model=OnboardingResponse)
async def onboarding_information(request: OnboardingRequest):
    """
    Two-step process for onboarding information:
    1. First call: Show information for confirmation
    2. Second call with confirm=True: Create task in RLEF
    """
    if not request.confirm:
        # Return the details for confirmation
        return OnboardingResponse(
            status="pending",
            message="Please confirm the following information",
            details={
                "action_name": request.action_name,
                "description": request.description,
                "objects": request.objects
            }
        )
    
    onboarding_information = OnboardingInformation(config)
    try:
        task_id = onboarding_information.confirm_information(
            request.action_name,
            request.description,
            request.objects,
            True
        )
        return OnboardingResponse(
            status="success",
            message="Action successfully onboarded",
            task_id=task_id
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error creating task: {str(e)}"
        )

@app.get("/confirm_onboarding")
async def confirm_onboarding(confirm: bool = False):
    """
    Endpoint to confirm the onboarding information.

    Args:
        confirm (bool): The confirmation status of the information.

    Returns:
        str: A message indicating the confirmation status.
    """
    print("Received confirmation request.")
    if confirm:
        print("Onboarding information confirmed.")
        return "Onboarding information confirmed."
    else:
        print("Onboarding information not confirmed.")
        return "Onboarding information not confirmed."
#-----------------------------------------------------#
# Onboarding Mode

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8888)
