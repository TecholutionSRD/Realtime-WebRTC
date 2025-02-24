"""This is the main file for the FastAPI application."""
import json
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
from typing import List, Optional, Tuple, Dict
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import load_config
from RAIT.cameras.recevier import CameraReceiver
from RAIT.functions.geminiInference import Gemini_Inference
from RAIT.functions.videoRecorder import VideoRecorder
from RAIT.functions.onboardingInformation import OnboardingInformation
from RAIT.functions.utilFunctions import *
from RAIT.functions.dataBase import *
from RAIT.functions.saveVideo import video_to_bucket
from RAIT.dsr_control_api.dsr_control_api.cobotclient import CobotClient
from RAIT.Fundamental_Actions.data_4d.generate_data import VideoProcessor4D
from RAIT.rlef.rlef_utils import RLEFManager
#--------------------Global Variables--------------------#
current_task_id = None
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
# Get Gemini Object Centers
class ObjectCentersResponse(BaseModel):
    centers: List[Tuple[float, float, float]]

def create_coordinate_tuple(coords):
    if isinstance(coords, (list, tuple)) and len(coords) == 3:
        return tuple(coords)
    elif isinstance(coords, (int, float)):
        return (float(coords), 0.0, 0.0)
    return None

@app.get("/object_centers")
async def object_centers(objects: str = None):
    print("Received request for object centers.")
    if objects is None:
        print("No objects provided.")
        return {"error": "No objects provided."}
    
    try:
        objects_list = json.loads(objects)
        print(f"Parsed objects list: {objects_list}")
        client = CobotClient(ip=config.get("Cobot_URL"))
        camera = CameraReceiver(config)
        gemini = Gemini_Inference(config)
        
        # Get detection results
        print("Starting object detection.")
        transformed_centers = await gemini.detect(camera, target_class=objects_list)
        print(f"Transformed centers: {transformed_centers}")
        
        # Convert to proper format
        if not isinstance(transformed_centers, list):
            transformed_centers = [transformed_centers[0]/1000, transformed_centers[1]/1000, transformed_centers[2]/1000]
            
        # Create objects_to_spawn dictionary
        objects_to_spawn = {}
        coordinates = []

        # Ensure we have 3 coordinates
        print(f"Number of transformed centers: {len(transformed_centers)}")
        if len(transformed_centers) >= 3:
            coords = [float(transformed_centers[0]), float(transformed_centers[1]), float(transformed_centers[2])]
            print(f"Coordinates: {coords}")
            
            for obj in objects_list:
                obj = obj.replace(" ", "_")
                objects_to_spawn[obj] = coords
                print(f"Object to spawn: {obj}, Coordinates: {coords}")
            
            # Add rotation angles
            coordinates = coords + [90, -90, 180]
            print(f"Final coordinates with rotation: {coordinates}")
        else:
            raise ValueError("Invalid number of coordinates")

        # Construct payload
        payload = {
            "objects_to_spawn": objects_to_spawn,
            "fundamental_actions": {
                "grasp": {
                    "is_trajectory": False,
                    "coordinates": coordinates,
                    "csv_file": None,
                    "gripper_mode_post_action": 2,
                    "grasp_mode": "HORIZONTAL"
                }
            }
        }
        
        client.send_trajectory_data(payload)
        print("Sent trajectory data.")
        
        # Format response
        response_centers = [(float(transformed_centers[0]), 
                           float(transformed_centers[1]), 
                           float(transformed_centers[2]))]
        print(f"Response centers: {response_centers}")
        return ObjectCentersResponse(centers=response_centers)
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

#-----------------------------------------------------#
# Record 8 Second Video

# class VideoRecordingResponse(BaseModel):
#     """
#     Response model for the video recording API.

#     This model is used to structure the response from the video recording endpoint.
#     It contains a comment indicating the status of the video recording.

#     Attributes:
#         comment (str): A message describing the result or status of the video recording.
#     """
#     comment: str
#     gcp_link: str

# @app.get("/record_video")
# async def record_videos(num_recordings: int = 1, action_name: str = "pouring", objects: List[str] = ["red soda can", "white cup"]):
#     """
#     Endpoint to initiate video recording using the provided parameters.

#     This function starts video recording for the specified number of recordings and stores them
#     in the corresponding directory. After the recording is complete, it returns a response indicating 
#     the status of the video recording.

#     Args:
#         num_recordings (int): The number of video recordings to perform.
#         action_name (str): The name of the action being performed (for directory naming).
#         objects (List[str]): List of objects relevant to the recording.

#     Returns:
#         VideoRecordingResponse: A response with a message indicating the completion of the video recording.
#     """
    
#     #------------------------------------#
#     rlef_config = load_config("../config/config.yaml")['RLEF']
#     manager = RLEFManager(rlef_config)
#     #------------------------------------#
    
#     receiver = CameraReceiver(config)
#     await receiver.connect()
#     recorder = VideoRecorder(receiver, config, num_recordings, action_name, objects)
#     sample_folder = await recorder.record_video()
    
#     # Send data to GCP for Hamer
#     try:
#         config_path = '../RAIT/config/config.yaml'
#         preprocessor = VideoProcessor4D(config_path)
#         preprocessor.set_details(objects, action_name)
#         num_samples = preprocessor.check_available_samples(action_name)
#         preprocessor.process_latest_recording(action_name)

#         status = ("Ready for learning" if num_samples >= 30 
#                  else f"Insufficient samples - need at least 30 (current: {num_samples})")
        
#     except Exception as e:
#         print(f"An error occurred while sending data to Hammer: {e}")
    
#     # Send data to GCP for storing
#     try:
#         num_samples = len(os.listdir(f"data/recordings/{action_name}/"))
#         source_video_path = f"data/recordings/{action_name}/sample_{num_samples}/{action_name}_video.mp4"
#         gcp_url = video_to_bucket(source_path=source_video_path)
#         print("Data stored succesfully in GCP")
#     except Exception as e:
#         gcp_url = ""
#         print(f"An error occurred while sending data to Cloud Storage: {e}")
    
#     # # Send data to RLEF
#     try:
#         #------------------------------------#
#         #Task ID remaining.
#         task_id = manager.get_or_create_task(action_name)
#         print(f"Current_Task_ID: {task_id}")
#         manager.upload_to_rlef(source_video_path,task_id)
#         #------------------------------------#
#         print("Data sent succesfully in RLEF")
#     except Exception as e:
#         rlef_link = ""
#         print(f"An error occurred while sending data to RLEF: {e}")
#     return VideoRecordingResponse(comment="Video recording complete.",gcp_link=gcp_url)

from fastapi import APIRouter
from typing import List
import asyncio

class VideoRecordingResponse(BaseModel):
    """
    Response model for the video recording API.

    This model is used to structure the response from the video recording endpoint.
    It contains a comment indicating the status of the video recording.

    Attributes:
        comment (str): A message describing the result or status of the video recording.
    """
    comment: str
    gcp_link: str

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
    
    #------------------------------------#
    rlef_config = load_config("../config/config.yaml")['RLEF']
    manager = RLEFManager(rlef_config)
    #------------------------------------#
    
    receiver = CameraReceiver(config)
    await receiver.connect()
    recorder = VideoRecorder(receiver, config, num_recordings, action_name, objects)
    sample_folder = await recorder.record_video()
    
    # Asynchronously handle Hammer processing so it doesn't block the rest of the process
    async def process_hammer_data():
        print("Hitting Hamer")
        try:
            print("Hitting Hamer TRY")
            config_path = '../config/config.yaml'
            preprocessor = VideoProcessor4D(config_path)
            print("Processor Loaded")
            preprocessor.set_details(objects, action_name)
            print("Details set")
            num_samples = preprocessor.check_available_samples(action_name)
            print(f"Num samples : {num_samples}")
            preprocessor.process_latest_recording(action_name)
            print("-----------Processing-------------")

            status = ("Ready for learning" if num_samples >= 30 
                    else f"Insufficient samples - need at least 30 (current: {num_samples})")
            print(status)
        except Exception as e:
            print(f"An error occurred while sending data to Hammer: {e}")

    # Run Hammer processing in the background
    asyncio.create_task(process_hammer_data())
    
    # Send data to GCP for storing
    try:
        num_samples = len(os.listdir(f"data/recordings/{action_name}/"))
        source_video_path = f"data/recordings/{action_name}/sample_{num_samples}/{action_name}_video.mp4"
        gcp_url = video_to_bucket(source_path=source_video_path)
        print("Data stored successfully in GCP")
    except Exception as e:
        gcp_url = ""
        print(f"An error occurred while sending data to Cloud Storage: {e}")
    
    # Send data to RLEF
    try:
        task_id = manager.get_or_create_task(action_name)
        print(f"Current_Task_ID: {task_id}")
        manager.upload_to_rlef(source_video_path, task_id)
        print("Data sent successfully to RLEF")
    except Exception as e:
        rlef_link = ""
        print(f"An error occurred while sending data to RLEF: {e}")
    
    return VideoRecordingResponse(comment="Video recording complete.", gcp_link=gcp_url)
#-----------------------------------------------------#
class DatabaseCheckResponse(BaseModel):
    """
    Response model for the database check API.

    This model is used to structure the response from the database check endpoint.
    It contains the list of known grasps and actions from the database.

    Attributes:
        grasps (List[str]): A list of known grasps stored in the database.
        actions (List[str]): A list of known actions stored in the database.
        action_objects (Dict[str, List[str]]): A dictionary mapping action names to their associated objects.
    """
    grasps: List[str]
    actions: List[str]
    action_objects: Dict[str, List[str]]

@app.get("/check_knowledge")
async def check_knowledge():
    """
    Checks all the databases and returns all the details.
    """
    known_grasps, known_actions = check_knowledgebase(config.get("DataBase", {}))
    
    # Get associated objects for each action
    action_objects_dict = {}
    for action in known_actions:
        objects = get_action_objects(config.get("DataBase", {}), action)
        action_objects_dict[action] = objects
    
    print("-"*100)
    print("Knowledge")
    print(f"Grasps: {known_grasps}")
    print(f"Actions: {known_actions}")
    print(f"Action Objects: {action_objects_dict}")
    
    return DatabaseCheckResponse(
        grasps=known_grasps,
        actions=known_actions,
        action_objects=action_objects_dict
    )
#-----------------------------------------------------#
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
    object_in_action: bool

@app.get("/check_specific_knowledge", response_model=SpecificKnowledgeCheckResponse)
async def check_specific_knowledge(name: str):
    """
    Checks the specific knowledge item and returns the details.
    
    Args:
        name (str): The name of the knowledge item to check.

    Returns:task_name
        SpecificKnowledgeCheckResponse: A response object containing the details of the specified knowledge item.
    """
    result = specific_knowledge_check(config.get("DataBase", {}), name)
    if result is None:
        raise HTTPException(status_code=500, detail="Database configuration is incomplete or incorrect.")
    return SpecificKnowledgeCheckResponse(**result)
#-----------------------------------------------------#
# Onboarding Mode Confirmation
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
        action_name (str): The ID of the task created in RLEF.
        details (dict): Additional details of the onboarding information.
    """
    status: str
    message: str
    action_name: Optional[str] = None
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
        # Update the task_id globally
        print(f"Current ID Updated : {current_task_id} to {task_id}")
        current_task_id = task_id
        
        
        return OnboardingResponse(
            status="success",
            message="Action successfully onboarded",
            action_name = request.action_name
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
class ExecutionRequest(BaseModel):
    """
    Request model for the execution mode API.
    
    Attributes:
        action (str): The name of the action to be executed
        objects (List[str]): List of objects involved in the action
    """
    action: str
    objects: List[str]

class ExecutionResponse(BaseModel):
    """
    Response model for the execution mode API.
    
    Attributes:
        status: str - Success/failure status
        message: str - Detailed message about the execution
        action: str - The action that was executed
        object_locations: Optional[List[Tuple[float, float, float]]] - Detected object coordinates
    """
    status: str
    message: str
    action: str
    object_locations: Optional[List[Tuple[float, float, float]]] = None

@app.post("/execute", response_model=ExecutionResponse)
async def execution_mode(request: ExecutionRequest):
    """
    Endpoint to execute a specific action on given objects.
    
    Args:
        request (ExecutionRequest): Contains action name and list of objects
    
    Returns:
        ExecutionResponse: Details about the execution status
    """
    print(f"[DEBUG] Starting execution mode with action: {request.action} and objects: {request.objects}")
    
    try:
        # Load configuration and check knowledge base
        config = load_config("../config/config.yaml")
        db_config = config.get("DataBase", {})
        grasp_names, action_names = check_knowledgebase(db_config)
        print(f"Grasps: {grasp_names}")
        print(f"Action: {action_names}")
        
        # Handle grasp/pick related actions
        grasp_related_actions = ["grasp", "grasping", "pick", "pickup"]
        if request.action.lower() in grasp_related_actions:
            print("[DEBUG] Recognized as grasp/pick related action")
            
            grasp_db_path = os.path.join(db_config["base_dir"], db_config["grasp"])
            grasp_df = pd.read_csv(grasp_db_path)
            
            related_objects = []
            for obj in request.objects:
                obj = obj.lower()
                if obj in grasp_df["name"].values:
                    grasp_distance = grasp_df[grasp_df["name"] == obj]["grasp_distance"].values[0]
                    pickup_mode = grasp_df[grasp_df["name"] == obj]["pickup_mode"].values[0]
                    related_objects.append({
                        "object_name": obj,
                        "grasp_distance": grasp_distance,
                        "pickup_mode": pickup_mode
                    })
                else:
                    print(f"[ERROR] Object '{obj}' not found in grasp database")

            if not related_objects:
                raise HTTPException(
                    status_code=404,
                    detail="None of the provided objects are found in grasping database"
                )

            # Step 2: Get object centers
            print("[DEBUG] Getting object centers")
            objects_json = json.dumps(request.objects)
            try:
                centers_response = await object_centers(objects=objects_json)
                print(f"[DEBUG] Received object centers: {centers_response.centers}")
            except Exception as e:
                print(f"[ERROR] Failed to get object centers: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to detect object locations: {str(e)}"
                )

            # Step 3: Send to cobot client
            try:
                print("[DEBUG] Connecting to cobot client")
                client = CobotClient(ip=config.get("Cobot_URL"))
                
                # Assuming centers_response.centers is available
                # Here we prepare coordinates (including rotation)
                if hasattr(centers_response, 'centers') and centers_response.centers:
                    coordinates = list(centers_response.centers[0]) + [90, -90, 180]
                    
                    # Prepare objects_to_spawn dictionary
                    objects_to_spawn = {
                        obj.replace(" ", "_"): list(centers_response.centers[0])
                        for obj in request.objects
                    }
                    
                    if request.action == "grasping":
                        request.action = "grasp"
                        
                    # Prepare payload
                    payload = {
                        "objects_to_spawn": objects_to_spawn,
                        "fundamental_actions": {
                            request.action: {
                                "is_trajectory": False,
                                "coordinates": coordinates,
                                "csv_file": None,
                                "gripper_mode_post_action": 2,
                                "grasp_mode": "HORIZONTAL"
                            }
                        }
                    }
                    
                    print(f"[DEBUG] Sending payload to cobot: {payload}")
                    client.send_trajectory_data(payload)
                    print("[DEBUG] Successfully sent trajectory data to cobot")
                
                else:
                    print(f"[ERROR] No object centers detected for action '{request.action}'")
                    raise HTTPException(
                        status_code=500,
                        detail="Failed to detect object centers"
                    )
                
            except Exception as e:
                print(f"[ERROR] Failed to communicate with cobot: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to execute action on cobot: {str(e)}"
                )

            # Prepare response after grasp action
            return ExecutionResponse(
                status="success",
                message="Grasping action successfully executed",
                action=request.action,
                object_locations=[]  # Empty as no object locations are involved here
            )

        # Handle general actions
        if request.action not in action_names:
            print(f"[ERROR] Action '{request.action}' not found in knowledge base")
            raise HTTPException(
                status_code=404,
                detail=f"Action '{request.action}' not found in knowledge base"
            )
        
        # Step 2: Get object centers
        print("[DEBUG] Getting object centers")
        objects_json = json.dumps(request.objects)
        try:
            centers_response = await object_centers(objects=objects_json)
            print(f"[DEBUG] Received object centers: {centers_response.centers}")
        except Exception as e:
            print(f"[ERROR] Failed to get object centers: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to detect object locations: {str(e)}"
            )

        # Step 3: Send to cobot client
        try:
            print("[DEBUG] Connecting to cobot client")
            client = CobotClient(ip=config.get("Cobot_URL"))
            
            # Assuming centers_response.centers is available
            # Here we prepare coordinates (including rotation)
            if hasattr(centers_response, 'centers') and centers_response.centers:
                coordinates = list(centers_response.centers[0]) + [90, -90, 180]
                
                # Prepare objects_to_spawn dictionary
                objects_to_spawn = {
                    obj.replace(" ", "_"): list(centers_response.centers[0])
                    for obj in request.objects
                }
                
                # Prepare payload
                payload = {
                    "objects_to_spawn": objects_to_spawn,
                    "fundamental_actions": {
                        request.action: {
                            "is_trajectory": False,
                            "coordinates": coordinates,
                            "csv_file": None,
                            "gripper_mode_post_action": 2,
                            "grasp_mode": "HORIZONTAL"
                        }
                    }
                }
                
                print(f"[DEBUG] Sending payload to cobot: {payload}")
                client.send_trajectory_data(payload)
                print("[DEBUG] Successfully sent trajectory data to cobot")
            
            else:
                print(f"[ERROR] No object centers detected for action '{request.action}'")
                raise HTTPException(
                    status_code=500,
                    detail="Failed to detect object centers"
                )
            
        except Exception as e:
            print(f"[ERROR] Failed to communicate with cobot: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to execute action on cobot: {str(e)}"
            )

        # Example response for non-grasp actions
        return ExecutionResponse(
            status="success",
            message="Action successfully executed",
            action=request.action,
            object_locations=[]  # Empty list as object locations aren't specified
        )
    
    except Exception as e:
        print(f"[ERROR] Error during execution: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during action execution"
        )
#-----------------------------------------------------------------------#
class HammerDataResponse(BaseModel):
    """
    Response model for the hammer data sending API.

    This model structures the response when sending data to the hammer function.

    Attributes:
        status (str): The status message about data processing and learning readiness.
        samples_count (int): Number of available samples for the action.
        action (str): The action name being processed.
    """
    status: str
    samples_count: int
    action: str

@app.post("/send_hammer_data", response_model=HammerDataResponse)
async def send_data_hammer(objects: List[str], action: str):
    """
    Endpoint to send data to hammer function for processing.

    Processes the latest recording of an action and checks if enough samples
    are available for learning.

    Args:
        objects (List[str]): List of objects involved in the action
        action (str): Name of the action being processed

    Returns:
        HammerDataResponse: Status of data processing and learning readiness
    """
    config_path = '../RAIT/config/config.yaml'
    preprocessor = VideoProcessor4D(config_path)
    preprocessor.set_details(objects, action)
    num_samples = preprocessor.check_available_samples(action)
    preprocessor.process_latest_recording(action)

    status = ("Ready for learning" if num_samples >= 30 
             else f"Insufficient samples - need at least 30 (current: {num_samples})")

    return HammerDataResponse(
        status=status,
        samples_count=num_samples,
        action=action
    )

        
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5002,
        ssl_certfile="/etc/letsencrypt/live/techolution.ddns.net/fullchain.pem",
        ssl_keyfile="/etc/letsencrypt/live/techolution.ddns.net/privkey.pem"
    )



