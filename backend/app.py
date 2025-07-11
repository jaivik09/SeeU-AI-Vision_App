import base64
import tempfile
from typing import List, Dict, Any

import io
from TTS.api import TTS

import cv2
import numpy as np
import requests
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from ultralytics import YOLO

app = FastAPI()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model initialization
model = YOLO('besttrain.pt').to(device)
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(device)
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform_midas = midas_transforms.small_transform

# Initialize TTS only once at startup
#tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")

# Constants
CALIBRATION_FACTOR = 114


class ImageRequest(BaseModel):
    image: str


class LLMRequest(BaseModel):
    query: str
    detections: List[Dict[str, Any]]
    image: str  # Now including the image data

class TTSRequest(BaseModel):
    text: str
    language: str = "hi"  # default to English
    speaker_wav: str = None  # optional base64-encoded audio for cloning

def detect_objects(image: np.ndarray) -> List[Dict[str, Any]]:
    """Detect objects in an image using YOLO model."""
    results = model(image, conf=0.3)
    detections = []
    print(f"Raw detections before filter: {len(results[0].boxes)}")
    
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            print(f"Object: {result.names[class_id]}, Confidence: {confidence}, BBox: {box.xyxy[0].tolist()}")
            
            if confidence > 0.3:
                object_name = result.names[class_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({
                    "object": object_name,
                    "confidence": round(confidence, 2),
                    "bbox": [x1, y1, x2, y2],
                })
    
    print(f"Filtered detections: {len(detections)}")
    return detections


def estimate_distance(image: np.ndarray, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Estimate distance to detected objects using MiDaS depth estimation."""
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_batch = transform_midas(img_rgb).to(device)
    
    with torch.no_grad():
        prediction = midas(input_batch)
        depth_map = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()

    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        depth_region = depth_map[y1:y2, x1:x2]
        avg_depth = np.mean(depth_region).item() if depth_region.size > 0 else 0
        
        if avg_depth > 0:
            detection["distance"] = round(CALIBRATION_FACTOR / avg_depth, 2)
        else:
            detection["distance"] = None
    
    return detections


@app.post("/detect/")
async def detect(request: ImageRequest):
    """Endpoint for object detection and distance estimation."""
    try:
        image_data = base64.b64decode(request.image)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return {"error": "Invalid image format"}

        detections = detect_objects(image)
        if detections:
            detections = estimate_distance(image, detections)

        return {"detections": detections}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")


@app.post("/ask_llm/")
async def ask_llm(request: LLMRequest):
    """Endpoint for querying the LLM with detection context."""
    try:
        base64_image = request.image
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid image data: {str(e)}"}
        )
    
    # Build more detailed context from detection data
    detection_texts = [
        f"{d['object']} detected at {d['distance']}m away from camera" if d['distance'] 
        else f"{d['object']} detected (distance unknown)"
        for d in request.detections
    ]
    
    # More structured context with clear instructions to focus on detections
    context = "IMPORTANT: Focus on these detected objects in your response: " + ", ".join(detection_texts) + "."
    
    print(f"Sending to LLM: {context} {request.query}")

    # Check if this is a distance query about a specific object
    is_distance_query = any(word in request.query.lower() for word in ["far", "distance", "away", "how far"])
    
    # Find which objects are being asked about
    queried_objects = []
    for detection in request.detections:
        obj_name = detection['object']
        if obj_name.lower() in request.query.lower():
            queried_objects.append(detection)
    
    # If it's a direct distance query about specific detected objects, handle it directly
    if is_distance_query and queried_objects:
        # Just use the direct approach for distance queries
        for detection in queried_objects:
            obj_name = detection['object']
            distance = detection.get('distance')
            if distance:
                return {"answer": f"Your {obj_name} is {distance}m away from you."}
    
    # For more complex queries or when not asking specifically about distance,
    # proceed with the LLM approach but with enhanced instructions
    
    # Updated system prompt to be more directive
    llm_payload = {
        "model": "llava-v1.5-7b",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a vision assistant for visually impaired users. "
                    "FOCUS PRIMARILY ON THE DETECTED OBJECTS AND THEIR DISTANCES listed in the user's query. "
                    "Your primary job is to describe the location and distance of objects the user asks about. "
                    "ALWAYS INCLUDE THE EXACT NUMERICAL DISTANCE MEASUREMENT when responding about an object's distance. "
                    #"Ignore any code, text, or other content in the image unless specifically asked about it. "
                    "Be concise and directly answer questions about the detected objects and distances. "
                    #"Format of your response for distance queries MUST be: 'The [object] is [X.XX]m away from you.'"
                )
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{context} {request.query}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        "max_tokens": 150
    }

    try:
        response = requests.post(
            "http://localhost:1234/v1/chat/completions",
            json=llm_payload,
            timeout=15
        )
        response.raise_for_status()
        llm_data = response.json()
        llm_answer = llm_data["choices"][0]["message"]["content"]
        print(f"LLM Response: {llm_answer}")
        
        # Post-process the LLM answer for distance queries to ensure distance is included
        if is_distance_query and queried_objects:
            # Check if the LLM response already includes distance information
            has_distance = any(f"{detection['distance']}m" in llm_answer for detection in queried_objects if detection.get('distance'))
            
            if not has_distance:
                # If LLM didn't include the distance, append it
                for detection in queried_objects:
                    obj_name = detection['object']
                    distance = detection.get('distance')
                    if distance:
                        llm_answer += f" Your {obj_name} is {distance}m away from you."
        
        return {"answer": llm_answer}
    
    except requests.exceptions.RequestException as e:
        print(f"LLM Error with image: {str(e)}")
        
        # Create a fallback response based on detections only
        if request.detections:
            # Handle direct distance queries first
            if is_distance_query:
                for detection in request.detections:
                    obj_name = detection['object']
                    distance = detection.get('distance')
                    if obj_name.lower() in request.query.lower() and distance:
                        return {"answer": f"Your {obj_name} is {distance}m away from you."}
            
            # General fallback for other queries
            simplified_answer = f"I can see {len(request.detections)} objects. "
            for detection in request.detections:
                obj_name = detection['object']
                distance = detection.get('distance')
                if obj_name in request.query.lower():
                    distance_text = f"{distance}m away" if distance else "at an unknown distance"
                    return {"answer": f"I can see a {obj_name} {distance_text}."}
            
            return {"answer": simplified_answer}
        else:
            return {"answer": "I don't see any objects that match your question."}
            
    #         # Text-only fallback approach
    #         text_only_payload = {
    #             "model": "llava-v1.5-7b",
    #             "messages": [
    #                 {
    #                     "role": "system",
    #                     "content": "You are a vision assistant to help the visually impaired users. I will describe what the camera detects, including objects and their distances. Keep it warm, casual, and positive—like talking to a good friend. If you don't have enough info, just roll with it and say something nice anyway! Keep it small and simple don't explain more"
    #                 },
    #                 {
    #                     "role": "user",
    #                     "content": f"{context} {request.query}"
    #                 }
    #             ],
    #             "max_tokens": 150
    #         }
            
    #         fallback_response = requests.post(
    #             "http://localhost:1234/v1/chat/completions",
    #             json=text_only_payload,
    #             timeout=15
    #         )
    #         fallback_response.raise_for_status()
    #         fallback_data = fallback_response.json()
    #         fallback_answer = fallback_data["choices"][0]["message"]["content"]
    #         return {"answer": fallback_answer}
    # except Exception as e:
    #     print(f"Error in ask_llm: {str(e)}")
    #     raise HTTPException(status_code=500, detail=f"LLM failed: {str(e)}")

@app.post("/text_to_speech/")
async def text_to_speech(request: TTSRequest):
    """Generate speech from text using XTTS-v2 with optional voice cloning and language support."""
    try:
        speaker_path = None
        if request.speaker_wav:
            # Decode base64 speaker audio and save to temp file
            decoded_speaker = base64.b64decode(request.speaker_wav)
            temp_speaker_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            temp_speaker_file.write(decoded_speaker)
            temp_speaker_file.close()
            speaker_path = temp_speaker_file.name

        # Use another temp file for output
        output_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name

        # Generate speech using XTTS (multi-lingual)
        tts.tts_to_file(
            text=request.text,
            speaker_wav=speaker_path,
            language=request.language,  # "hi" for Hindi
            file_path=output_audio_file
        )

        # Read generated audio
        with open(output_audio_file, "rb") as f:
            audio_data = f.read()
        audio_base64 = base64.b64encode(audio_data).decode("utf-8")

        return {"audio": audio_base64, "format": "wav"}

    except Exception as e:
        print(f"TTS Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")
        
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)