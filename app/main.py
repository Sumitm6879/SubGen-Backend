import os
import uuid
import shutil
import asyncio
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.processing import process_video_logic

# --- App Initialization and Configuration ---
app = FastAPI(title="AI Media Accessibility API")

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Mount the 'outputs' directory to serve static files
app.mount("/outputs", StaticFiles(directory=OUTPUT_FOLDER), name="outputs")

# Configure CORS to allow the React frontend to connect
origins = [
    "http://localhost:5173",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store for job details. For production we can use a database like Redis.
jobs = {}

# Endpoint for streaming the process
@app.post("/upload")
async def upload_for_processing(
    video: UploadFile = File(...),
    censor: bool = Form(False),
    show_confidence: bool = Form(False),
    diarize_srt : bool = Form(False),
    analyze_bias: bool = Form(False),
    translate_to: str = Form("None")
):
    """
    Step 1: Uploads the video and options, creates a job, and returns a job ID.
    The frontend will call this first.
    """
    job_id = str(uuid.uuid4())
    video_extension = os.path.splitext(video.filename)[1]
    video_path = os.path.join(UPLOAD_FOLDER, f"{job_id}{video_extension}")

    try:
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
    finally:
        video.file.close()

    options = {
        "censor": censor,
        "show_confidence": show_confidence,
        "diarize_srt": diarize_srt,
        "analyze_bias": analyze_bias,
        "translate_to": translate_to if translate_to != "None" else None,
    }

    # Store job details for the streaming endpoint to use
    jobs[job_id] = {"video_path": video_path, "options": options}

    return JSONResponse(content={"job_id": job_id})

@app.get("/stream/{job_id}")
async def stream_processing(job_id: str, request: Request):
    """
    Step 2: Processes the video for the given job_id and streams progress updates
    back to the frontend using Server-Sent Events (SSE).
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found. It may have expired or been completed.")

    async def event_generator():
        video_path = jobs[job_id]["video_path"]
        options = jobs[job_id]["options"]
        base_url = str(request.base_url)[:-1] # Remove trailing slash

        # Loop through the yielded updates from the processing logic
        for update in process_video_logic(video_path, options, OUTPUT_FOLDER, job_id, base_url):
            yield f"data: {update}\n\n"
            await asyncio.sleep(0.1) # Small delay to ensure messages are sent

        # Clean up the job and file after processing is complete
        if os.path.exists(video_path):
            os.remove(video_path)
        del jobs[job_id]

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/")
def read_root():
    return {"message": "AI Media Accessibility API is running."}