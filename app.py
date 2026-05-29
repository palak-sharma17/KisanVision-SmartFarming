import os
import shutil
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from predict import CropDiseasePredictor

app = FastAPI(title="KisanVision AI - Smart Farming Platform")

# Ensure static and upload directories exist
UPLOAD_DIR = "static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize Predictor Engine
predictor = CropDiseasePredictor()

# In-memory session tracking for the Dashboard metrics
db_history = [
    {"date": "2026-05-25", "crop": "Tomato", "status": "Diseased", "issue": "Bacterial Spot"},
    {"date": "2026-05-27", "crop": "Potato", "status": "Healthy", "issue": "None"},
]

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # Simulated weather forecast payload
    weather_data = {
        "temp": "31°C",
        "humidity": "62%",
        "condition": "Partly Cloudy",
        "location": "Punjab, Region-4"
    }
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "weather": weather_data,
        "history": db_history,
        "history_count": len(db_history)
    })

@app.post("/analyze")
async def analyze_crop(file: UploadFile = File(...)):
    try:
        # Save file safely to disk
        file_location = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Run through prediction framework
        analysis = predictor.predict(file_location)
        
        # Log to runtime dashboard memory
        crop_type = analysis["crop_disease"].split(" - ")[0]
        db_history.append({
            "date": datetime.today().strftime('%Y-%m-%d'),
            "crop": crop_type,
            "status": analysis["status"],
            "issue": analysis["crop_disease"].split(" - ")[1] if " - " in analysis["crop_disease"] else "None"
        })
        
        # Add relative path for preview rendering
        analysis["image_url"] = f"/{file_location}"
        return JSONResponse(content=analysis)
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

   