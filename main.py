"""
KisanVision AI - Smart Farming Platform
FastAPI Backend
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from datetime import datetime
import shutil
import uuid

from models import init_db, get_db, PredictionHistory, Farmer, FarmActivity, WeatherLog, CropData
from disease_detector import predict_disease, DISEASE_CLASSES, FERTILIZER_RECOMMENDATIONS
from weather_engine import get_weather_data

# Initialize app
app = FastAPI(
    title="KisanVision AI",
    description="AI-Powered Smart Farming & Crop Disease Detection Platform",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files and templates
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
app.mount("/uploads", StaticFiles(directory=os.path.join(BASE_DIR, "uploads")), name="uploads")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Ensure uploads dir exists
os.makedirs(os.path.join(BASE_DIR, "uploads"), exist_ok=True)

# Initialize database
init_db()


# ─────────────────────────────────────────
# PAGE ROUTES
# ─────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/detect", response_class=HTMLResponse)
async def detect_page(request: Request):
    return templates.TemplateResponse("detect.html", {"request": request})

@app.get("/weather", response_class=HTMLResponse)
async def weather_page(request: Request):
    return templates.TemplateResponse("weather.html", {"request": request})

@app.get("/history", response_class=HTMLResponse)
async def history_page(request: Request):
    return templates.TemplateResponse("history.html", {"request": request})


# ─────────────────────────────────────────
# API ROUTES
# ─────────────────────────────────────────

@app.post("/api/predict")
async def predict_crop_disease(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload crop image and get AI disease prediction"""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are allowed")
    
    # Save uploaded file
    file_ext = file.filename.split(".")[-1] if "." in file.filename else "jpg"
    filename = f"{uuid.uuid4().hex}.{file_ext}"
    upload_path = os.path.join(BASE_DIR, "uploads", filename)
    
    image_bytes = await file.read()
    with open(upload_path, "wb") as f:
        f.write(image_bytes)
    
    # Run AI prediction
    result = predict_disease(image_bytes)
    
    # Save to database
    prediction = PredictionHistory(
        image_path=f"/uploads/{filename}",
        crop_type=result["crop_type"],
        disease_detected=result["disease_detected"],
        confidence_score=result["confidence_score"],
        treatment_recommendation=result["treatment"]["immediate_action"],
        severity=result["severity"]
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)
    
    result["prediction_id"] = prediction.id
    result["image_url"] = f"/uploads/{filename}"
    result["timestamp"] = datetime.now().strftime("%d %b %Y, %I:%M %p")
    
    return JSONResponse(content=result)


@app.get("/api/weather")
async def get_weather(location: str = "Jaipur", db: Session = Depends(get_db)):
    """Get weather data with farming recommendations"""
    weather = get_weather_data(location)
    
    # Log weather
    log = WeatherLog(
        location=location,
        temperature=weather["temperature"],
        humidity=weather["humidity"],
        rainfall=weather["rainfall"],
        wind_speed=weather["wind_speed"],
        condition=weather["condition"],
        irrigation_advice=weather["irrigation_advice"]["message"]
    )
    db.add(log)
    db.commit()
    
    return JSONResponse(content=weather)


@app.get("/api/history")
async def get_history(limit: int = 20, db: Session = Depends(get_db)):
    """Get prediction history"""
    records = db.query(PredictionHistory).order_by(
        PredictionHistory.created_at.desc()
    ).limit(limit).all()
    
    result = []
    for r in records:
        result.append({
            "id": r.id,
            "crop_type": r.crop_type,
            "disease_detected": r.disease_detected,
            "confidence_score": r.confidence_score,
            "severity": r.severity,
            "treatment": r.treatment_recommendation,
            "image_path": r.image_path,
            "created_at": r.created_at.strftime("%d %b %Y, %I:%M %p") if r.created_at else "N/A"
        })
    
    return JSONResponse(content={"history": result, "total": len(result)})


@app.get("/api/dashboard/stats")
async def get_dashboard_stats(db: Session = Depends(get_db)):
    """Get dashboard statistics"""
    total_predictions = db.query(PredictionHistory).count()
    
    disease_counts = {}
    crop_counts = {}
    severity_counts = {"None": 0, "Low": 0, "Moderate": 0, "High": 0, "Critical": 0}
    
    predictions = db.query(PredictionHistory).all()
    for p in predictions:
        disease_counts[p.disease_detected] = disease_counts.get(p.disease_detected, 0) + 1
        crop_counts[p.crop_type] = crop_counts.get(p.crop_type, 0) + 1
        if p.severity in severity_counts:
            severity_counts[p.severity] += 1
    
    healthy_count = disease_counts.get("Healthy", 0)
    diseased_count = total_predictions - healthy_count
    health_rate = round((healthy_count / total_predictions * 100) if total_predictions > 0 else 0, 1)
    
    top_diseases = sorted(disease_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    return JSONResponse(content={
        "total_predictions": total_predictions,
        "healthy_crops": healthy_count,
        "diseased_crops": diseased_count,
        "health_rate": health_rate,
        "top_diseases": [{"name": d, "count": c} for d, c in top_diseases],
        "crop_distribution": [{"crop": k, "count": v} for k, v in crop_counts.items()],
        "severity_distribution": severity_counts
    })


@app.get("/api/crops")
async def get_crop_diseases(crop: str = None, db: Session = Depends(get_db)):
    """Get crop disease database"""
    query = db.query(CropData)
    if crop:
        query = query.filter(CropData.crop_name.ilike(f"%{crop}%"))
    
    crops = query.all()
    return JSONResponse(content={
        "crops": [{
            "id": c.id,
            "crop_name": c.crop_name,
            "disease_name": c.disease_name,
            "symptoms": c.symptoms,
            "treatment": c.treatment,
            "prevention": c.prevention,
            "season": c.season
        } for c in crops]
    })


@app.get("/api/fertilizer/{crop_name}")
async def get_fertilizer_recommendation(crop_name: str):
    """Get fertilizer recommendation for a crop"""
    rec = FERTILIZER_RECOMMENDATIONS.get(crop_name, None)
    if not rec:
        # Default recommendation
        rec = {"N": "80 kg/ha", "P": "40 kg/ha", "K": "40 kg/ha", "tip": "Apply balanced NPK as per soil test report"}
    return JSONResponse(content={"crop": crop_name, "recommendation": rec})


@app.post("/api/activity")
async def add_farm_activity(
    activity_type: str = Form(...),
    crop: str = Form(...),
    notes: str = Form(""),
    scheduled_date: str = Form(""),
    db: Session = Depends(get_db)
):
    activity = FarmActivity(
        activity_type=activity_type,
        crop=crop,
        notes=notes,
        scheduled_date=scheduled_date
    )
    db.add(activity)
    db.commit()
    return JSONResponse(content={"message": "Activity added successfully", "id": activity.id})


@app.get("/api/activities")
async def get_activities(db: Session = Depends(get_db)):
    activities = db.query(FarmActivity).order_by(FarmActivity.created_at.desc()).limit(20).all()
    return JSONResponse(content={
        "activities": [{
            "id": a.id,
            "activity_type": a.activity_type,
            "crop": a.crop,
            "notes": a.notes,
            "scheduled_date": a.scheduled_date,
            "completed": a.completed,
            "created_at": a.created_at.strftime("%d %b %Y") if a.created_at else "N/A"
        } for a in activities]
    })


@app.get("/health")
async def health_check():
    return {"status": "running", "platform": "KisanVision AI v1.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)