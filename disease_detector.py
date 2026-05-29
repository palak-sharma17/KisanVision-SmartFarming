"""
KisanVision AI - Disease Detection Engine
Uses Claude Vision API for crop disease detection
"""
import base64
import json
import re
import httpx
from typing import Optional

DISEASE_CLASSES = [
    "Healthy", "Early Blight", "Late Blight", "Leaf Rust", "Powdery Mildew",
    "Bacterial Spot", "Septoria Leaf Spot", "Yellow Leaf Curl Virus",
    "Leaf Mold", "Target Spot", "Mosaic Virus", "Northern Leaf Blight",
    "Common Rust", "Gray Leaf Spot", "Brown Spot", "Blast",
    "Bacterial Blight", "Downy Mildew", "Anthracnose", "Cercospora Leaf Spot"
]

FERTILIZER_RECOMMENDATIONS = {
    "Tomato":  {"N": "120 kg/ha", "P": "60 kg/ha", "K": "80 kg/ha", "tip": "Split nitrogen into 3 doses; apply potassium at fruit set"},
    "Wheat":   {"N": "100 kg/ha", "P": "50 kg/ha", "K": "40 kg/ha", "tip": "Apply 50% N as basal + 50% at first irrigation"},
    "Rice":    {"N": "80 kg/ha",  "P": "40 kg/ha", "K": "40 kg/ha", "tip": "Apply in splits: basal, tillering, panicle initiation"},
    "Cotton":  {"N": "100 kg/ha", "P": "50 kg/ha", "K": "50 kg/ha", "tip": "Foliar spray of micronutrients at squaring stage"},
    "Maize":   {"N": "120 kg/ha", "P": "60 kg/ha", "K": "40 kg/ha", "tip": "Top dress urea at knee-high stage"},
    "Potato":  {"N": "150 kg/ha", "P": "80 kg/ha", "K": "100 kg/ha","tip": "High potassium improves tuber quality and storage"},
    "Mustard": {"N": "80 kg/ha",  "P": "40 kg/ha", "K": "30 kg/ha", "tip": "Apply sulphur 20 kg/ha for better oil content"},
    "Soybean": {"N": "25 kg/ha",  "P": "60 kg/ha", "K": "40 kg/ha", "tip": "Rhizobium seed inoculation reduces N requirement"},
    "Grape":   {"N": "100 kg/ha", "P": "50 kg/ha", "K": "80 kg/ha", "tip": "Foliar potassium spray improves berry quality"},
    "Sugarcane":{"N":"250 kg/ha", "P": "80 kg/ha", "K": "120 kg/ha","tip": "Apply in 4–5 splits over the growing season"},
}

CLAUDE_PROMPT = """You are an expert agricultural AI system. Analyze this plant/crop image and provide a detailed disease diagnosis.

Return ONLY a valid JSON object with this exact structure (no markdown, no extra text):
{
  "crop_type": "crop name (e.g. Tomato, Wheat, Rice, Potato, or Unknown)",
  "disease_detected": "disease name or Healthy",
  "confidence_score": 0.0 to 1.0,
  "severity": "None | Low | Moderate | High | Critical",
  "is_healthy": true or false,
  "symptoms_observed": ["symptom1", "symptom2"],
  "top_predictions": [
    {"disease": "name", "confidence": 0.0},
    {"disease": "name2", "confidence": 0.0},
    {"disease": "name3", "confidence": 0.0}
  ],
  "treatment": {
    "immediate_action": "what to do right now",
    "chemical_control": "recommended pesticide/fungicide with dosage",
    "organic_alternative": "organic/biological option",
    "preventive_measures": "how to prevent in future"
  },
  "fertilizer": {
    "N": "kg/ha value",
    "P": "kg/ha value",
    "K": "kg/ha value",
    "tip": "application tip"
  }
}

Be precise. If you cannot identify the crop, use "Unknown". Severity None means healthy."""


def predict_disease(image_bytes: bytes) -> dict:
    """Run AI disease prediction on crop image using Claude Vision"""
    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

    # Detect media type
    media_type = "image/jpeg"
    if image_bytes[:4] == b'\x89PNG':
        media_type = "image/png"
    elif image_bytes[:4] in (b'GIF8', b'GIF9'):
        media_type = "image/gif"
    elif image_bytes[:2] == b'BM':
        media_type = "image/bmp"

    try:
        response = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={"Content-Type": "application/json"},
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1000,
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": b64
                            }
                        },
                        {"type": "text", "text": CLAUDE_PROMPT}
                    ]
                }]
            },
            timeout=30.0
        )

        response.raise_for_status()
        data = response.json()
        raw = data["content"][0]["text"].strip()

        # Strip any markdown fences
        raw = re.sub(r"```json|```", "", raw).strip()
        result = json.loads(raw)

        # Enrich with fertilizer lookup if crop known
        crop = result.get("crop_type", "Unknown")
        if crop in FERTILIZER_RECOMMENDATIONS and not result.get("fertilizer"):
            result["fertilizer"] = FERTILIZER_RECOMMENDATIONS[crop]

        return _normalise(result)

    except Exception as e:
        return _fallback(str(e))


def _normalise(r: dict) -> dict:
    """Ensure all expected keys exist"""
    r.setdefault("crop_type", "Unknown")
    r.setdefault("disease_detected", "Unknown")
    r.setdefault("confidence_score", 0.5)
    r.setdefault("severity", "Moderate")
    r.setdefault("is_healthy", r["disease_detected"].lower() == "healthy")
    r.setdefault("symptoms_observed", [])
    r.setdefault("top_predictions", [])
    r.setdefault("treatment", {
        "immediate_action": "Consult local agriculture extension officer",
        "chemical_control": "Apply broad-spectrum fungicide as precaution",
        "organic_alternative": "Neem oil spray (3 ml/L water)",
        "preventive_measures": "Maintain field hygiene and proper spacing"
    })
    r.setdefault("fertilizer", {"N": "80 kg/ha", "P": "40 kg/ha", "K": "40 kg/ha", "tip": "Apply as per soil test"})
    return r


def _fallback(error: str) -> dict:
    """Return a safe fallback if AI call fails"""
    return {
        "crop_type": "Unknown",
        "disease_detected": "Analysis Failed",
        "confidence_score": 0.0,
        "severity": "Unknown",
        "is_healthy": False,
        "symptoms_observed": ["Could not process image"],
        "top_predictions": [],
        "treatment": {
            "immediate_action": "Please try again with a clearer image",
            "chemical_control": "N/A",
            "organic_alternative": "N/A",
            "preventive_measures": "Ensure image shows plant leaves clearly"
        },
        "fertilizer": {"N": "—", "P": "—", "K": "—", "tip": f"Error: {error[:100]}"},
        "_error": error
    }