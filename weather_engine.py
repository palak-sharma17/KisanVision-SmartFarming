"""
KisanVision AI - Weather Engine
Fetches real weather via Open-Meteo (free, no API key needed)
and generates farm-specific irrigation advice.
"""
import httpx
from datetime import datetime, timedelta
from typing import Optional
import math

# Geocoding map for common Indian farming cities
CITY_COORDS = {
    "jaipur":     (26.9124, 75.7873),
    "delhi":      (28.6139, 77.2090),
    "mumbai":     (19.0760, 72.8777),
    "pune":       (18.5204, 73.8567),
    "nagpur":     (21.1458, 79.0882),
    "lucknow":    (26.8467, 80.9462),
    "chandigarh": (30.7333, 76.7794),
    "hyderabad":  (17.3850, 78.4867),
    "bangalore":  (12.9716, 77.5946),
    "chennai":    (13.0827, 80.2707),
    "kolkata":    (22.5726, 88.3639),
    "ahmedabad":  (23.0225, 72.5714),
    "bhopal":     (23.2599, 77.4126),
    "patna":      (25.5941, 85.1376),
    "indore":     (22.7196, 75.8577),
    "varanasi":   (25.3176, 82.9739),
    "amritsar":   (31.6340, 74.8723),
    "jodhpur":    (26.2389, 73.0243),
    "kota":       (25.2138, 75.8648),
    "udaipur":    (24.5854, 73.7125),
}

WMO_CONDITIONS = {
    0: ("Clear Sky", "☀️"), 1: ("Mainly Clear", "🌤️"), 2: ("Partly Cloudy", "⛅"),
    3: ("Overcast", "☁️"), 45: ("Foggy", "🌫️"), 48: ("Icy Fog", "🌫️"),
    51: ("Light Drizzle", "🌦️"), 53: ("Drizzle", "🌦️"), 55: ("Heavy Drizzle", "🌧️"),
    61: ("Slight Rain", "🌧️"), 63: ("Moderate Rain", "🌧️"), 65: ("Heavy Rain", "🌧️"),
    71: ("Slight Snowfall", "🌨️"), 73: ("Moderate Snow", "❄️"), 75: ("Heavy Snow", "❄️"),
    80: ("Slight Showers", "🌦️"), 81: ("Moderate Showers", "🌧️"), 82: ("Violent Showers", "⛈️"),
    95: ("Thunderstorm", "⛈️"), 96: ("Thunderstorm + Hail", "⛈️"), 99: ("Thunderstorm + Hail", "⛈️"),
}


def _geocode(location: str):
    loc = location.strip().lower()
    if loc in CITY_COORDS:
        return CITY_COORDS[loc]
    # Try Open-Meteo geocoding
    try:
        r = httpx.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": location, "count": 1, "language": "en", "format": "json"},
            timeout=8.0
        )
        data = r.json()
        if data.get("results"):
            res = data["results"][0]
            return (res["latitude"], res["longitude"])
    except Exception:
        pass
    return CITY_COORDS["jaipur"]  # Default


def get_weather_data(location: str = "Jaipur") -> dict:
    lat, lon = _geocode(location)
    today = datetime.now()
    

    try:
        r = httpx.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": [
                    "temperature_2m", "relative_humidity_2m", "apparent_temperature",
                    "precipitation", "wind_speed_10m", "wind_direction_10m",
                    "weather_code", "uv_index", "surface_pressure"
                ],
                "daily": [
                    "temperature_2m_max", "temperature_2m_min", "precipitation_sum",
                    "weather_code", "wind_speed_10m_max", "precipitation_probability_max",
                    "relative_humidity_2m_max"
                ],
                "timezone": "Asia/Kolkata",
                "forecast_days": 7
            },
            timeout=10.0
        )
        r.raise_for_status()
        raw = r.json()
        cur = raw["current"]
        daily = raw["daily"]

        temp = cur["temperature_2m"]
        humidity = cur["relative_humidity_2m"]
        rain = cur["precipitation"]
        wind = cur["wind_speed_10m"]
        feels = cur["apparent_temperature"]
        wcode = cur["weather_code"]
        uv = cur.get("uv_index", 0)
        pressure = cur.get("surface_pressure", 1013)

        cond_label, cond_icon = WMO_CONDITIONS.get(wcode, ("Unknown", "🌡️"))

        # Irrigation advice logic
        irr = _irrigation_advice(temp, humidity, rain, daily)

        # Farm alerts
        alerts = _farm_alerts(temp, wind, rain, humidity, uv, daily)

        # 5-day forecast
        forecast = []
        for i in range(min(5, len(daily["time"]))):
            fc_code = daily["weather_code"][i]
            fc_label, fc_icon = WMO_CONDITIONS.get(fc_code, ("—", "🌡️"))
            forecast.append({
                "date": _fmt_date(daily["time"][i]),
                "condition": fc_label,
                "icon": fc_icon,
                "temp_max": round(daily["temperature_2m_max"][i], 1),
                "temp_min": round(daily["temperature_2m_min"][i], 1),
                "rainfall": round(daily["precipitation_sum"][i], 1),
                "rain_prob": daily["precipitation_probability_max"][i],
                "humidity": daily["relative_humidity_2m_max"][i],
                "wind_max": round(daily["wind_speed_10m_max"][i], 1),
            })

        return {
            "location": location.title(),
            "temperature": round(temp, 1),
            "feels_like": round(feels, 1),
            "humidity": humidity,
            "rainfall": round(rain, 1),
            "wind_speed": round(wind, 1),
            "wind_dir": _wind_direction(cur.get("wind_direction_10m", 0)),
            "condition": cond_label,
            "condition_icon": cond_icon,
            "uv_index": round(uv, 1),
            "pressure": round(pressure, 0),
            "updated": today.strftime("%d %b %Y, %I:%M %p"),
            "irrigation_advice": irr,
            "farm_alerts": alerts,
            "forecast": forecast,
        }

    except Exception as e:
        return _fallback_weather(location, str(e))


def _irrigation_advice(temp, humidity, rain, daily) -> dict:
    tomorrow_rain = daily["precipitation_sum"][1] if len(daily["precipitation_sum"]) > 1 else 0

    if rain > 10:
        return {"status": "Skip Irrigation", "message": "Adequate rainfall today. Skipping irrigation will save water and prevent root rot.", "urgency": "low", "water_needed": "0 mm", "next_irrigation": "Check in 2–3 days", "color": "blue"}
    if tomorrow_rain > 8:
        return {"status": "Postpone Irrigation", "message": "Heavy rain forecast tomorrow. Hold irrigation to avoid waterlogging.", "urgency": "low", "water_needed": "0 mm", "next_irrigation": "After rainfall assessment", "color": "cyan"}
    if temp > 38:
        return {"status": "Urgent Irrigation Needed", "message": "High heat stress. Irrigate in the early morning (5–7 AM) or evening (6–8 PM) to minimise evaporation.", "urgency": "high", "water_needed": "35–45 mm", "next_irrigation": "Today — early morning", "color": "red"}
    if humidity < 40:
        return {"status": "Irrigation Recommended", "message": "Low humidity causing moisture stress. Deep irrigation advised to support root development.", "urgency": "medium", "water_needed": "25–35 mm", "next_irrigation": "Within 24 hours", "color": "orange"}
    return {"status": "Normal Irrigation Schedule", "message": "Conditions are favourable. Maintain regular irrigation schedule as per crop stage.", "urgency": "normal", "water_needed": "20–25 mm", "next_irrigation": "As per schedule", "color": "green"}


def _farm_alerts(temp, wind, rain, humidity, uv, daily) -> list:
    alerts = []
    if temp > 40:
        alerts.append({"type": "Heat Wave", "level": "Critical", "icon": "🌡️", "message": "Extreme heat. Apply mulching, shade nets; irrigate in early morning.", "color": "red"})
    elif temp > 35:
        alerts.append({"type": "High Temperature", "level": "Warning", "icon": "🔆", "message": "Heat stress possible. Monitor crops and increase irrigation frequency.", "color": "orange"})
    if wind > 50:
        alerts.append({"type": "Strong Winds", "level": "Warning", "icon": "💨", "message": "Damaging winds. Stake tall crops; avoid spraying pesticides today.", "color": "orange"})
    if rain > 20:
        alerts.append({"type": "Heavy Rainfall", "level": "Alert", "icon": "🌧️", "message": "Risk of waterlogging and fungal diseases. Ensure field drainage is clear.", "color": "blue"})
    if humidity > 85:
        alerts.append({"type": "High Humidity", "level": "Caution", "icon": "💧", "message": "Favourable conditions for fungal diseases. Preventive fungicide spray recommended.", "color": "blue"})
    if uv > 8:
        alerts.append({"type": "High UV Index", "level": "Info", "icon": "☀️", "message": "High UV. Avoid spraying chemicals between 10 AM–4 PM to prevent phytotoxicity.", "color": "orange"})
    if not alerts:
        alerts.append({"type": "All Clear", "level": "Normal", "icon": "✅", "message": "No adverse weather alerts. Good conditions for farm operations.", "color": "green"})
    return alerts


def _wind_direction(deg: float) -> str:
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return dirs[round(deg / 45) % 8]


def _fmt_date(date_str: str) -> str:
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.strftime("%a, %d %b")
    except Exception:
        return date_str


def _fallback_weather(location: str, error: str) -> dict:
    return {
        "location": location.title(),
        "temperature": 32.0,
        "feels_like": 35.0,
        "humidity": 60,
        "rainfall": 0.0,
        "wind_speed": 12.0,
        "wind_dir": "NW",
        "condition": "Partly Cloudy",
        "condition_icon": "⛅",
        "uv_index": 6.0,
        "pressure": 1010.0,
        "updated": datetime.now().strftime("%d %b %Y, %I:%M %p"),
        "irrigation_advice": {"status": "Data Unavailable", "message": "Weather API unreachable. Please check manually.", "urgency": "normal", "water_needed": "—", "next_irrigation": "—", "color": "green"},
        "farm_alerts": [{"type": "Weather Service", "level": "Info", "icon": "⚠️", "message": f"Could not fetch live data: {error[:80]}", "color": "orange"}],
        "forecast": [],
        "_error": error
    }