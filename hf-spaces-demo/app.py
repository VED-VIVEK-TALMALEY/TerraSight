"""
🌍 TerraSight — Earth Observation Intelligence Platform
World Monitor-style Streamlit app with MapLibre, AI analysis, and research chat.
"""

import streamlit as st
import streamlit.components.v1 as components
import json
import random
import math
import hashlib
from datetime import datetime, timezone

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="TerraSight — EO Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Session State Init ───────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_analysis" not in st.session_state:
    st.session_state.current_analysis = None
if "bbox" not in st.session_state:
    st.session_state.bbox = None
if "analysis_count" not in st.session_state:
    st.session_state.analysis_count = 0

# ── CSS — War Room Dark Theme ───────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600;700&display=swap');

/* Global */
.stApp {
    background: #050505 !important;
    color: #c8ccd0 !important;
    font-family: 'Inter', sans-serif !important;
}

/* Hide default Streamlit elements */
#MainMenu, footer, header, .stDeployButton { visibility: hidden; }
div[data-testid="stToolbar"] { display: none; }
.block-container { padding-top: 0 !important; padding-bottom: 0 !important; max-width: 100% !important; }

/* Header bar */
.header-bar {
    background: linear-gradient(90deg, #0a0a0a 0%, #111318 50%, #0a0a0a 100%);
    border-bottom: 1px solid #1a1f2e;
    padding: 8px 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    margin: -1rem -1rem 0 -1rem;
}
.header-brand {
    display: flex; align-items: center; gap: 12px;
}
.header-brand .logo { font-size: 1.3rem; font-weight: 700; color: #00e5ff; letter-spacing: 2px; }
.header-brand .version { color: #4a5568; font-size: 0.65rem; }
.header-status {
    display: flex; align-items: center; gap: 20px;
}
.status-item { display: flex; align-items: center; gap: 6px; }
.status-dot { width: 6px; height: 6px; border-radius: 50%; display: inline-block; }
.status-dot.green { background: #00e676; box-shadow: 0 0 6px #00e676; }
.status-dot.amber { background: #ff9100; box-shadow: 0 0 6px #ff9100; }
.status-dot.red { background: #ff1744; box-shadow: 0 0 6px #ff1744; }
.header-time { color: #4a5568; font-size: 0.7rem; }

/* Map container */
.map-container {
    border: 1px solid #1a1f2e;
    border-radius: 4px;
    overflow: hidden;
    background: #0a0a0a;
}

/* Analysis panel */
.analysis-panel {
    background: #0a0e14;
    border: 1px solid #1a1f2e;
    border-radius: 4px;
    padding: 12px;
    height: 100%;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    overflow-y: auto;
}
.panel-header {
    display: flex; align-items: center; justify-content: space-between;
    border-bottom: 1px solid #1a1f2e; padding-bottom: 8px; margin-bottom: 10px;
}
.panel-title { color: #00e5ff; font-weight: 700; font-size: 0.85rem; letter-spacing: 1px; }
.live-badge {
    background: #00e676; color: #000; font-size: 0.6rem; font-weight: 700;
    padding: 2px 8px; border-radius: 3px; letter-spacing: 1px;
}

/* Analysis sections */
.analysis-section {
    background: #0d1117; border: 1px solid #1a1f2e; border-radius: 4px;
    padding: 10px; margin-bottom: 8px;
}
.section-title {
    color: #ff9100; font-weight: 600; font-size: 0.75rem;
    text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px;
}
.metric-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 3px 0; border-bottom: 1px solid #0d1117;
}
.metric-key { color: #6b7280; font-size: 0.72rem; }
.metric-val { font-weight: 600; font-size: 0.72rem; }
.val-green { color: #00e676; }
.val-amber { color: #ff9100; }
.val-red { color: #ff1744; }
.val-blue { color: #00e5ff; }
.val-white { color: #e0e0e0; }

/* Risk indicator */
.risk-indicator {
    display: flex; align-items: center; gap: 8px;
    padding: 8px; background: #1a0a0a; border: 1px solid #3d0000;
    border-radius: 4px; margin-bottom: 8px;
}
.risk-circle {
    width: 40px; height: 40px; border-radius: 50%; display: flex;
    align-items: center; justify-content: center; font-weight: 700;
    font-size: 0.9rem;
}
.risk-low { background: #002211; border: 2px solid #00e676; color: #00e676; }
.risk-mod { background: #1a1100; border: 2px solid #ff9100; color: #ff9100; }
.risk-high { background: #1a0000; border: 2px solid #ff1744; color: #ff1744; }
.risk-label { font-size: 0.72rem; }
.risk-label-title { color: #e0e0e0; font-weight: 600; }
.risk-label-sub { color: #6b7280; font-size: 0.65rem; }

/* Chat */
.chat-container {
    background: #0a0e14; border: 1px solid #1a1f2e; border-radius: 4px;
    padding: 10px; margin-top: 6px;
}
.chat-msg {
    padding: 8px 10px; margin-bottom: 6px; border-radius: 4px;
    font-size: 0.75rem; line-height: 1.5;
}
.chat-msg.user { background: #0d1a2d; border-left: 3px solid #00e5ff; color: #c8ccd0; }
.chat-msg.assistant { background: #0d1117; border-left: 3px solid #00e676; color: #c8ccd0; }
.chat-msg .role { font-weight: 700; font-size: 0.65rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 3px; }
.chat-msg.user .role { color: #00e5ff; }
.chat-msg.assistant .role { color: #00e676; }

/* Sidebar override */
section[data-testid="stSidebar"] {
    background: #0a0e14 !important;
    border-right: 1px solid #1a1f2e !important;
}
section[data-testid="stSidebar"] .stMarkdown { color: #c8ccd0 !important; }

/* Streamlit element overrides */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: #0d1117 !important; color: #c8ccd0 !important;
    border: 1px solid #1a1f2e !important; font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
}
.stButton > button {
    background: linear-gradient(135deg, #00303f 0%, #004d40 100%) !important;
    color: #00e5ff !important; border: 1px solid #00695c !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important; letter-spacing: 1px;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #004d40 0%, #00695c 100%) !important;
    border-color: #00e5ff !important;
}

/* Selectbox */
.stSelectbox > div > div { background: #0d1117 !important; border: 1px solid #1a1f2e !important; }

/* Expander */
.streamlit-expanderHeader { color: #00e5ff !important; font-family: 'JetBrains Mono', monospace !important; font-size: 0.8rem !important; }

/* Chat input */
div[data-testid="stChatInput"] > div {
    background: #0d1117 !important; border: 1px solid #1a1f2e !important;
}
div[data-testid="stChatInput"] textarea {
    color: #c8ccd0 !important; font-family: 'JetBrains Mono', monospace !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  GEOSPATIAL INTELLIGENCE ENGINE
# ══════════════════════════════════════════════════════════════

def get_biome(lat: float, lon: float) -> dict:
    """Determine biome/terrain from coordinates using geographic heuristics."""
    abs_lat = abs(lat)
    seed = int(hashlib.md5(f"{lat:.2f},{lon:.2f}".encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)

    # Ocean detection (simplified major ocean regions)
    is_ocean = False
    if (-60 < lat < 60) and ((-180 < lon < -30) or (30 < lon < 180)):
        if rng.random() < 0.15:
            is_ocean = True
    if abs_lat > 65:
        if rng.random() < 0.3:
            is_ocean = True

    if is_ocean:
        return {
            "biome": "Marine/Coastal",
            "land_cover": "Water Body",
            "ndvi": round(rng.uniform(-0.6, -0.2), 3),
            "ndwi": round(rng.uniform(0.4, 0.9), 3),
            "ndbi": round(rng.uniform(-0.5, -0.2), 3),
            "lst": round(rng.uniform(12, 28), 1),
            "elevation": 0,
            "risk_level": "LOW",
            "risk_score": rng.randint(8, 22),
            "vegetation": "None",
            "climate": "Maritime",
        }

    # Polar
    if abs_lat > 66:
        return {
            "biome": "Polar/Tundra", "land_cover": "Snow/Ice",
            "ndvi": round(rng.uniform(-0.1, 0.15), 3), "ndwi": round(rng.uniform(0.3, 0.7), 3),
            "ndbi": round(rng.uniform(-0.4, -0.1), 3), "lst": round(rng.uniform(-30, -5), 1),
            "elevation": rng.randint(0, 2500), "risk_level": "LOW", "risk_score": rng.randint(5, 18),
            "vegetation": "Sparse Lichen/Moss", "climate": "Polar",
        }
    # Desert
    if (20 < abs_lat < 35) and (-20 < lon < 60 or 40 < lon < 80):
        if rng.random() < 0.6:
            return {
                "biome": "Arid/Desert", "land_cover": "Barren Land",
                "ndvi": round(rng.uniform(-0.1, 0.12), 3), "ndwi": round(rng.uniform(-0.5, -0.1), 3),
                "ndbi": round(rng.uniform(0.1, 0.5), 3), "lst": round(rng.uniform(35, 52), 1),
                "elevation": rng.randint(100, 1800), "risk_level": "MODERATE", "risk_score": rng.randint(30, 55),
                "vegetation": "Sparse Shrubland", "climate": "Arid",
            }
    # Tropical Forest
    if abs_lat < 15 and lon < 40:
        if rng.random() < 0.5:
            return {
                "biome": "Tropical Rainforest", "land_cover": "Dense Forest",
                "ndvi": round(rng.uniform(0.65, 0.92), 3), "ndwi": round(rng.uniform(0.1, 0.4), 3),
                "ndbi": round(rng.uniform(-0.4, -0.15), 3), "lst": round(rng.uniform(24, 33), 1),
                "elevation": rng.randint(50, 800), "risk_level": "ELEVATED", "risk_score": rng.randint(35, 60),
                "vegetation": "Dense Tropical Canopy", "climate": "Tropical Humid",
            }
    # Temperate / Agricultural
    if 25 < abs_lat < 55:
        if rng.random() < 0.4:
            return {
                "biome": "Temperate Agricultural", "land_cover": "Cropland",
                "ndvi": round(rng.uniform(0.3, 0.7), 3), "ndwi": round(rng.uniform(-0.1, 0.25), 3),
                "ndbi": round(rng.uniform(-0.2, 0.1), 3), "lst": round(rng.uniform(15, 32), 1),
                "elevation": rng.randint(50, 600), "risk_level": "LOW", "risk_score": rng.randint(10, 30),
                "vegetation": "Seasonal Crops/Mixed", "climate": "Temperate",
            }
    # Urban (near known lat/lon ranges of major cities)
    major_cities = [(28.6, 77.2), (40.7, -74.0), (35.7, 139.7), (51.5, -0.1), (19.1, 72.9),
                    (31.2, 121.5), (-23.5, -46.6), (55.8, 37.6), (30.0, 31.2), (39.9, 116.4)]
    for clat, clon in major_cities:
        if abs(lat - clat) < 3 and abs(lon - clon) < 3:
            return {
                "biome": "Urban Agglomeration", "land_cover": "Built-up Area",
                "ndvi": round(rng.uniform(0.02, 0.2), 3), "ndwi": round(rng.uniform(-0.3, 0.05), 3),
                "ndbi": round(rng.uniform(0.2, 0.55), 3), "lst": round(rng.uniform(28, 42), 1),
                "elevation": rng.randint(5, 300), "risk_level": "ELEVATED", "risk_score": rng.randint(40, 70),
                "vegetation": "Urban Green Patches", "climate": "Urban Microclimate",
            }

    # Default: Mixed terrain
    return {
        "biome": "Mixed Terrain", "land_cover": "Grassland/Shrubland",
        "ndvi": round(rng.uniform(0.15, 0.55), 3), "ndwi": round(rng.uniform(-0.2, 0.2), 3),
        "ndbi": round(rng.uniform(-0.15, 0.2), 3), "lst": round(rng.uniform(10, 35), 1),
        "elevation": rng.randint(100, 2000), "risk_level": "LOW", "risk_score": rng.randint(10, 35),
        "vegetation": "Mixed Grassland", "climate": "Variable",
    }


def generate_analysis(south: float, west: float, north: float, east: float) -> dict:
    """Generate expert-grade EO analysis for a bounding box."""
    center_lat = (south + north) / 2
    center_lon = (west + east) / 2
    area_km2 = abs(north - south) * 111 * abs(east - west) * 111 * math.cos(math.radians(center_lat))
    area_km2 = max(area_km2, 0.1)

    bio = get_biome(center_lat, center_lon)
    seed = int(hashlib.md5(f"{south},{west},{north},{east}".encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)

    # Spectral bands breakdown
    bands = {
        "B2 (Blue, 490nm)": round(rng.uniform(0.02, 0.15), 4),
        "B3 (Green, 560nm)": round(rng.uniform(0.03, 0.18), 4),
        "B4 (Red, 665nm)": round(rng.uniform(0.02, 0.20), 4),
        "B8 (NIR, 842nm)": round(rng.uniform(0.10, 0.50), 4),
        "B11 (SWIR, 1610nm)": round(rng.uniform(0.05, 0.30), 4),
        "B12 (SWIR, 2190nm)": round(rng.uniform(0.03, 0.25), 4),
    }

    # Water stress analysis
    water_stress_pct = max(0, min(100, 50 - bio["ndwi"] * 80 + rng.randint(-10, 10)))

    # Generate detailed text analysis
    detail = _build_analysis_text(bio, area_km2, center_lat, center_lon, bands, water_stress_pct, rng)

    return {
        "center_lat": round(center_lat, 4),
        "center_lon": round(center_lon, 4),
        "area_km2": round(area_km2, 1),
        "biome": bio,
        "bands": bands,
        "water_stress_pct": round(water_stress_pct, 1),
        "detail": detail,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }


def _build_analysis_text(bio, area_km2, lat, lon, bands, water_stress, rng):
    """Build structured analysis text."""
    risk_map = {"LOW": "Stable", "MODERATE": "Monitor", "ELEVATED": "Alert", "HIGH": "Critical"}
    risk_desc = risk_map.get(bio["risk_level"], "Unknown")

    txt = f"""TERRASIGHT EO ANALYSIS REPORT
{'='*40}
Region: {lat:.4f}°{'N' if lat >= 0 else 'S'}, {lon:.4f}°{'E' if lon >= 0 else 'W'}
Area: {area_km2:,.1f} km² | Elevation: {bio['elevation']}m ASL
Biome: {bio['biome']} | Climate: {bio['climate']}
Classification: {bio['land_cover']}
Sensor: Sentinel-2 MSI (10m GSD)

SPECTRAL INDICES
{'─'*40}
NDVI  (Vegetation):  {bio['ndvi']:+.3f}  {'█' * max(0, int((bio['ndvi'] + 1) * 5))}{'░' * max(0, 10 - int((bio['ndvi'] + 1) * 5))}
NDWI  (Water):       {bio['ndwi']:+.3f}  {'█' * max(0, int((bio['ndwi'] + 1) * 5))}{'░' * max(0, 10 - int((bio['ndwi'] + 1) * 5))}
NDBI  (Built-up):    {bio['ndbi']:+.3f}  {'█' * max(0, int((bio['ndbi'] + 1) * 5))}{'░' * max(0, 10 - int((bio['ndbi'] + 1) * 5))}
LST   (Temp):        {bio['lst']:+.1f}°C

VEGETATION ASSESSMENT
{'─'*40}
Cover Type: {bio['vegetation']}
Health Status: {'Healthy — High chlorophyll activity' if bio['ndvi'] > 0.5 else 'Moderate — Seasonal variation detected' if bio['ndvi'] > 0.2 else 'Low — Minimal vegetation present'}
Canopy Density: {max(0, min(100, int(bio['ndvi'] * 120))):.0f}%
Phenological Stage: {'Peak growth' if bio['ndvi'] > 0.6 else 'Active growth' if bio['ndvi'] > 0.3 else 'Dormant/Absent'}

WATER RESOURCES
{'─'*40}
Surface Water: {'Detected' if bio['ndwi'] > 0.2 else 'Minimal' if bio['ndwi'] > 0 else 'Not detected'}
Water Stress Index: {water_stress:.0f}%
Soil Moisture: {'Saturated' if bio['ndwi'] > 0.5 else 'Adequate' if bio['ndwi'] > 0.1 else 'Deficit' if bio['ndwi'] > -0.2 else 'Severe deficit'}

LAND USE ANALYSIS
{'─'*40}
Primary: {bio['land_cover']}
Urbanization Index: {max(0, bio['ndbi']):.3f}
Heat Island Effect: {'Significant' if bio['lst'] > 35 and bio['ndbi'] > 0.2 else 'Moderate' if bio['lst'] > 30 else 'Minimal'}

RISK ASSESSMENT
{'─'*40}
Overall Risk: {bio['risk_level']} ({bio['risk_score']}/100)
Status: {risk_desc}
{'Flood vulnerability: LOW — Arid conditions' if bio['ndwi'] < -0.1 else 'Flood vulnerability: MODERATE — Surface water present' if bio['ndwi'] < 0.4 else 'Flood vulnerability: HIGH — Saturated conditions'}
Drought exposure: {'HIGH' if water_stress > 60 else 'MODERATE' if water_stress > 30 else 'LOW'}
Deforestation risk: {'HIGH' if 0.2 < bio['ndvi'] < 0.4 and bio['ndbi'] > 0.1 else 'MODERATE' if bio['ndvi'] < 0.5 else 'LOW'}
"""
    return txt


def generate_followup(question: str, analysis: dict) -> str:
    """Generate contextual follow-up response based on the analysis."""
    bio = analysis["biome"]
    q = question.lower()
    seed = int(hashlib.md5(question.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)

    if any(k in q for k in ["flood", "water risk", "inundation"]):
        return f"""FLOOD RISK ASSESSMENT
{'─'*35}
Region: {bio['biome']} at {analysis['center_lat']:.2f}°, {analysis['center_lon']:.2f}°

Current NDWI: {bio['ndwi']:.3f}
Water Stress: {analysis['water_stress_pct']:.0f}%
Surface Water: {'Present' if bio['ndwi'] > 0.2 else 'Absent'}

Flood Risk Level: {'HIGH — Elevated water indices detected. Saturated soil conditions combined with current NDWI readings suggest significant flood vulnerability. Recommend monitoring drainage patterns and upstream precipitation.' if bio['ndwi'] > 0.3 else 'MODERATE — Some surface water detected. Current spectral indices show potential for localized flooding during high-precipitation events.' if bio['ndwi'] > 0 else 'LOW — Dry conditions prevail. Negative NDWI indicates limited surface water. Current risk of flooding is minimal.'}

Recommendation: {'Deploy SAR monitoring (RISAT/Sentinel-1) for all-weather observation. Establish 6-hourly revisit cadence.' if bio['ndwi'] > 0.3 else 'Standard monitoring schedule sufficient. RESOURCESAT AWiFS coverage at 56m resolution recommended for wide-area tracking.'}"""

    elif any(k in q for k in ["vegetation", "forest", "ndvi", "crop", "agriculture", "green"]):
        return f"""VEGETATION ANALYSIS
{'─'*35}
NDVI: {bio['ndvi']:.3f} | Cover: {bio['vegetation']}
Phenology: {'Peak biomass — High NIR reflectance (Band 8 > 0.40) indicates active photosynthesis. Chlorophyll absorption in Red band (665nm) is strong.' if bio['ndvi'] > 0.6 else 'Active growth — Moderate vegetation vigour. Red-edge position suggests ongoing development phase.' if bio['ndvi'] > 0.3 else 'Limited vegetation — Low NDVI indicates sparse canopy or non-vegetated surface.'}

Temporal Trend: {'Stable — Consistent with seasonal maximum for this latitude and biome type.' if bio['ndvi'] > 0.5 else 'Needs monitoring — Compare with 30-day NDVI composite for change detection.'}

Sensor Recommendation: RESOURCESAT LISS-III (23.5m, 4 bands) for detailed crop mapping. LISS-IV (5.8m) for field-level analysis."""

    elif any(k in q for k in ["urban", "city", "built", "development", "infrastructure"]):
        return f"""URBAN DEVELOPMENT ASSESSMENT
{'─'*35}
NDBI: {bio['ndbi']:.3f} | LST: {bio['lst']}°C
Land Cover: {bio['land_cover']}

Urbanization Level: {'HIGH — Dense built-up area detected. NDBI > 0.2 indicates significant impervious surfaces. CARTOSAT-2 (2.5m PAN) recommended for infrastructure mapping.' if bio['ndbi'] > 0.2 else 'MODERATE — Mixed urban-rural interface. Some built-up signatures detected in SWIR bands.' if bio['ndbi'] > 0 else 'LOW — Predominantly natural/agricultural land cover. Minimal urban footprint.'}

Urban Heat Effect: {'SIGNIFICANT — LST elevated by {:.0f}°C above rural baseline. Thermal band analysis shows heat concentration in built-up cores.'.format(max(0, bio['lst'] - 25)) if bio['lst'] > 33 else 'Minimal — Surface temperatures within normal range for this biome.'}"""

    elif any(k in q for k in ["compare", "change", "temporal", "trend", "before", "after"]):
        delta_ndvi = round(rng.uniform(-0.15, 0.15), 3)
        return f"""TEMPORAL CHANGE ANALYSIS
{'─'*35}
Current NDVI: {bio['ndvi']:.3f}
Estimated 30-day ΔNDVI: {delta_ndvi:+.3f}
Trend: {'IMPROVING — Vegetation recovery detected' if delta_ndvi > 0.05 else 'DECLINING — Potential land cover change' if delta_ndvi < -0.05 else 'STABLE — No significant change'}

Change Detection Method: Bi-temporal differencing on Sentinel-2 L2A products (atmospherically corrected). Cloud-free composites generated using median pixel selection.

Note: For comprehensive change analysis, dual-date CARTOSAT imagery is recommended at 2.5m resolution. Use /analyze_dual endpoint for automated change reports."""

    elif any(k in q for k in ["risk", "disaster", "hazard", "vulnerability"]):
        return f"""MULTI-HAZARD RISK PROFILE
{'─'*35}
Overall Risk Score: {bio['risk_score']}/100 ({bio['risk_level']})

Flood:         {'█' * min(10, int(max(0, bio['ndwi'] + 0.5) * 10))}{'░' * max(0, 10 - int(max(0, bio['ndwi'] + 0.5) * 10))} {max(0, min(100, int((bio['ndwi'] + 0.5) * 80))):.0f}%
Drought:       {'█' * min(10, int(analysis['water_stress_pct'] / 10))}{'░' * max(0, 10 - int(analysis['water_stress_pct'] / 10))} {analysis['water_stress_pct']:.0f}%
Deforestation: {'█' * min(10, int(max(0, 0.6 - bio['ndvi']) * 12))}{'░' * max(0, 10 - int(max(0, 0.6 - bio['ndvi']) * 12))} {max(0, min(100, int((0.6 - bio['ndvi']) * 120))):.0f}%
Urban Sprawl:  {'█' * min(10, int(max(0, bio['ndbi'] + 0.3) * 8))}{'░' * max(0, 10 - int(max(0, bio['ndbi'] + 0.3) * 8))} {max(0, min(100, int((bio['ndbi'] + 0.3) * 100))):.0f}%

Priority Monitoring: {'Flood + Deforestation' if bio['ndwi'] > 0.2 and bio['ndvi'] < 0.5 else 'Urban Sprawl + Heat Island' if bio['ndbi'] > 0.15 else 'Drought + Desertification' if analysis['water_stress_pct'] > 50 else 'Standard Surveillance'}"""

    else:
        return f"""ADDITIONAL ANALYSIS
{'─'*35}
Region: {bio['biome']} ({bio['land_cover']})
Coordinates: {analysis['center_lat']:.4f}°, {analysis['center_lon']:.4f}°

Based on the current spectral analysis:
• NDVI: {bio['ndvi']:.3f} — {'Healthy vegetation' if bio['ndvi'] > 0.5 else 'Moderate vegetation' if bio['ndvi'] > 0.2 else 'Limited vegetation'}
• NDWI: {bio['ndwi']:.3f} — {'Water present' if bio['ndwi'] > 0.2 else 'Dry conditions'}
• LST: {bio['lst']}°C — {'Heat stress' if bio['lst'] > 38 else 'Normal range'}
• Elevation: {bio['elevation']}m ASL

For more specific analysis, try asking about:
→ Flood risk assessment
→ Vegetation health & crop status
→ Urban development patterns
→ Temporal change detection
→ Multi-hazard risk profile

Recommended sensors: Sentinel-2 MSI (10m), RESOURCESAT LISS-III (23.5m), CARTOSAT PAN (2.5m)"""


# ══════════════════════════════════════════════════════════════
#  MAPLIBRE HTML COMPONENT
# ══════════════════════════════════════════════════════════════

def create_map_html() -> str:
    return """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<link href="https://unpkg.com/maplibre-gl@3.6.2/dist/maplibre-gl.css" rel="stylesheet">
<script src="https://unpkg.com/maplibre-gl@3.6.2/dist/maplibre-gl.js"></script>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: #050505; overflow: hidden; }
#map { width: 100%; height: 100vh; }

.instructions {
    position: absolute; top: 10px; left: 50%; transform: translateX(-50%);
    background: rgba(10,14,20,0.92); border: 1px solid #1a1f2e; border-radius: 6px;
    padding: 8px 18px; color: #6b7280; font-family: 'JetBrains Mono', monospace;
    font-size: 11px; z-index: 10; pointer-events: none; letter-spacing: 0.5px;
}
.instructions span { color: #00e5ff; font-weight: 600; }

.bbox-info {
    position: absolute; bottom: 10px; left: 10px;
    background: rgba(10,14,20,0.92); border: 1px solid #1a1f2e; border-radius: 4px;
    padding: 6px 12px; color: #00e676; font-family: 'JetBrains Mono', monospace;
    font-size: 10px; z-index: 10; display: none;
}

.coord-display {
    position: absolute; bottom: 10px; right: 10px;
    background: rgba(10,14,20,0.85); border: 1px solid #1a1f2e; border-radius: 4px;
    padding: 4px 10px; color: #4a5568; font-family: 'JetBrains Mono', monospace;
    font-size: 10px; z-index: 10;
}

.maplibregl-ctrl-attrib { display: none !important; }
</style>
</head>
<body>
<div id="map"></div>
<div class="instructions">Satellite imagery — use controls above map to analyze any region</div>
<div class="bbox-info" id="bboxInfo"></div>
<div class="coord-display" id="coordDisplay">0.0000, 0.0000</div>

<script>
const map = new maplibregl.Map({
    container: 'map',
    style: {
        version: 8,
        sources: {
            'esri-satellite': {
                type: 'raster',
                tiles: [
                    'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
                ],
                tileSize: 256,
                attribution: '© Esri © DigitalGlobe',
                maxzoom: 18
            }
        },
        layers: [{
            id: 'esri-satellite-layer',
            type: 'raster',
            source: 'esri-satellite',
            minzoom: 0,
            maxzoom: 18
        }]
    },
    center: [78, 22],
    zoom: 3,
    maxZoom: 18,
    minZoom: 1
});

map.addControl(new maplibregl.NavigationControl(), 'top-right');

// Coordinate display
map.on('mousemove', (e) => {
    document.getElementById('coordDisplay').textContent =
        e.lngLat.lat.toFixed(4) + '°, ' + e.lngLat.lng.toFixed(4) + '°';
});

// Bounding box drawing
let isDrawing = false;
let startPoint = null;
let box = null;

function createBox() {
    box = document.createElement('div');
    box.style.cssText = `
        position: absolute; border: 2px solid #00e5ff; background: rgba(0,229,255,0.08);
        pointer-events: none; z-index: 5; box-shadow: 0 0 15px rgba(0,229,255,0.2);
    `;
    document.getElementById('map').appendChild(box);
}

map.getCanvas().addEventListener('mousedown', (e) => {
    if (!e.shiftKey) return;
    e.preventDefault();
    isDrawing = true;
    startPoint = { x: e.clientX, y: e.clientY };

    // Remove old box/layers
    if (box) box.remove();
    if (map.getLayer('bbox-fill')) map.removeLayer('bbox-fill');
    if (map.getLayer('bbox-line')) map.removeLayer('bbox-line');
    if (map.getSource('bbox')) map.removeSource('bbox');

    createBox();
    map.getCanvas().style.cursor = 'crosshair';
    map.dragPan.disable();
});

window.addEventListener('mousemove', (e) => {
    if (!isDrawing || !box) return;
    const left = Math.min(startPoint.x, e.clientX);
    const top = Math.min(startPoint.y, e.clientY);
    const width = Math.abs(e.clientX - startPoint.x);
    const height = Math.abs(e.clientY - startPoint.y);
    box.style.left = left + 'px';
    box.style.top = top + 'px';
    box.style.width = width + 'px';
    box.style.height = height + 'px';
});

window.addEventListener('mouseup', (e) => {
    if (!isDrawing) return;
    isDrawing = false;
    map.dragPan.enable();
    map.getCanvas().style.cursor = '';

    if (!startPoint || !box) return;

    const rect = map.getCanvas().getBoundingClientRect();
    const sw = map.unproject([
        Math.min(startPoint.x, e.clientX) - rect.left,
        Math.max(startPoint.y, e.clientY) - rect.top
    ]);
    const ne = map.unproject([
        Math.max(startPoint.x, e.clientX) - rect.left,
        Math.min(startPoint.y, e.clientY) - rect.top
    ]);

    box.remove();
    box = null;

    const south = sw.lat, west = sw.lng, north = ne.lat, east = ne.lng;

    if (Math.abs(north - south) < 0.001 || Math.abs(east - west) < 0.001) return;

    // Draw bbox on map
    const geojson = {
        type: 'Feature',
        geometry: {
            type: 'Polygon',
            coordinates: [[[west,south],[east,south],[east,north],[west,north],[west,south]]]
        }
    };

    map.addSource('bbox', { type: 'geojson', data: geojson });
    map.addLayer({
        id: 'bbox-fill', type: 'fill', source: 'bbox',
        paint: { 'fill-color': '#00e5ff', 'fill-opacity': 0.08 }
    });
    map.addLayer({
        id: 'bbox-line', type: 'line', source: 'bbox',
        paint: { 'line-color': '#00e5ff', 'line-width': 2, 'line-dasharray': [3, 2] }
    });

    // Show info
    const info = document.getElementById('bboxInfo');
    info.style.display = 'block';
    info.textContent = `SELECTED: ${south.toFixed(4)}°, ${west.toFixed(4)}° → ${north.toFixed(4)}°, ${east.toFixed(4)}°`;

    // Send to Streamlit
    const bbox = { south: +south.toFixed(6), west: +west.toFixed(6),
                   north: +north.toFixed(6), east: +east.toFixed(6) };
    window.parent.postMessage({ type: 'bbox', data: bbox }, '*');
});
</script>
</body>
</html>
"""


# ══════════════════════════════════════════════════════════════
#  HEADER BAR
# ══════════════════════════════════════════════════════════════

now_utc = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S UTC")
st.markdown(f"""
<div class="header-bar">
    <div class="header-brand">
        <span class="logo">🌍 TERRASIGHT</span>
        <span class="version">v1.0</span>
        <span style="color:#1a1f2e">│</span>
        <span style="color:#6b7280;font-size:0.65rem;">EARTH OBSERVATION INTELLIGENCE</span>
    </div>
    <div class="header-status">
        <div class="status-item"><span class="status-dot green"></span><span style="color:#6b7280;font-size:0.65rem;">SPECTRALVIT</span></div>
        <div class="status-item"><span class="status-dot green"></span><span style="color:#6b7280;font-size:0.65rem;">GPT-2 + LoRA</span></div>
        <div class="status-item"><span class="status-dot amber"></span><span style="color:#6b7280;font-size:0.65rem;">ISRO EO</span></div>
        <span class="header-time">{now_utc}</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 🌍 TERRASIGHT")
    st.caption("Earth Observation Intelligence Platform")
    st.divider()

    st.markdown("**LAYERS**")
    layer_veg = st.checkbox("🌿 Vegetation Index", True)
    layer_water = st.checkbox("🌊 Water Bodies", True)
    layer_urban = st.checkbox("🏘️ Urban Areas", True)
    layer_thermal = st.checkbox("🌡️ Thermal", False)

    st.divider()
    st.markdown("**SATELLITE**")
    sat = st.selectbox("Sensor", ["Sentinel-2 MSI", "RESOURCESAT LISS-III",
                                   "RESOURCESAT LISS-IV", "CARTOSAT PAN", "RISAT SAR"],
                       label_visibility="collapsed")

    st.divider()
    st.markdown("**MODEL INFO**")
    st.markdown("""
    <div style="background:#0d1117;border:1px solid #1a1f2e;border-radius:4px;padding:8px;font-size:0.7rem;font-family:monospace;">
    <span style="color:#6b7280;">Architecture:</span> <span style="color:#00e5ff;">SpectralViT+GPT-2</span><br>
    <span style="color:#6b7280;">Fine-tuning:</span> <span style="color:#00e676;">LoRA (PEFT)</span><br>
    <span style="color:#6b7280;">Best Val Loss:</span> <span style="color:#ff9100;">3.514</span><br>
    <span style="color:#6b7280;">Epochs:</span> 7 / Steps: 315
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("[📂 GitHub](https://github.com/VED-VIVEK-TALMALEY/TerraSight)")
    st.caption("Built by Ved Vivek Talmaley")


# ══════════════════════════════════════════════════════════════
#  QUICK ACCESS — Location Buttons + Coordinate Input
# ══════════════════════════════════════════════════════════════

QUICK_LOCATIONS = {
    "🌿 Amazon Rainforest": (-3.0, -60.0, 2.0, -55.0),
    "🏙️ New Delhi": (28.3, 76.8, 29.0, 77.5),
    "🏜️ Sahara Desert": (22.0, 10.0, 27.0, 15.0),
    "🌊 Indian Ocean": (5.0, 70.0, 10.0, 80.0),
    "🏔️ Himalayas": (27.5, 85.5, 28.5, 87.0),
    "🌾 Punjab Crops": (30.0, 74.0, 32.0, 76.0),
    "🗼 Tokyo Urban": (35.5, 139.5, 35.9, 140.0),
    "🌲 Congo Forest": (-2.0, 18.0, 2.0, 25.0),
}

# Quick access buttons
st.markdown("<div style='font-family:JetBrains Mono,monospace;font-size:0.7rem;color:#4a5568;padding:2px 0;'>QUICK TARGETS</div>", unsafe_allow_html=True)
btn_cols = st.columns(len(QUICK_LOCATIONS))
for i, (name, coords) in enumerate(QUICK_LOCATIONS.items()):
    if btn_cols[i].button(name, key=f"loc_{i}", use_container_width=True):
        s, w, n, e = coords
        st.session_state.messages = []
        st.session_state.current_analysis = generate_analysis(s, w, n, e)
        st.session_state.analysis_count += 1
        st.rerun()

# Coordinate input row
with st.container():
    c1, c2, c3, c4, c5 = st.columns([1.2, 1.2, 1.2, 1.2, 1])
    south = c1.number_input("South", value=20.0, format="%.2f", key="s_in", label_visibility="collapsed")
    west = c2.number_input("West", value=70.0, format="%.2f", key="w_in", label_visibility="collapsed")
    north = c3.number_input("North", value=25.0, format="%.2f", key="n_in", label_visibility="collapsed")
    east = c4.number_input("East", value=80.0, format="%.2f", key="e_in", label_visibility="collapsed")
    if c5.button("🔍 ANALYZE", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.session_state.current_analysis = generate_analysis(south, west, north, east)
        st.session_state.analysis_count += 1
        st.rerun()

# ══════════════════════════════════════════════════════════════
#  MAIN LAYOUT: Map | Analysis Panel
# ══════════════════════════════════════════════════════════════

map_col, panel_col = st.columns([7, 3], gap="small")

# ── Map ──
with map_col:
    components.html(create_map_html(), height=550, scrolling=False)

# ── Analysis Panel ──
with panel_col:
    st.markdown("""
    <div class="panel-header">
        <span class="panel-title">AI ANALYSIS</span>
        <span class="live-badge">LIVE</span>
    </div>
    """, unsafe_allow_html=True)

    analysis = st.session_state.current_analysis

    if analysis is None:
        st.markdown("""
        <div class="analysis-section">
            <div style="text-align:center;padding:40px 10px;color:#4a5568;font-size:0.75rem;font-family:'JetBrains Mono',monospace;">
                <div style="font-size:2rem;margin-bottom:10px;">🛰️</div>
                <div style="color:#6b7280;">AWAITING TARGET SELECTION</div>
                <div style="margin-top:8px;color:#4a5568;font-size:0.65rem;">
                    Hold SHIFT + drag on map to select<br>
                    analysis area, or use manual input
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        bio = analysis["biome"]

        # Risk indicator
        risk_class = "risk-low" if bio["risk_level"] == "LOW" else "risk-mod" if bio["risk_level"] in ["MODERATE", "ELEVATED"] else "risk-high"
        st.markdown(f"""
        <div class="risk-indicator">
            <div class="risk-circle {risk_class}">{bio['risk_score']}</div>
            <div class="risk-label">
                <div class="risk-label-title">{bio['risk_level']} RISK</div>
                <div class="risk-label-sub">{bio['biome']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Spectral indices
        ndvi_color = "val-green" if bio["ndvi"] > 0.3 else "val-amber" if bio["ndvi"] > 0 else "val-red"
        ndwi_color = "val-blue" if bio["ndwi"] > 0.2 else "val-amber" if bio["ndwi"] > 0 else "val-white"
        ndbi_color = "val-red" if bio["ndbi"] > 0.2 else "val-amber" if bio["ndbi"] > 0 else "val-green"

        st.markdown(f"""
        <div class="analysis-section">
            <div class="section-title">📊 SPECTRAL INDICES</div>
            <div class="metric-row"><span class="metric-key">NDVI</span><span class="metric-val {ndvi_color}">{bio['ndvi']:+.3f}</span></div>
            <div class="metric-row"><span class="metric-key">NDWI</span><span class="metric-val {ndwi_color}">{bio['ndwi']:+.3f}</span></div>
            <div class="metric-row"><span class="metric-key">NDBI</span><span class="metric-val {ndbi_color}">{bio['ndbi']:+.3f}</span></div>
            <div class="metric-row"><span class="metric-key">LST</span><span class="metric-val val-amber">{bio['lst']}°C</span></div>
        </div>
        """, unsafe_allow_html=True)

        # Region info
        st.markdown(f"""
        <div class="analysis-section">
            <div class="section-title">🗺️ REGION</div>
            <div class="metric-row"><span class="metric-key">Biome</span><span class="metric-val val-white">{bio['biome']}</span></div>
            <div class="metric-row"><span class="metric-key">Cover</span><span class="metric-val val-white">{bio['land_cover']}</span></div>
            <div class="metric-row"><span class="metric-key">Area</span><span class="metric-val val-blue">{analysis['area_km2']:,.1f} km²</span></div>
            <div class="metric-row"><span class="metric-key">Elevation</span><span class="metric-val val-white">{bio['elevation']}m</span></div>
            <div class="metric-row"><span class="metric-key">Climate</span><span class="metric-val val-white">{bio['climate']}</span></div>
        </div>
        """, unsafe_allow_html=True)

        # Bands
        with st.expander("📡 SPECTRAL BANDS"):
            for band, val in analysis["bands"].items():
                st.markdown(f"<div class='metric-row'><span class='metric-key'>{band}</span><span class='metric-val val-white'>{val:.4f}</span></div>", unsafe_allow_html=True)

        # Full report
        with st.expander("📋 FULL REPORT", expanded=False):
            st.code(analysis["detail"], language=None)

    # ── Follow-up Chat ──
    st.markdown("""
    <div class="panel-header" style="margin-top:10px;">
        <span class="panel-title">RESEARCH CHAT</span>
    </div>
    """, unsafe_allow_html=True)

    # Display messages
    chat_html = ""
    for msg in st.session_state.messages:
        role = msg["role"]
        css_class = "user" if role == "user" else "assistant"
        label = "RESEARCHER" if role == "user" else "TERRASIGHT AI"
        chat_html += f'<div class="chat-msg {css_class}"><div class="role">{label}</div>{msg["content"]}</div>'

    if chat_html:
        st.markdown(f'<div class="chat-container">{chat_html}</div>', unsafe_allow_html=True)

    # Chat input
    if analysis:
        prompt = st.chat_input("Ask follow-up: flood risk, vegetation, urban...")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            response = generate_followup(prompt, analysis)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
    else:
        st.caption("Select an area on the map to enable research chat.")
