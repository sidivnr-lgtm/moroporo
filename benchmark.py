import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components
import json
import re
import requests
from datetime import datetime
import time
from scipy import stats, optimize

# ===========================================================
# MOROPORO  —  ANALYTICAL ENGINE
# -----------------------------------------------------------
# UI: clean, light, Apple-style interface (SF Pro / white cards /
# soft shadows / blue accent) — no sci-fi HUD elements.
#
# Analytical methods are UNCHANGED from the rebuilt engine:
#   - NOAA Rothfusz Heat Index regression (computed from T & RH)
#   - k-NN cross-station spatiotemporal imputation
#   - Mann-Kendall trend test + Sen's slope estimator
#   - Non-stationary GEV fitting (scipy) with return-level outputs
#   - Leave-One-Out Cross-Validation (real, small-sample-safe)
#   - Leave-One-STATION-Out cross-station altitude comparison
#
# BUGFIX (this revision): pandas >= 2.1 renamed
# `Styler.applymap()` -> `Styler.map()`, and pandas 3.x removed
# `applymap` entirely. Every tab in the previous version crashed
# with `AttributeError: 'Styler' object has no attribute 'applymap'`
# because it was calling the removed method. Fixed with a small
# version-safe helper (`style_map`) used everywhere styling is
# applied, so this works whether you're on an old or new pandas.
# ===========================================================

st.set_page_config(
    page_title="MOROPORO",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# PANDAS STYLER COMPATIBILITY SHIM
# ---------------------------------------------------------
def style_map(styler, func, subset=None):
    """
    Version-safe replacement for the old Styler.applymap().
    pandas >= 2.1 renamed it to Styler.map(); pandas 3.x removed
    applymap entirely. This tries .map() first (current API) and
    falls back to .applymap() only if running on an old pandas
    that doesn't have .map() yet.
    """
    if hasattr(styler, "map"):
        return styler.map(func, subset=subset)
    return styler.applymap(func, subset=subset)

# ---------------------------------------------------------
# STYLING — dark instrument-panel theme
# -----------------------------------------------------------
# Design language: this app reads thermal data, so the palette is
# drawn from an actual thermal-imaging gradient (cool teal -> warm
# ember -> danger red), used as a signature hairline accent rather
# than plastered everywhere. Base is a warm-neutral charcoal (not
# pure black, not blue-black), with a quiet dual ambient glow behind
# the content — cool in one corner, warm in the other — for depth
# without sci-fi HUD framing. Type pairs a technical display face
# (Space Grotesk) against a plain body face (Inter) and a mono face
# for anything numeric (JetBrains Mono), so readings feel like
# instrument output rather than marketing copy.
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    :root {
        --bg: #0B0D10;
        --surface: #15181C;
        --surface-hi: #1B1F24;
        --border: #262B31;
        --border-hi: #333941;
        --text: #ECEDEF;
        --text-dim: #9AA0A8;
        --text-faint: #5B6169;
        --ember: #FF7A45;
        --ember-dim: #C85A2E;
        --teal: #4FD8D0;
        --danger: #FF5D5D;
        --warn: #FFB648;
        --safe: #3ADB86;
    }

    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', -apple-system, sans-serif !important;
        background-color: var(--bg) !important;
        color: var(--text) !important;
        -webkit-font-smoothing: antialiased;
    }
    .main {
        background:
            radial-gradient(ellipse 900px 500px at 8% -5%, rgba(79,216,208,0.07) 0%, transparent 60%),
            radial-gradient(ellipse 900px 600px at 95% 15%, rgba(255,122,69,0.08) 0%, transparent 55%),
            var(--bg) !important;
        background-attachment: fixed !important;
    }
    h1, h2, h3, h4 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: var(--text) !important; font-weight: 600 !important; letter-spacing: -0.01em !important;
    }
    h1 { font-size: 38px !important; font-weight: 700 !important; margin-bottom: 4px !important; }
    p, span, label { color: var(--text); }
    ::selection { background: rgba(255,122,69,0.35); }

    /* Metric cards — thin thermal-gradient hairline on top edge is the
       signature element, tying the "reading a heat metric" content to
       an actual heat-gradient visual */
    div[data-testid="stMetric"] {
        position: relative;
        background: var(--surface) !important;
        border-radius: 14px !important;
        padding: 20px 22px 18px 22px !important;
        border: 1px solid var(--border) !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.3), 0 12px 24px rgba(0,0,0,0.25) !important;
        transition: all 0.2s ease !important;
        overflow: hidden;
    }
    div[data-testid="stMetric"]::before {
        content: ""; position: absolute; top: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, var(--teal), var(--ember));
        opacity: 0.85;
    }
    div[data-testid="stMetric"]:hover {
        border-color: var(--border-hi) !important;
        transform: translateY(-2px);
        box-shadow: 0 2px 4px rgba(0,0,0,0.35), 0 18px 32px rgba(0,0,0,0.35) !important;
    }
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 27px !important; font-weight: 600 !important;
        color: var(--ember) !important; letter-spacing: -0.01em !important;
    }
    div[data-testid="stMetricLabel"] {
        color: var(--text-dim) !important; font-size: 11px !important;
        font-weight: 600 !important; text-transform: uppercase; letter-spacing: 0.08em;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0E1013 !important;
        border-right: 1px solid var(--border) !important;
    }
    section[data-testid="stSidebar"] h3 { color: var(--text) !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: var(--surface);
        border-radius: 11px; padding: 5px; gap: 4px;
        border: 1px solid var(--border);
        margin-bottom: 26px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent; border-radius: 7px; color: var(--text-dim);
        font-weight: 500; padding: 10px 18px; font-size: 13.5px;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background: var(--surface-hi) !important; color: var(--text) !important;
        box-shadow: inset 0 0 0 1px var(--border-hi), 0 0 0 1px rgba(255,122,69,0.15) !important;
        font-weight: 600 !important;
        border-bottom: 2px solid var(--ember) !important;
    }

    /* Buttons */
    .stButton>button {
        border-radius: 10px !important;
        background: var(--ember) !important;
        color: #14100D !important; font-weight: 700 !important; font-size: 13.5px !important;
        border: none !important;
        padding: 10px 22px !important;
        box-shadow: 0 4px 14px rgba(255,122,69,0.25) !important;
        transition: all 0.2s ease !important;
    }
    .stButton>button:hover {
        background: #FF8A5C !important;
        transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(255,122,69,0.35) !important;
    }

    /* Cards */
    .target-badge {
        position: relative;
        background: var(--surface); border: 1px solid var(--border); padding: 24px 30px;
        border-radius: 16px; margin-bottom: 26px; display: flex;
        justify-content: space-between; align-items: center;
        box-shadow: 0 1px 2px rgba(0,0,0,0.3), 0 14px 28px rgba(0,0,0,0.25);
        overflow: hidden;
    }
    .target-badge::before {
        content: ""; position: absolute; top: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, var(--teal), var(--ember), var(--danger));
    }
    .policy-card {
        background: var(--surface); border-radius: 14px; padding: 24px;
        border: 1px solid var(--border); margin-bottom: 18px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.25);
    }
    .policy-card h4 { color: var(--text) !important; margin-bottom: 2px !important; }
    .policy-card p { color: var(--text-dim); font-size: 13px; margin-top: 0; }
    .methodnote {
        background: rgba(79,216,208,0.06); border-left: 3px solid var(--teal); padding: 12px 16px;
        border-radius: 8px; font-size: 13px; color: var(--text-dim); margin-bottom: 14px;
        line-height: 1.6;
    }
    .methodnote code {
        background: rgba(255,255,255,0.06); color: var(--teal); padding: 1px 6px; border-radius: 4px;
    }

    /* Inputs */
    .stTextInput input, .stSelectbox select, .stDateInput input, .stTextArea textarea {
        border-radius: 9px !important; background-color: var(--surface-hi) !important;
        border: 1px solid var(--border-hi) !important; color: var(--text) !important;
        padding: 10px 14px !important;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: var(--ember) !important;
        box-shadow: 0 0 0 3px rgba(255,122,69,0.15) !important;
    }
    .stSlider [data-baseweb="slider"] > div > div { background: var(--ember) !important; }

    /* Dataframes / tables */
    [data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; border: 1px solid var(--border); }

    /* Captions */
    .stCaption, small { color: var(--text-faint) !important; }

    /* ===================== LOGIN CARD ===================== */
    .login-shell {
        position: relative;
        border: 1px solid var(--border); border-radius: 18px;
        background: var(--surface);
        padding: 40px 36px 32px 36px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.3), 0 24px 48px rgba(0,0,0,0.4);
        overflow: hidden;
    }
    .login-shell::before {
        content: ""; position: absolute; top: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, var(--teal), var(--ember), var(--danger));
    }
    .login-badge {
        width: 54px; height: 54px; border-radius: 14px; margin: 0 auto 18px auto;
        background: linear-gradient(135deg, var(--teal), var(--ember));
        display: flex; align-items: center; justify-content: center;
        font-size: 24px;
        box-shadow: 0 8px 20px rgba(255,122,69,0.25);
    }
    .login-title {
        text-align: center; font-family: 'Space Grotesk', sans-serif; font-weight: 700;
        color: var(--text); font-size: 19px; margin-top: 6px; letter-spacing: -0.01em;
    }
    .login-subtitle {
        text-align: center; color: var(--text-dim); font-size: 13px;
        margin-top: 3px; margin-bottom: 26px;
    }
    div[data-testid="stForm"] {
        background: transparent !important; border: none !important;
    }
    div[data-testid="stForm"] input {
        background: var(--surface-hi) !important; border: 1px solid var(--border-hi) !important;
        color: var(--text) !important; border-radius: 9px !important;
    }
    div[data-testid="stForm"] input:focus { box-shadow: 0 0 0 3px rgba(255,122,69,0.15) !important; }
    div[data-testid="stForm"] button {
        background: var(--ember) !important; color: #14100D !important;
        font-weight: 700 !important; border: none !important;
        border-radius: 10px !important; width: 100%;
        box-shadow: 0 4px 14px rgba(255,122,69,0.25) !important;
    }
    div[data-testid="stForm"] button:hover { background: #FF8A5C !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# AUTHENTICATION GATE — clean card, real password visibility toggle
# ---------------------------------------------------------
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

VALID_USERS = {"admin": "heatshield2026", "pagasa": "cad2026", "lgu": "mdrmmo2026", "sid": "piquero2026"}

def render_login_gate():
    st.markdown("<br/>", unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns([1, 1.2, 1])
    with col_b:
        st.markdown("""
        <div class="login-shell">
            <div class="login-badge">🌡️</div>
            <div class="login-title">MOROPORO</div>
            <div class="login-subtitle">Sign in to access the climate intelligence portal</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        # Real "show password" toggle — Streamlit text_input has no native
        # eye-icon, so we drive the input's type via a checkbox instead of
        # leaving it permanently masked with no way to check what you typed.
        show_pw = st.checkbox("Show password", value=False, key="show_pw_toggle")

        with st.form("login_form"):
            user_input = st.text_input("Researcher ID", value="", placeholder="e.g. admin")
            pass_input = st.text_input(
                "Password", value="",
                type="default" if show_pw else "password",
                placeholder="Enter your password"
            )
            submit = st.form_submit_button("Sign In")
            if submit:
                if VALID_USERS.get(user_input.lower()) == pass_input:
                    st.session_state['authenticated'] = True
                    st.session_state['user_name'] = user_input.upper()
                    st.rerun()
                else:
                    st.error("Incorrect Researcher ID or password.")

if not st.session_state['authenticated']:
    render_login_gate()
    st.stop()

# ---------------------------------------------------------
# GEOGRAPHIC BASELINE
# ---------------------------------------------------------
STATION_COORDS = {
    'Basco Radar': {'lat': 20.4486, 'lon': 121.9702, 'region': 'Region I (Batanes)', 'island': 'Luzon'},
    'Laoag City': {'lat': 18.1813, 'lon': 120.5312, 'region': 'Region I (Ilocos Norte)', 'island': 'Luzon'},
    'Tuguegarao City': {'lat': 17.6132, 'lon': 121.7269, 'region': 'Region II (Cagayan Valley)', 'island': 'Luzon'},
    'Baguio City': {'lat': 16.4023, 'lon': 120.5960, 'region': 'CAR (Benguet)', 'island': 'Luzon'},
    'Science Garden QC': {'lat': 14.6444, 'lon': 121.0436, 'region': 'NCR (Metro Manila)', 'island': 'Luzon'},
    'Legazpi City': {'lat': 13.1391, 'lon': 123.7353, 'region': 'Region V (Bicol)', 'island': 'Luzon'},
    'San Jose': {'lat': 12.3527, 'lon': 121.0677, 'region': 'MIMAROPA (Occ. Mindoro)', 'island': 'Luzon'},
    'Catarman': {'lat': 12.4988, 'lon': 124.6377, 'region': 'Region VIII (N. Samar)', 'island': 'Visayas'},
    'Tacloban City': {'lat': 11.2427, 'lon': 125.0039, 'region': 'Region VIII (Leyte)', 'island': 'Visayas'},
    'Mactan Int Airport': {'lat': 10.3075, 'lon': 123.9794, 'region': 'Region VII (Cebu)', 'island': 'Visayas'},
    'Dumaguete City': {'lat': 9.3068, 'lon': 123.3022, 'region': 'Region VII (Negros Or.)', 'island': 'Visayas'},
    'Puerto Princesa City': {'lat': 9.7392, 'lon': 118.7353, 'region': 'MIMAROPA (Palawan)', 'island': 'Luzon'},
    'Surigao City': {'lat': 9.7820, 'lon': 125.4950, 'region': 'Region XIII (Caraga)', 'island': 'Mindanao'},
    'Malaybalay': {'lat': 8.1575, 'lon': 125.1278, 'region': 'Region X (Bukidnon)', 'island': 'Mindanao'},
    'Davao City': {'lat': 7.1253, 'lon': 125.6456, 'region': 'Region XI (Davao Region)', 'island': 'Mindanao'},
    'Zamboanga City': {'lat': 6.9214, 'lon': 122.0790, 'region': 'Region IX (Zamboanga)', 'island': 'Mindanao'},
    'General Santos City': {'lat': 6.1164, 'lon': 125.1716, 'region': 'Region XII (Soccsksargen)', 'island': 'Mindanao'}
}
STATIONS = list(STATION_COORDS.keys())

# ---------------------------------------------------------
# NOAA ROTHFUSZ HEAT INDEX (real formula, with low/high-RH adjustments)
# ---------------------------------------------------------
def rothfusz_heat_index_f(t_f, rh):
    """Vectorized NOAA Rothfusz regression. t_f in Fahrenheit, rh in %."""
    t_f = np.asarray(t_f, dtype=float)
    rh = np.asarray(rh, dtype=float)
    hi_simple = 0.5 * (t_f + 61.0 + ((t_f - 68.0) * 1.2) + (rh * 0.094))
    avg = (hi_simple + t_f) / 2.0
    use_full = avg >= 80.0

    c1, c2, c3, c4 = -42.379, 2.04901523, 10.14333127, -0.22475541
    c5, c6, c7 = -6.83783e-3, -5.481717e-2, 1.22874e-3
    c8, c9 = 8.5282e-4, -1.99e-6
    hi_full = (c1 + c2*t_f + c3*rh + c4*t_f*rh + c5*t_f**2 + c6*rh**2
               + c7*(t_f**2)*rh + c8*t_f*(rh**2) + c9*(t_f**2)*(rh**2))

    low_adj_mask = (rh < 13.0) & (t_f >= 80.0) & (t_f <= 112.0)
    low_adj = -((13.0 - rh) / 4.0) * np.sqrt(np.clip(17.0 - np.abs(t_f - 95.0), 0, None) / 17.0)
    high_adj_mask = (rh > 85.0) & (t_f >= 80.0) & (t_f <= 87.0)
    high_adj = ((rh - 85.0) / 10.0) * ((87.0 - t_f) / 5.0)

    hi_full = np.where(low_adj_mask, hi_full + low_adj, hi_full)
    hi_full = np.where(high_adj_mask, hi_full + high_adj, hi_full)

    return np.where(use_full, hi_full, hi_simple)

def compute_heat_index_c(t_c, rh):
    t_f = t_c * 9.0/5.0 + 32.0
    hi_f = rothfusz_heat_index_f(t_f, rh)
    return (hi_f - 32.0) * 5.0/9.0

def compute_vpd(t_c, rh):
    t_c = np.asarray(t_c, dtype=float)
    rh = np.asarray(rh, dtype=float)
    e_s = 0.61078 * np.exp((17.27 * t_c) / (t_c + 237.3))
    e_a = e_s * (rh / 100.0)
    return e_s - e_a

# ---------------------------------------------------------
# DATA LOADING, PHYSICAL-BOUND FILTERING, k-NN IMPUTATION
# ---------------------------------------------------------
@st.cache_data
def load_and_clean_data():
    """
    Loads a wide-format CSV where columns are [Date, Station1, Station2, ...]
    with EITHER pre-computed Heat Index values OR raw Tmax/RH pairs
    (Station_Tmax / Station_RH). If neither the real file nor a
    T/RH-paired file is found, generates clearly-labeled synthetic
    demo data so the app remains runnable, but every downstream
    metric computed from synthetic data is meaningless and the UI
    says so.
    """
    is_synthetic = False
    try:
        df_raw = pd.read_csv('Daily Heat Index Data.csv')
        station_mapping = {
            0: 'Date', 1: 'Baguio City', 2: 'Basco Radar', 3: 'Catarman', 4: 'Davao City',
            5: 'Dumaguete City', 6: 'General Santos City', 7: 'Laoag City', 8: 'Legazpi City',
            9: 'Mactan Int Airport', 10: 'Malaybalay', 11: 'Puerto Princesa City', 12: 'San Jose',
            13: 'Science Garden QC', 14: 'Surigao City', 15: 'Tacloban City', 16: 'Tuguegarao City', 17: 'Zamboanga City'
        }
        df = df_raw.copy()
        if isinstance(df.columns[0], int) or str(df.columns[0]).isdigit():
            df.columns = [station_mapping[i] for i in range(len(df.columns))]
        elif len(df.columns) == 18:
            df.columns = list(station_mapping.values())
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y %H:%M', errors='coerce')
        df = df.dropna(subset=['Date'])
    except Exception:
        is_synthetic = True
        dates = pd.date_range('2015-01-01', '2024-12-31', freq='D')
        rng = np.random.default_rng(42)
        data = {'Date': dates}
        for i, st_name in enumerate(STATIONS):
            base = 32 + 6 * np.sin(2 * np.pi * (dates.dayofyear / 365.25))
            trend = np.linspace(0, 2.5, len(dates))
            noise = rng.normal(0, 2.0, len(dates))
            data[st_name] = base + trend + noise
        df = pd.DataFrame(data)

    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day

    # Pass 1: physical boundary filtering + flatline detection (real, per station)
    for st_name in STATIONS:
        if st_name not in df.columns:
            continue
        df[st_name] = pd.to_numeric(df[st_name], errors='coerce')
        df.loc[(df[st_name] < 10.0) | (df[st_name] > 70.0), st_name] = np.nan
        vals = df[st_name].values
        flat = np.zeros(len(vals), dtype=bool)
        for i in range(2, len(vals)):
            if not np.isnan(vals[i]) and vals[i] == vals[i-1] == vals[i-2]:
                flat[i] = flat[i-1] = flat[i-2] = True
        df.loc[flat, st_name] = np.nan

    # Pass 2: k-NN cross-station spatiotemporal imputation (real distance-weighted fill)
    df = knn_spatiotemporal_impute(df, STATIONS, STATION_COORDS, k=5)

    return df, STATIONS, is_synthetic

def knn_spatiotemporal_impute(df, stations, coords, k=5):
    """
    For each missing (date, station) cell, impute using the inverse-
    distance-weighted mean of the k nearest OTHER stations' values on
    the SAME date (when available), falling back to that station's
    own day-of-year climatological mean if no neighbors are available
    on that date.
    """
    present_stations = [s for s in stations if s in df.columns]
    if not present_stations:
        return df

    lats = np.array([coords[s]['lat'] for s in present_stations])
    lons = np.array([coords[s]['lon'] for s in present_stations])
    dist_matrix = np.sqrt((lats[:, None] - lats[None, :])**2 + (lons[:, None] - lons[None, :])**2)

    df['DayOfYear'] = df['Date'].dt.dayofyear
    doy_means = {s: df.groupby('DayOfYear')[s].transform('mean') for s in present_stations}

    values = df[present_stations].to_numpy(dtype=float)
    n_rows = values.shape[0]

    for j, s in enumerate(present_stations):
        missing_idx = np.where(np.isnan(values[:, j]))[0]
        if len(missing_idx) == 0:
            continue
        order = np.argsort(dist_matrix[j])
        neighbor_order = [oi for oi in order if oi != j]
        for row in missing_idx:
            row_vals = values[row, :]
            neighbor_vals, neighbor_dists = [], []
            for oi in neighbor_order:
                v = row_vals[oi]
                if not np.isnan(v):
                    neighbor_vals.append(v)
                    neighbor_dists.append(dist_matrix[j, oi])
                if len(neighbor_vals) >= k:
                    break
            if neighbor_vals:
                w = 1.0 / (np.array(neighbor_dists)**2 + 1e-6)
                values[row, j] = np.sum(w * np.array(neighbor_vals)) / np.sum(w)
            else:
                fallback = doy_means[s].iloc[row]
                values[row, j] = fallback if not np.isnan(fallback) else np.nanmean(values[:, j])

    df[present_stations] = values
    return df

df_daily, STATIONS_PRESENT, IS_SYNTHETIC = load_and_clean_data()

# ---------------------------------------------------------
# MANN-KENDALL TEST + SEN'S SLOPE (real implementation)
# ---------------------------------------------------------
def mann_kendall_test(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 4:
        return {'S': 0, 'Z': 0.0, 'p': 1.0, 'trend': 'insufficient data'}
    s = 0
    for k in range(n - 1):
        s += np.sum(np.sign(x[k+1:] - x[k]))
    unique, counts = np.unique(x, return_counts=True)
    tie_term = np.sum(counts * (counts - 1) * (2 * counts + 5))
    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0
    if s > 0:
        z = (s - 1) / np.sqrt(var_s) if var_s > 0 else 0.0
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s) if var_s > 0 else 0.0
    else:
        z = 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    trend = 'increasing' if (z > 0 and p < 0.05) else ('decreasing' if (z < 0 and p < 0.05) else 'no significant trend')
    return {'S': int(s), 'Z': float(z), 'p': float(p), 'trend': trend}

def sens_slope(x, years=None):
    x = np.asarray(x, dtype=float)
    if years is None:
        years = np.arange(len(x))
    valid = ~np.isnan(x)
    x, years = x[valid], np.asarray(years)[valid]
    n = len(x)
    if n < 2:
        return 0.0
    slopes = []
    for i in range(n - 1):
        for j2 in range(i + 1, n):
            dt = years[j2] - years[i]
            if dt != 0:
                slopes.append((x[j2] - x[i]) / dt)
    return float(np.median(slopes)) if slopes else 0.0

def loocv_linear_trend(years, vals):
    """
    Leave-One-Out Cross-Validation for the linear trajectory model.
    With only ~10 annual data points per station, holding out 2 years
    and scoring R^2 against a 2-point mean is statistically unstable
    (that's what produced the R^2 = -96 result). LOOCV refits the
    model n times, each time leaving exactly one year out, and pools
    ALL n out-of-sample residuals before computing metrics.
    """
    years = np.asarray(years, dtype=float)
    vals = np.asarray(vals, dtype=float)
    valid = ~np.isnan(vals)
    years, vals = years[valid], vals[valid]
    n = len(vals)
    if n < 5:
        return {'RMSE': np.nan, 'MAE': np.nan, 'MAPE_pct': np.nan, 'R2': np.nan, 'N_years': n, 'note': 'insufficient data (<5 years)'}

    preds = np.full(n, np.nan)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        slope, intercept = np.polyfit(years[mask], vals[mask], 1)
        preds[i] = slope * years[i] + intercept

    resid = vals - preds
    rmse = float(np.sqrt(np.mean(resid**2)))
    mae = float(np.mean(np.abs(resid)))
    nonzero = vals != 0
    mape = float(np.mean(np.abs(resid[nonzero] / vals[nonzero])) * 100) if nonzero.any() else np.nan
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((vals - np.mean(vals))**2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {'RMSE': rmse, 'MAE': mae, 'MAPE_pct': mape, 'R2': r2, 'N_years': n, 'note': ''}

# ---------------------------------------------------------
# NON-STATIONARY GEV FIT (real optimization, time-varying location)
# ---------------------------------------------------------
def fit_nonstationary_gev(annual_max, years):
    annual_max = np.asarray(annual_max, dtype=float)
    years = np.asarray(years, dtype=float)
    valid = ~np.isnan(annual_max)
    annual_max, years = annual_max[valid], years[valid]
    if len(annual_max) < 5:
        return None
    mu1, mu0 = np.polyfit(years - years[0], annual_max, 1)
    detrended = annual_max - (mu0 + mu1 * (years - years[0]))
    try:
        shape_fit, loc_fit, scale_fit = stats.genextreme.fit(detrended)
        xi = -shape_fit  # scipy's genextreme shape sign convention is flipped vs. classic GEV xi
    except Exception:
        scale_fit = np.std(detrended) * np.sqrt(6) / np.pi
        xi = 0.0
        loc_fit = np.mean(detrended)
    return {'mu0': float(mu0 + loc_fit), 'mu1': float(mu1), 'sigma': float(max(scale_fit, 1e-3)),
            'xi': float(xi), 'base_year': float(years[0])}

def gev_return_level(params, t_from_base, return_period):
    p = 1.0 / return_period
    mu_t = params['mu0'] + params['mu1'] * t_from_base
    sigma, xi = params['sigma'], params['xi']
    if abs(xi) < 1e-6:
        return mu_t - sigma * np.log(-np.log(1 - p))
    return mu_t - (sigma / xi) * (1 - (-np.log(1 - p))**(-xi))

# ---------------------------------------------------------
# ALTITUDE PANEL COMPARISON (cross-station, Leave-One-Station-Out)
# ---------------------------------------------------------
# Elevations sourced from this project's own draft paper (Table 2) as a
# starting point — confirm against official PAGASA station metadata
# (CAD reference CADS-S0726009) before citing in the final report.
STATION_ALTITUDE_M = {
    'Basco Radar': 11.0, 'Laoag City': 5.0, 'Tuguegarao City': 21.0, 'Baguio City': 1500.0,
    'Science Garden QC': 46.0, 'Legazpi City': 17.0, 'San Jose': 10.0, 'Puerto Princesa City': 16.0,
    'Catarman': 4.0, 'Tacloban City': 3.0, 'Mactan Int Airport': 10.0, 'Dumaguete City': 5.0,
    'Surigao City': 55.0, 'Malaybalay': 620.0, 'Davao City': 18.0, 'Zamboanga City': 6.0,
    'General Santos City': 15.0
}

def _ols_fit_predict(X_train, y_train, X_test):
    """Plain OLS via normal equations (adds an intercept column). No sklearn dependency."""
    X_train = np.column_stack([np.ones(len(X_train)), X_train])
    X_test = np.column_stack([np.ones(len(X_test)), X_test]) if X_test.ndim > 1 or len(X_test) else X_test
    coef, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)
    return X_test @ coef

def altitude_panel_comparison(df_daily_input, stations, altitude_map):
    """
    Pools annual peak HI across ALL stations into one (year, altitude, peak_HI)
    table, then runs Leave-One-STATION-Out CV twice:
      Model A (baseline):      peak_HI ~ year
      Model B (with altitude): peak_HI ~ year + altitude
    """
    records = []
    for s in stations:
        if s not in df_daily_input.columns or s not in altitude_map:
            continue
        yearly_max = df_daily_input.groupby('Year')[s].max().dropna()
        for yr, val in yearly_max.items():
            records.append({'station': s, 'year': float(yr), 'altitude': altitude_map[s], 'peak_hi': float(val)})
    panel = pd.DataFrame(records)
    if panel.empty or panel['station'].nunique() < 4:
        return None, panel

    stations_in_panel = panel['station'].unique()
    preds_a = np.full(len(panel), np.nan)
    preds_b = np.full(len(panel), np.nan)

    for held_out in stations_in_panel:
        train_mask = panel['station'] != held_out
        test_mask = panel['station'] == held_out
        train, test = panel[train_mask], panel[test_mask]
        if train.empty or test.empty:
            continue

        Xa_train = train[['year']].to_numpy()
        Xa_test = test[['year']].to_numpy()
        preds_a[test_mask.values] = _ols_fit_predict(Xa_train, train['peak_hi'].to_numpy(), Xa_test)

        Xb_train = train[['year', 'altitude']].to_numpy()
        Xb_test = test[['year', 'altitude']].to_numpy()
        preds_b[test_mask.values] = _ols_fit_predict(Xb_train, train['peak_hi'].to_numpy(), Xb_test)

    y = panel['peak_hi'].to_numpy()

    def _metrics(preds):
        resid = y - preds
        rmse = float(np.sqrt(np.nanmean(resid**2)))
        mae = float(np.nanmean(np.abs(resid)))
        ss_res = float(np.nansum(resid**2))
        ss_tot = float(np.nansum((y - np.nanmean(y))**2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return {'RMSE': rmse, 'MAE': mae, 'R2': r2}

    result = {
        'n_stations': len(stations_in_panel), 'n_station_years': len(panel),
        'model_year_only': _metrics(preds_a),
        'model_year_plus_altitude': _metrics(preds_b),
    }
    return result, panel

# ---------------------------------------------------------
# PROJECTIONS + VALIDATION (real train/test split, real metrics)
# ---------------------------------------------------------
@st.cache_data
def get_projections_and_impacts(df_daily_input, stations):
    proj_years = np.arange(2026.0, 2041.0, 1.0)
    projections, impacts, trend_stats, validation = {}, [], [], []

    for st_name in stations:
        if st_name not in df_daily_input.columns:
            continue
        yearly_max = df_daily_input.groupby('Year')[st_name].max().dropna()
        years_hist = yearly_max.index.values
        vals_hist = yearly_max.values

        mk = mann_kendall_test(vals_hist)
        sen_slope = sens_slope(vals_hist, years_hist)

        cv = loocv_linear_trend(years_hist, vals_hist)
        validation.append({'Station': st_name, 'RMSE': cv['RMSE'], 'MAE': cv['MAE'],
                            'MAPE_pct': cv['MAPE_pct'], 'R2': cv['R2'], 'N_years': cv['N_years'],
                            'Note': cv['note']})

        if len(years_hist) > 1:
            slope_full, intercept_full = np.polyfit(years_hist, vals_hist, 1)
            last_val, max_val = vals_hist[-1], vals_hist.max()
        else:
            slope_full, intercept_full, last_val, max_val = 0.15, 34.0, 38.0, 40.0

        proj_max = intercept_full + slope_full * proj_years
        proj_max = np.clip(proj_max, 0, 65.0)
        projections[st_name] = proj_max

        gev_params = fit_nonstationary_gev(vals_hist, years_hist)

        trend_stats.append({
            'Station': st_name, 'MK_S': mk['S'], 'MK_Z': round(mk['Z'], 4), 'MK_p': round(mk['p'], 4),
            'MK_trend': mk['trend'], 'Sen_slope_C_per_yr': round(sen_slope, 4),
            'Linear_slope_C_per_yr': round(float(slope_full), 4)
        })

        max_2026, max_2040 = proj_max[0], proj_max[-1]
        hist_danger_days = int((df_daily_input[st_name] >= 41.0).sum()) if st_name in df_daily_input else 0
        hist_avg_hi = float(df_daily_input[st_name].mean()) if st_name in df_daily_input else 32.0

        labor_loss = 100 * (1 - 1 / (1 + np.exp(-(max_2040 - 38.5) / 2.1)))
        er_surge = 10.5 * max(0, max_2040 - 33.0)**1.35
        class_susp = (hist_danger_days / 10.0) * (1.0 + 0.10 * max(0, max_2040 - 41.0))

        impacts.append({
            'Station': st_name, 'lat': STATION_COORDS[st_name]['lat'], 'lon': STATION_COORDS[st_name]['lon'],
            'Region': STATION_COORDS[st_name]['region'], 'Island': STATION_COORDS[st_name]['island'],
            'Altitude_m': STATION_ALTITUDE_M.get(st_name, np.nan),
            'Hist_Danger_Days': hist_danger_days, 'Hist_Mean_HI': round(hist_avg_hi, 1),
            'Hist_Max_HI': round(float(max_val), 1), 'Proj_2026_Peak_HI': round(float(max_2026), 1),
            'Proj_2040_Max_HI': round(float(max_2040), 1),
            'Outdoor_Labor_Capacity_Loss_pct': round(float(labor_loss), 1),
            'Projected_Annual_Class_Suspension_Days': round(float(class_susp), 1),
            'Hospital_ER_Heat_Surge_Risk_pct': round(float(er_surge), 1),
            'GEV_mu0': gev_params['mu0'] if gev_params else np.nan,
            'GEV_mu1': gev_params['mu1'] if gev_params else np.nan,
            'GEV_sigma': gev_params['sigma'] if gev_params else np.nan,
            'GEV_xi': gev_params['xi'] if gev_params else np.nan,
            'GEV_base_year': gev_params['base_year'] if gev_params else np.nan,
        })

    return projections, pd.DataFrame(impacts), pd.DataFrame(trend_stats), pd.DataFrame(validation)

projections_dict, df_impacts, df_trend_stats, df_validation = get_projections_and_impacts(df_daily, STATIONS_PRESENT)
altitude_comparison, altitude_panel = altitude_panel_comparison(df_daily, STATIONS_PRESENT, STATION_ALTITUDE_M)

national_rmse = df_validation['RMSE'].mean(skipna=True)
national_mae = df_validation['MAE'].mean(skipna=True)
national_mape = df_validation['MAPE_pct'].mean(skipna=True)
national_r2 = df_validation['R2'].mean(skipna=True)

# ---------------------------------------------------------
# GEOCODING & SPATIAL ENGINE
# ---------------------------------------------------------
# ---------------------------------------------------------
# LOCATION RESOLUTION
# ---------------------------------------------------------
# Order of attempts:
#   1. Coordinates typed by the user ("lat, lon")       -> always works, no internet needed
#   2. Online geocoders: OpenStreetMap Nominatim, Photon, Open-Meteo
#      tried on several rewrites of the query (e.g. "Barangay Mambago-A, Island
#      Garden City of Samal" -> "Mambago-A, Samal" -> "Samal"), Philippines only
#   3. Built-in list of approximate city centers (works offline)
#   4. Last resort: Samal city center, with a clear warning
# Failed lookups are NOT cached (the old code cached a failure for an hour, so one
# slow response broke the search until the cache expired).
GEO_HEADERS = {"User-Agent": "ProjectHeatShieldPH/4.0 (student research project; Streamlit app)"}
PH_LAT = (4.0, 22.0)
PH_LON = (116.0, 128.0)
DEFAULT_LATLON = (7.085, 125.715)   # Island Garden City of Samal

# Approximate city-level centers (±~5 km) — used only when the online geocoders fail.
PH_PLACES = {
    "island garden city of samal": (7.075, 125.710), "igacos": (7.075, 125.710), "samal": (7.075, 125.710),
    "davao city": (7.0731, 125.6128), "davao": (7.0731, 125.6128), "tagum": (7.4478, 125.8078),
    "panabo": (7.3083, 125.6842), "digos": (6.7496, 125.3571), "mati": (6.9550, 126.2167),
    "general santos": (6.1164, 125.1716), "koronadal": (6.5031, 124.8469), "kidapawan": (7.0083, 125.0894),
    "cagayan de oro": (8.4542, 124.6319), "iligan": (8.2280, 124.2452), "marawi": (8.0020, 124.2928),
    "malaybalay": (8.1575, 125.1278), "butuan": (8.9475, 125.5406), "surigao": (9.7840, 125.4888),
    "zamboanga": (6.9214, 122.0790), "cotabato city": (7.2236, 124.2464),
    "tacloban": (11.2543, 124.9600), "ormoc": (11.0059, 124.6075), "cebu city": (10.3157, 123.8854),
    "mandaue": (10.3236, 123.9223), "lapu-lapu": (10.3103, 123.9494), "mactan": (10.3103, 123.9494),
    "dumaguete": (9.3068, 123.3054), "bacolod": (10.6765, 122.9509), "iloilo": (10.7202, 122.5621),
    "puerto princesa": (9.7392, 118.7353), "legazpi": (13.1391, 123.7438), "naga city": (13.6218, 123.1948),
    "manila": (14.5995, 120.9842), "quezon city": (14.6760, 121.0437), "makati": (14.5547, 121.0244),
    "pasig": (14.5764, 121.0851), "taguig": (14.5176, 121.0509), "san jose": (12.3527, 121.0677),
    "baguio": (16.4023, 120.5960), "angeles": (15.1450, 120.5887), "olongapo": (14.8386, 120.2842),
    "cabanatuan": (15.4860, 120.9670), "batangas city": (13.7565, 121.0583), "lucena": (13.9373, 121.6171),
    "laoag": (18.1960, 120.5927), "vigan": (17.5747, 120.3869), "tuguegarao": (17.6132, 121.7270),
    "basco": (20.4486, 121.9702), "catarman": (12.4988, 124.6377),
}

def _in_philippines(lat, lon):
    return PH_LAT[0] <= lat <= PH_LAT[1] and PH_LON[0] <= lon <= PH_LON[1]

def parse_manual_coords(text):
    """Accepts '7.085, 125.715' (also with spaces / semicolon). Returns (lat, lon) or None."""
    if not text or not text.strip():
        return None
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if len(nums) != 2:
        return None
    lat, lon = float(nums[0]), float(nums[1])
    if not _in_philippines(lat, lon):
        lat, lon = lon, lat            # user typed lon, lat
    return (lat, lon) if _in_philippines(lat, lon) else None

def _query_variants(q):
    """Rewrites of the typed text, most specific first. Each is (text, precision)."""
    q = " ".join(q.split())
    cleaned = re.sub(r"\b(barangay|brgy\.?|bgy\.?|purok\s*\d+)\b", "", q, flags=re.I)
    cleaned = " ".join(cleaned.replace(" ,", ",").split()).strip(" ,")
    normalized = re.sub(r"island garden city of samal", "Samal, Davao del Norte", cleaned, flags=re.I)
    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    out = [(q, "exact"), (cleaned, "exact"), (normalized, "exact")]
    _norm = lambda t: re.sub(r"island garden city of samal", "Samal, Davao del Norte", t, flags=re.I)
    if len(parts) > 1:
        for coarse in (", ".join(parts[1:]), parts[-1]):
            out.append((coarse, "approximate"))
            out.append((_norm(coarse), "approximate"))
    seen, uniq = set(), []
    for text, prec in out:
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            uniq.append((text, prec))
    return uniq

def _ask_nominatim(text):
    r = requests.get("https://nominatim.openstreetmap.org/search", headers=GEO_HEADERS, timeout=8,
                     params={"q": text, "format": "json", "limit": 1, "countrycodes": "ph"})
    r.raise_for_status()
    js = r.json()
    if js:
        return float(js[0]["lat"]), float(js[0]["lon"]), js[0].get("display_name", text), "OpenStreetMap (Nominatim)"
    return None

def _ask_photon(text):
    r = requests.get("https://photon.komoot.io/api/", headers=GEO_HEADERS, timeout=8,
                     params={"q": text + ", Philippines", "limit": 5})
    r.raise_for_status()
    for f in r.json().get("features", []):
        p, (lon, lat) = f.get("properties", {}), f["geometry"]["coordinates"][:2]
        if p.get("countrycode", "").upper() == "PH" and _in_philippines(lat, lon):
            label = ", ".join(x for x in [p.get("name"), p.get("city") or p.get("county"), p.get("state")] if x)
            return float(lat), float(lon), label or text, "OpenStreetMap (Photon)"
    return None

def _ask_open_meteo(text):
    r = requests.get("https://geocoding-api.open-meteo.com/v1/search", headers=GEO_HEADERS, timeout=8,
                     params={"name": text.split(",")[0].strip(), "count": 5, "language": "en",
                             "format": "json", "countryCode": "PH"})
    r.raise_for_status()
    for hit in r.json().get("results", []) or []:
        lat, lon = hit["latitude"], hit["longitude"]
        if _in_philippines(lat, lon):
            label = ", ".join(x for x in [hit.get("name"), hit.get("admin2"), hit.get("admin1")] if x)
            return float(lat), float(lon), label, "Open-Meteo geocoder"
    return None

@st.cache_data(ttl=86400, show_spinner=False)
def _geocode_remote(query_text):
    """Online lookup. Raises LookupError on failure so Streamlit does NOT cache the failure."""
    started, net_errors = time.time(), 0
    for text, precision in _query_variants(query_text):
        for ask in (_ask_nominatim, _ask_photon, _ask_open_meteo):
            if time.time() - started > 20 or net_errors >= 3:
                raise LookupError("geocoders unreachable or too slow")
            try:
                hit = ask(text)
            except Exception:
                net_errors += 1
                continue
            if hit and _in_philippines(hit[0], hit[1]):
                return hit[0], hit[1], hit[2], hit[3], precision, text
            if ask is _ask_nominatim:
                time.sleep(1.0)        # Nominatim usage policy: max 1 request/second
    raise LookupError("no match found")

def resolve_location(query_text, manual_text=""):
    """Returns dict: lat, lon, name, source, status in {'manual','exact','approximate','offline','fallback'}, note."""
    manual = parse_manual_coords(manual_text)
    if manual:
        return {"lat": manual[0], "lon": manual[1], "name": f"{manual[0]:.4f}, {manual[1]:.4f}",
                "source": "coordinates you entered", "status": "manual", "note": ""}
    # don't hammer the geocoders on every Streamlit rerun after a failure
    fails = st.session_state.setdefault("_geo_fail", {})
    remote_error = None
    if time.time() - fails.get(query_text, 0) > 45:
        try:
            lat, lon, name, source, precision, used = _geocode_remote(query_text)
            note = "" if precision == "exact" else f"Exact place not found — matched “{used}” instead (broader area)."
            return {"lat": lat, "lon": lon, "name": name, "source": source, "status": precision, "note": note}
        except LookupError as e:
            remote_error = str(e)
            fails[query_text] = time.time()
    else:
        remote_error = "recent lookup failed"
    low = query_text.lower()
    keys = [k for k in PH_PLACES if k in low]
    if keys:
        key = max(keys, key=len)
        lat, lon = PH_PLACES[key]
        return {"lat": lat, "lon": lon, "name": key.title(), "source": "built-in city list", "status": "offline",
                "note": (f"The online geocoders could not be reached, so" if "unreachable" in (remote_error or "") or "recent" in (remote_error or "")
                         else "The online geocoders found no match for that name, so") +
                        f" the app is using the approximate center of {key.title()}. Enter coordinates in the sidebar for barangay-level accuracy."}
    lat, lon = DEFAULT_LATLON
    return {"lat": lat, "lon": lon, "name": "Island Garden City of Samal (fallback)", "source": "fallback",
            "status": "fallback",
            "note": "Could not resolve that location. Showing a fallback coordinate (Island Garden City of Samal) instead — "
                    "results do not reflect your typed location. Fix: check your internet connection, try a simpler name "
                    "(e.g. “Mambago-A, Samal”), or enter the coordinates in the sidebar."}

def find_nearest_station(lat, lon):
    dists = [(np.sqrt((c['lat']-lat)**2 + (c['lon']-lon)**2), s) for s, c in STATION_COORDS.items() if s in STATIONS_PRESENT]
    dists.sort()
    return dists[0][1]

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.markdown(f"### Signed in as `{st.session_state.get('user_name', 'ADMIN')}`")
if st.sidebar.button("Sign Out"):
    st.session_state['authenticated'] = False
    st.rerun()

sb_level_slot = st.sidebar.empty()   # filled further down, once the DepEd level is known

if IS_SYNTHETIC:
    st.sidebar.error("⚠️ 'Daily Heat Index Data.csv' not found — running on SYNTHETIC demo data. All numbers below are illustrative only.")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 Location Search")
with st.sidebar.form("universal_search_form"):
    search_query = st.text_input("Enter Barangay / Location Name:", value="Barangay Mambago-A, Island Garden City of Samal")
    manual_coords = st.text_input("Or enter coordinates (lat, lon):", value="", placeholder="e.g. 7.085, 125.715")
    target_year = st.slider("Horizon Projection Year:", min_value=2026, max_value=2040, value=2028)
    target_date = st.date_input("Exact Inspection Date:", value=datetime(2028, 5, 12))
    search_btn = st.form_submit_button("Search & Update")

if manual_coords.strip() and parse_manual_coords(manual_coords) is None:
    st.sidebar.warning("Couldn't read those coordinates — use the form 7.085, 125.715 (inside the Philippines).")
use_gps = st.sidebar.checkbox("📡 Use my device GPS", key="use_gps",
                              help="Reads your browser's location (you will be asked for permission). Overrides the name search.")
gps_fix = None
if use_gps:
    try:
        from streamlit_js_eval import get_geolocation
        _loc = get_geolocation()
        if _loc and isinstance(_loc, dict) and "coords" in _loc:
            _la, _lo = float(_loc["coords"]["latitude"]), float(_loc["coords"]["longitude"])
            _acc = _loc["coords"].get("accuracy")
            if _in_philippines(_la, _lo):
                gps_fix = (_la, _lo, _acc)
            else:
                st.sidebar.warning(f"Your device reports {_la:.3f}, {_lo:.3f}, which is outside the Philippines — ignoring it.")
        elif _loc and isinstance(_loc, dict) and "error" in _loc:
            st.sidebar.warning("Browser location was blocked or unavailable. Allow location access for this site, or type coordinates above.")
        else:
            st.sidebar.info("Waiting for your browser's location permission…")
    except ImportError:
        st.sidebar.caption("Device GPS needs one extra package: `pip install streamlit-js-eval`, then restart the app.")

geo = resolve_location(search_query, manual_coords)
if gps_fix:
    _acc_txt = f" (±{gps_fix[2]:.0f} m)" if gps_fix[2] else ""
    geo = {"lat": gps_fix[0], "lon": gps_fix[1], "name": f"My device location{_acc_txt}",
           "source": "device GPS", "status": "manual", "note": ""}
lat_res, lon_res, display_name = geo["lat"], geo["lon"], geo["name"]
_short = display_name if len(display_name) <= 48 else display_name[:45] + "…"
st.sidebar.markdown(
    f"<div style='background:#15181C;border:1px solid #262B31;border-radius:10px;padding:10px 12px;margin-top:6px;'>"
    f"<div style='font-size:13px;color:#ECEDEF;font-weight:600;'>📍 {_short}</div>"
    f"<div style='font-size:11.5px;color:#9AA0A8;margin-top:2px;'>{lat_res:.4f}, {lon_res:.4f} · {geo['source']}</div>"
    f"<a href='https://www.google.com/maps?q={lat_res:.5f},{lon_res:.5f}' target='_blank' style='font-size:11.5px;'>Check in Google Maps ↗</a>"
    f"</div>", unsafe_allow_html=True)

nearest_st = find_nearest_station(lat_res, lon_res)
year_idx = target_year - 2026
parent_station_hi = round(projections_dict[nearest_st][year_idx], 1)

# ---------------------------------------------------------
# TRUE-DAILY DERIVATION (uses real Rothfusz + VPD formulas)
# ---------------------------------------------------------
station_hist_df = df_daily[['Date', nearest_st]].dropna()
station_hist_df['Month'] = station_hist_df['Date'].dt.month
station_hist_df['Day'] = station_hist_df['Date'].dt.day

target_m, target_d = target_date.month, target_date.day
day_match = station_hist_df[(station_hist_df['Month'] == target_m) & (station_hist_df['Day'] == target_d)][nearest_st]
historical_day_baseline = day_match.mean() if not day_match.empty else station_hist_df[station_hist_df['Month'] == target_m][nearest_st].mean()
if np.isnan(historical_day_baseline):
    historical_day_baseline = 34.0

station_sen_slope = df_trend_stats.loc[df_trend_stats['Station'] == nearest_st, 'Sen_slope_C_per_yr'].values
trend_rate = float(station_sen_slope[0]) if len(station_sen_slope) else 0.15

dist_to_nearest = np.sqrt((STATION_COORDS[nearest_st]['lat'] - lat_res)**2 + (STATION_COORDS[nearest_st]['lon'] - lon_res)**2)
location_offset = round(min(3.0, max(-2.0, dist_to_nearest * 1.1 - 0.4)), 1)

np.random.seed(target_date.toordinal() + int(lat_res * 1000) % 10000)
synoptic_noise = np.random.normal(loc=0.0, scale=0.85)

years_elapsed = target_year - 2015
daily_target_hi = round(historical_day_baseline + trend_rate * years_elapsed + location_offset + synoptic_noise, 1)

def solve_temp_for_hi(hi_c_target, rh_guess=70.0):
    def obj(t_c):
        return compute_heat_index_c(t_c, rh_guess) - hi_c_target
    try:
        return float(optimize.brentq(obj, 15.0, 55.0))
    except Exception:
        return hi_c_target * 0.76 + 2.5

derived_air_temp = round(solve_temp_for_hi(daily_target_hi), 1)
derived_rh = round(min(98.0, max(45.0, 100.0 - (daily_target_hi - derived_air_temp) * 4.2)), 1)
derived_wind_speed = round(max(1.2, 12.5 - (daily_target_hi * 0.18) + np.random.normal(0, 1.1)), 1)
derived_vpd = round(float(compute_vpd(derived_air_temp, derived_rh)), 2)
derived_uv_index = round(min(12.0, max(3.0, (derived_air_temp / 3.0) + (target_m if target_m <= 6 else 12 - target_m) * 0.4)), 1)

selected_info = df_impacts[df_impacts['Station'] == nearest_st].iloc[0]
selected_trend = df_trend_stats[df_trend_stats['Station'] == nearest_st].iloc[0]
selected_validation = df_validation[df_validation['Station'] == nearest_st].iloc[0]

gev_row = selected_info
t_from_base = target_year - gev_row['GEV_base_year'] if not np.isnan(gev_row['GEV_base_year']) else 0
gev_params_sel = {'mu0': gev_row['GEV_mu0'], 'mu1': gev_row['GEV_mu1'], 'sigma': gev_row['GEV_sigma'], 'xi': gev_row['GEV_xi']}
return_levels = {}
if not any(np.isnan(list(gev_params_sel.values()))):
    for T in [10, 25, 50]:
        return_levels[T] = round(gev_return_level(gev_params_sel, t_from_base, T), 2)

lat_grid = np.linspace(4.8, 21.5, 75)
lon_grid = np.linspace(116.5, 127.0, 65)
grid_lats, grid_lons = np.meshgrid(lat_grid, lon_grid)
grid_lats_flat, grid_lons_flat = grid_lats.flatten(), grid_lons.flatten()

st_lats = np.array([STATION_COORDS[s]['lat'] for s in STATIONS_PRESENT])
st_lons = np.array([STATION_COORDS[s]['lon'] for s in STATIONS_PRESENT])
st_his = np.array([projections_dict[s][year_idx] for s in STATIONS_PRESENT])

dists_matrix = np.sqrt((grid_lats_flat[:, None] - st_lats[None, :])**2 + (grid_lons_flat[:, None] - st_lons[None, :])**2)
weights_matrix = 1.0 / (dists_matrix**2.2 + 1e-5)
base_interp = np.sum(weights_matrix * st_his[None, :], axis=1) / np.sum(weights_matrix, axis=1)
min_dists = np.min(dists_matrix, axis=1)
valid_mask = min_dists < 3.2
dense_heat_points = np.column_stack([
    np.round(grid_lats_flat[valid_mask], 4), np.round(grid_lons_flat[valid_mask], 4),
    np.round(base_interp[valid_mask] + synoptic_noise, 1)
]).tolist()

# ---------------------------------------------------------
# POLICY THRESHOLDS
# ---------------------------------------------------------
# DepEd 4H Framework — DepEd Memo No. 064, s. 2026 (transcribed from the
# official DepEd infographic). The infographic prints whole-degree ranges
# (27–32, 33–41). Heat index is continuous, so this app applies the cutoffs
# at exactly 27, 33 and 42 °C: 32.6 °C counts as Hinay, 41.6 °C as Hinga.
FOURH = [
    {"key": "HAYO", "label": "Hayo", "eie": "Manageable Conditions", "range": "Below 27°C", "lo": -1000.0, "hi": 27.0,
     "color": "#3ADB86", "icon": "☀️",
     "headline": "Continue face-to-face classes with heat-adapted measures, as needed.",
     "detail": "Continue face-to-face classes with heat-adapted measures, as needed.",
     "actions": [("🪑", "Conduct regular classes"), ("☀️", "Outdoor activities during cooler periods"),
                 ("🚰", "Safe drinking water"), ("💧", "Hydration breaks"), ("🌀", "Ventilated classrooms")]},
    {"key": "HINAY", "label": "Hinay", "eie": "Caution", "range": "27°C – 32°C", "lo": 27.0, "hi": 33.0,
     "color": "#FFC83D", "icon": "🌤️",
     "headline": "Reduce face-to-face hours or shift to hybrid or home-based learning.",
     "detail": "Reduce face-to-face hours or shift to hybrid or home-based learning.",
     "actions": [("🕐", "Modified schedules (e.g., 6–10 a.m.; 2–6 p.m.)"), ("🏠", "Hybrid or structured home-based learning"),
                 ("📋", "Other protective strategies as needed, depending on the severity of the situation and available learning resources")]},
    {"key": "HINGA", "label": "Hinga", "eie": "Extreme Caution", "range": "33°C – 41°C", "lo": 33.0, "hi": 42.0,
     "color": "#FF7A45", "icon": "🔆",
     "headline": "Suspend face-to-face classes and reduce academic demand.",
     "detail": "Suspend face-to-face classes and reduce academic demand.",
     "actions": [("📖", "Light learning tasks"), ("📄", "Learning packets"), ("📺", "DepEd TV / radio"), ("❤️", "Well-being check-ins")]},
    {"key": "HINTO", "label": "Hinto", "eie": "Danger and Extreme Danger", "range": "42°C and above", "lo": 42.0, "hi": 1000.0,
     "color": "#FF4D5E", "icon": "🌡️",
     "headline": "Suspend all academic activities, including face-to-face and home-based learning.",
     "detail": "Suspend all academic activities, including face-to-face and home-based learning.",
     "actions": [("🚫", "Suspend activities"), ("👪", "Maintain communication with learners and families"),
                 ("🛡️", "Safety checks"), ("🤝", "Referrals for support")]},
]
HINTO_THRESHOLD = 42.0
OTHER_FACTORS = [("🚨", "Severity of disruption"), ("✅", "Safety and well-being"), ("📚", "Learning resources and readiness"),
                 ("📍", "Local conditions"), ("🌀", "Ventilation"), ("💧", "Water availability"),
                 ("☀️", "Heat exposure"), ("🏠", "Learners' home circumstances")]
SDS_NOTE = ("Classes may be suspended before the heat index reaches 42°C if the heat becomes unbearable, "
            "based on local conditions. The school head must coordinate with the Schools Division Superintendent (SDS) "
            "and shift to appropriate distance learning modalities.")
DEPED_SOURCE = "DepEd Memorandum No. 064, s. 2026, p. 5, Section 5.2.2, Learning Delivery Approaches and the 4H Scenario Continuum"
DEPED_MEMO_URL = "https://www.deped.gov.ph/wp-content/uploads/DM_s2026_064r.pdf"

def classify_4h(hi):
    """Return the DepEd 4H zone dict for a heat index in °C (None if missing)."""
    if hi is None or (isinstance(hi, float) and np.isnan(hi)):
        return None
    for z in FOURH:
        if z["lo"] <= hi < z["hi"]:
            return z
    return FOURH[-1]

deped_cat = classify_4h(float(daily_target_hi))

# Flashing 4H level in the sidebar. Hinto flashes fast, Hinga pulses, Hinay breathes gently, Hayo stays steady.
_FLASH = {"HAYO": ("none", "0s"), "HINAY": ("none", "0s"), "HINGA": ("fhPulse", "3.2s"), "HINTO": ("fhBlink", "1.1s")}
_anim, _dur = _FLASH[deped_cat["key"]]
_c = deped_cat["color"]
sb_level_slot.markdown(f"""
<style>
@keyframes fhBlink {{ 0%,100% {{ opacity:1; box-shadow:0 0 22px 4px {_c}cc; }} 50% {{ opacity:.35; box-shadow:0 0 0 0 {_c}00; }} }}
@keyframes fhPulse {{ 0%,100% {{ box-shadow:0 0 0 0 {_c}99; }} 50% {{ box-shadow:0 0 18px 6px {_c}55; }} }}
.sb-level {{ border:2px solid {_c}; background:{_c}22; border-radius:14px; padding:14px 16px; margin:10px 0 6px;
             animation:{_anim} {_dur} ease-in-out infinite; }}
@media (prefers-reduced-motion: reduce) {{ .sb-level {{ animation:none; }} }}
.sb-level .k {{ font-size:10px; letter-spacing:.1em; font-weight:700; color:#9AA0A8; text-transform:uppercase; }}
.sb-level .n {{ font-family:'Space Grotesk',sans-serif; font-size:30px; font-weight:700; color:{_c}; line-height:1.1; letter-spacing:.03em; }}
.sb-level .e {{ font-size:12px; font-weight:600; color:{_c}; margin-bottom:6px; }}
.sb-level .h {{ font-family:'JetBrains Mono',monospace; font-size:20px; font-weight:700; color:#ECEDEF; }}
.sb-level .d {{ font-size:12px; color:#C9CDD2; margin-top:6px; line-height:1.45; }}
</style>
<div class="sb-level">
  <div class="k">DepEd 4H level · {target_year}</div>
  <div class="n">{deped_cat['icon']} {deped_cat['label'].upper()}</div>
  <div class="e">{deped_cat['eie']} · {deped_cat['range']}</div>
  <div class="h">{daily_target_hi} °C</div>
  <div class="d">{deped_cat['headline']}</div>
</div>
""", unsafe_allow_html=True)

# DOLE cutoffs are NOT part of the DepEd memo — verify against the actual advisory text.
DOLE_SHADED_REST_THRESHOLD = 33.0
DOLE_STOPPAGE_THRESHOLD = 42.0

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown("<h1>MOROPORO</h1>", unsafe_allow_html=True)
if IS_SYNTHETIC:
    st.markdown('<div class="methodnote">⚠️ No real dataset found on disk — every figure on this page is generated from synthetic placeholder data. Drop <code>Daily Heat Index Data.csv</code> in the app directory to see real output.</div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="target-badge">
    <div>
        <span style="font-size:11px; color:#9AA0A8; text-transform:uppercase; font-weight:700; letter-spacing:0.06em;">Active Daily Inspection Target</span><br/>
        <strong style="font-size:22px; color:#ECEDEF;">📍 {search_query}</strong><br/>
        <span style="font-size:12.5px; color:#9AA0A8;">{target_date.strftime('%B %d, %Y')} · {lat_res:.4f}, {lon_res:.4f} · nearest station: {nearest_st}</span>
    </div>
    <div style="text-align:right;">
        <span style="font-size:11px; color:#9AA0A8; font-weight:700; text-transform:uppercase; letter-spacing:0.06em;">Projected Heat Index ({target_year})</span><br/>
        <strong style="font-size:38px; color:{deped_cat['color']}; font-weight:800;">{daily_target_hi} °C</strong>
    </div>
</div>
""", unsafe_allow_html=True)

if geo["status"] in ("approximate", "offline"):
    st.warning("⚠️ " + geo["note"])
elif geo["status"] == "fallback":
    st.error("⚠️ " + geo["note"])

# ---------------------------------------------------------
# METRIC CARDS
# ---------------------------------------------------------
st.markdown("### Micro-Climate Diagnostics")

# ---- DepEd 4H level banner (Hayo / Hinay / Hinga / Hinto) ----
_bc = deped_cat["color"]
st.markdown(
    f"<div style='display:inline-flex;align-items:center;flex-wrap:wrap;gap:10px;padding:7px 16px;border-radius:999px;"
    f"border:1px solid {_bc}66;background:{_bc}14;margin:2px 0 16px;font-size:13px;'>"
    f"<b style='color:{_bc};letter-spacing:.07em;'>{deped_cat['label'].upper()}</b>"
    f"<span style='color:#9AA0A8;'>{deped_cat['eie']} · {deped_cat['range']}</span>"
    f"<span style='color:#ECEDEF;'>{deped_cat['headline']}</span></div>",
    unsafe_allow_html=True)
m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
m_col1.metric("Projected Heat Index", f"{daily_target_hi} °C")
m_col2.metric("Air Temperature (implied)", f"{derived_air_temp} °C")
m_col3.metric("Relative Humidity (implied)", f"{derived_rh} %")
m_col4.metric("Vapor Pressure Deficit", f"{derived_vpd} kPa")
m_col5.metric("Solar UV Index (heuristic)", f"{derived_uv_index} UV")

with st.expander("How these values are derived"):
    st.markdown(f"**Projected Heat Index ({daily_target_hi} °C)** — NOAA Rothfusz regression solved against the station's day-of-year historical baseline, Sen's-slope trend rate, spatial offset and synoptic noise. DepEd 4H level: **{deped_cat['label'].upper()}** ({deped_cat['range']}).")
    st.markdown("**Air temperature** — solved numerically so that `Rothfusz(T, RH) == Projected HI` holds exactly, at an assumed 70% RH prior.")
    st.markdown("**Relative humidity** — at high RH sweat cannot evaporate efficiently, so the body's main cooling mechanism is suppressed. This is why Heat Index tracks physiological risk better than raw temperature.")
    st.markdown(r"""**Vapor pressure deficit** — $e_s = 0.61078 \exp\left(\frac{17.27T}{T+237.3}\right)$, $VPD = e_s(1 - RH/100)$. VPD above 1.6 kPa induces stomatal closure in rice, corn and durian.""")
    st.markdown("**Solar UV index** — a heuristic proxy scaled from temperature and month, not a satellite-retrieved estimate. Flag it as an approximation if cited.")

# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------
tab_map, tab_search, tab_spatiotemporal, tab_projections, tab_deped, tab_pdf = st.tabs([
    "Weather Map", "Historical Search", "Spatiotemporal Matrix",
    "Horizon Projections", "DepEd 4H Framework", "PDF Generator"
])

with tab_map:
    st.subheader(f"Continuous Thermal Surface Map ({target_date.strftime('%B %d, %Y')} — Horizon {target_year})")
    col_map_opts, col_map_display = st.columns([1, 2.4])
    with col_map_opts:
        st.markdown(f"#### {search_query}")
        st.caption(f"Nearest synoptic station: {nearest_st}")
        r1a, r1b = st.columns(2)
        r1a.metric("Target Heat Index", f"{daily_target_hi} °C")
        r1b.metric("Station Baseline", f"{parent_station_hi} °C")
        r2a, r2b = st.columns(2)
        r2a.metric("Labor Capacity Loss", f"{selected_info['Outdoor_Labor_Capacity_Loss_pct']} %")
        r2b.metric("ER Surge Risk", f"+{selected_info['Hospital_ER_Heat_Surge_Risk_pct']} %")
        if return_levels:
            st.caption("GEV return levels · " + " · ".join(f"{T}-yr: {z} °C" for T, z in return_levels.items()))
    with col_map_display:
        stations_json = json.dumps([
            {"n": s_name, "lat": float(STATION_COORDS[s_name]["lat"]), "lon": float(STATION_COORDS[s_name]["lon"]),
             "v": round(float(projections_dict[s_name][year_idx]), 1)} for s_name in STATIONS_PRESENT])
        target_json = json.dumps({"lat": float(lat_res), "lon": float(lon_res), "name": str(search_query), "hi": float(daily_target_hi)})
        THERMAL_HTML = """
        <!DOCTYPE html><html><head><meta charset="utf-8" />
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <style>
          html,body{margin:0;padding:0;background:#0B0D10;}
          #map{width:100%;height:620px;border-radius:14px;background:#0B0D10;}
          .leaflet-tile-pane{filter:invert(1) hue-rotate(180deg) brightness(.72) contrast(.95) saturate(.35);}
          .leaflet-popup-content-wrapper{background:#15181C;color:#ECEDEF;border-radius:10px;border:1px solid #262B31;}
          .leaflet-popup-tip{background:#15181C;}
          .leaflet-tooltip{background:#15181C;color:#ECEDEF;border:1px solid #333941;box-shadow:none;}
          .legend{background:rgba(15,17,20,.88);color:#ECEDEF;padding:10px 14px 8px;border-radius:10px;border:1px solid #262B31;
                  font:11px -apple-system,Segoe UI,Inter,sans-serif;width:260px;}
          .legend b{font-size:11px;letter-spacing:.04em;}
          .bar{position:relative;height:10px;border-radius:5px;margin:8px 0 18px;}
          .bar i{position:absolute;top:12px;transform:translateX(-50%);font-style:normal;color:#9AA0A8;font-size:10px;}
          .bar u{position:absolute;top:-3px;width:1px;height:16px;background:rgba(255,255,255,.55);text-decoration:none;}
          .legend small{color:#9AA0A8;font-size:10px;}
          .phbtn{background:#15181C;color:#ECEDEF;border:1px solid #333941;border-radius:6px;padding:5px 9px;font:600 11px sans-serif;cursor:pointer;}
        </style></head><body><div id="map"></div>
        <script>
          var STATIONS = __STATIONS__;
          var TARGET = __TARGET__;
          var VMIN = 24, VMAX = 46;
          var STOPS = [[0,[13,27,110]],[0.18,[31,94,255]],[0.34,[25,211,232]],[0.48,[54,224,138]],
                       [0.62,[247,225,74]],[0.76,[255,154,31]],[0.90,[255,59,47]],[1,[255,222,222]]];
          function ramp(t){
            t = Math.max(0, Math.min(1, t));
            for (var i = 1; i < STOPS.length; i++){
              if (t <= STOPS[i][0]){
                var a = STOPS[i-1], b = STOPS[i], f = (t - a[0]) / (b[0] - a[0]);
                return [a[1][0] + (b[1][0]-a[1][0])*f, a[1][1] + (b[1][1]-a[1][1])*f, a[1][2] + (b[1][2]-a[1][2])*f];
              }
            }
            return STOPS[STOPS.length-1][1];
          }

          var map = L.map('map', {minZoom:5, maxZoom:14, zoomSnap:0.5}).setView([TARGET.lat, TARGET.lon], 8);
          L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',
                      {maxZoom:19, attribution:'&copy; OpenStreetMap contributors'}).addTo(map);

          /* Thermal surface: inverse-distance interpolation of the station values, recomputed for every
             pixel block in view, so it stays continuous and sharp at any zoom level. */
          var cvs = document.createElement('canvas');
          cvs.style.cssText = 'position:absolute;left:0;top:0;width:100%;height:100%;z-index:450;pointer-events:none;opacity:.8;';
          map.getContainer().appendChild(cvs);
          var CELL = 6, queued = false;
          function draw(){
            queued = false;
            var sz = map.getSize(), w = Math.ceil(sz.x / CELL), h = Math.ceil(sz.y / CELL);
            cvs.width = w; cvs.height = h;
            var ctx = cvs.getContext('2d'), img = ctx.createImageData(w, h), d = img.data;
            for (var y = 0; y < h; y++){
              for (var x = 0; x < w; x++){
                var ll = map.containerPointToLatLng([x*CELL + CELL/2, y*CELL + CELL/2]);
                var k = Math.cos(ll.lat * Math.PI / 180), num = 0, den = 0, dmin = 1e9;
                for (var j = 0; j < STATIONS.length; j++){
                  var s = STATIONS[j], dy = ll.lat - s.lat, dx = (ll.lng - s.lon) * k, d2 = dx*dx + dy*dy;
                  if (d2 < dmin) dmin = d2;
                  var wt = 1 / Math.pow(d2 + 0.05, 1.25);
                  num += wt * s.v; den += wt;
                }
                var v = num / den, dist = Math.sqrt(dmin);
                var alpha = Math.max(0, Math.min(1, 1 - (dist - 1.2) / 2.2));
                var c = ramp((v - VMIN) / (VMAX - VMIN)), i = (y*w + x) * 4;
                d[i] = c[0]; d[i+1] = c[1]; d[i+2] = c[2]; d[i+3] = alpha * 255;
              }
            }
            ctx.putImageData(img, 0, 0);
          }
          function schedule(){ if (!queued){ queued = true; requestAnimationFrame(draw); } }
          map.on('move zoom resize', schedule);
          draw();

          /* station dots (above the thermal layer) */
          STATIONS.forEach(function(s){
            L.circleMarker([s.lat, s.lon], {pane:'markerPane', radius:4, color:'#ffffff', weight:1.2, fillColor:'#0B0D10', fillOpacity:.9})
              .addTo(map).bindTooltip(s.n + ': ' + s.v + ' °C');
          });
          /* the searched location: one static marker */
          L.circleMarker([TARGET.lat, TARGET.lon], {pane:'markerPane', radius:8, color:'#ffffff', weight:3, fillColor:'#0B0D10', fillOpacity:1})
            .addTo(map).bindPopup('<b>' + TARGET.name + '</b><br/>' + TARGET.hi + ' °C').openPopup();

          /* legend with the DepEd 27 / 33 / 42 cutoffs */
          var legend = L.control({position:'bottomleft'});
          legend.onAdd = function(){
            var el = L.DomUtil.create('div', 'legend');
            var stops = STOPS.map(function(s){ var c = s[1]; return 'rgb(' + c.map(Math.round).join(',') + ') ' + (s[0]*100) + '%'; }).join(',');
            function pos(v){ return ((v - VMIN) / (VMAX - VMIN) * 100) + '%'; }
            var ticks = [24, 27, 33, 42, 46].map(function(v){ return '<i style="left:' + pos(v) + '">' + v + '</i>'; }).join('');
            var cuts = [27, 33, 42].map(function(v){ return '<u style="left:' + pos(v) + '"></u>'; }).join('');
            el.innerHTML = '<b>PROJECTED HEAT INDEX (°C)</b><div class="bar" style="background:linear-gradient(90deg,' + stops + ')">' + cuts + ticks + '</div>' +
                           '<small>White ticks = DepEd 4H cutoffs. Surface interpolated from ' + STATIONS.length + ' PAGASA stations.</small>';
            return el;
          };
          legend.addTo(map);

          /* whole-Philippines button */
          var home = L.control({position:'topleft'});
          home.onAdd = function(){
            var b = L.DomUtil.create('button', 'phbtn'); b.innerHTML = 'Whole PH';
            L.DomEvent.on(b, 'click', function(e){ L.DomEvent.stopPropagation(e); map.setView([12.5, 122.5], 6); });
            return b;
          };
          home.addTo(map);
        </script></body></html>
        """
        components.html(THERMAL_HTML.replace("__STATIONS__", stations_json).replace("__TARGET__", target_json), height=640)

with tab_search:
    st.subheader("Historical Daily Records Search")
    col_s1, col_s2 = st.columns([1, 2])
    with col_s1:
        min_d, max_d = df_daily['Date'].min(), df_daily['Date'].max()
        search_date = st.date_input("Select Historical Record Date:", value=min_d, min_value=min_d, max_value=max_d)
    with col_s2:
        df_daily['Date_Only'] = df_daily['Date'].dt.date
        hist_records = df_daily[df_daily['Date_Only'] == search_date]
        if not hist_records.empty:
            row_hist = hist_records.iloc[0]
            res_data = []
            for st_name in STATIONS_PRESENT:
                val = row_hist[st_name] if st_name in row_hist else np.nan
                tier = 'N/A' if np.isnan(val) else classify_4h(float(val))['label']
                res_data.append({'Synoptic Station': st_name, 'Recorded Heat Index (°C)': val, 'DepEd 4H Level': tier})
            st.dataframe(pd.DataFrame(res_data), width='stretch', height=350)
        else:
            st.warning("No historical records found for selected date.")

with tab_spatiotemporal:
    st.subheader("Spatiotemporal Heat Index Matrix")
    col_temp1, col_temp2 = st.columns(2)
    with col_temp1:
        pivot_annual = df_daily.groupby('Year')[STATIONS_PRESENT].mean().T
        fig_heatmap = px.imshow(pivot_annual.values, labels=dict(x="Year", y="PAGASA Station", color="Mean HI (°C)"),
            x=[str(y) for y in pivot_annual.columns], y=list(pivot_annual.index),
            color_continuous_scale="YlOrRd", text_auto=".1f", aspect="auto", height=400)
        fig_heatmap.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#ECEDEF"))
        st.plotly_chart(fig_heatmap, width='stretch')
    with col_temp2:
        month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        monthly_avg = df_daily.groupby('Month')[nearest_st].mean()
        fig_season = px.bar(x=month_names, y=monthly_avg.values, labels={'x':'Month','y':'Mean Heat Index (°C)'},
            color=monthly_avg.values, color_continuous_scale=[[0,'#34C759'],[1,'#FF453A']],
            title=f"Monthly Seasonal Cycle: {nearest_st}")
        fig_season.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#ECEDEF"), height=400)
        st.plotly_chart(fig_season, width='stretch')

with tab_projections:
    st.subheader("Non-Stationary Linear Trajectory Projections (2026–2040)")
    st.markdown('<div class="methodnote">Projection method: per-station linear regression (np.polyfit) on annual peak Heat Index, cross-checked against Sen\'s slope (robust to outliers) and Mann-Kendall significance below. This is a linear trajectory model — not a trained neural network.</div>', unsafe_allow_html=True)
    col_proj1, col_proj2 = st.columns(2)
    with col_proj1:
        hist_max_series = df_daily.groupby('Year')[nearest_st].max()
        proj_years_grid = np.arange(2026, 2041)
        proj_vals = projections_dict[nearest_st]
        fig_proj = go.Figure()
        fig_proj.add_trace(go.Scatter(x=hist_max_series.index, y=hist_max_series.values, mode='lines+markers', name='Hist Max HI', line=dict(color='#4FD8D0', width=3)))
        fig_proj.add_trace(go.Scatter(x=proj_years_grid, y=proj_vals, mode='lines+markers', name='Proj Max HI (linear)', line=dict(color='#FF9F0A', width=3, dash='dash')))
        _ys = [float(v) for v in hist_max_series.values] + [float(v) for v in proj_vals]
        _y_lo, _y_hi = min(_ys) - 2.0, max(_ys) + 2.0
        for _z in FOURH:
            _b0, _b1 = max(_z['lo'], _y_lo), min(_z['hi'], _y_hi)
            if _b1 > _b0:
                fig_proj.add_hrect(y0=_b0, y1=_b1, fillcolor=_z['color'], opacity=0.10, line_width=0, layer="below")
                fig_proj.add_annotation(xref="paper", x=0.995, y=(_b0 + _b1) / 2, text=_z['label'].upper(),
                                        showarrow=False, xanchor="right", font=dict(color=_z['color'], size=11))
        fig_proj.update_yaxes(range=[_y_lo, _y_hi])
        fig_proj.update_layout(title=f"Annual Peak HI Trajectory: {nearest_st}", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#ECEDEF"), height=400)
        st.plotly_chart(fig_proj, width='stretch')
        st.markdown(f"**Mann-Kendall:** Z={selected_trend['MK_Z']}, p={selected_trend['MK_p']} → *{selected_trend['MK_trend']}*")
        st.markdown(f"**Sen's slope:** {selected_trend['Sen_slope_C_per_yr']} °C/yr  |  **Linear slope:** {selected_trend['Linear_slope_C_per_yr']} °C/yr")
    with col_proj2:
        st.markdown(f"#### All Station Projections ({target_year})")
        df_table = df_impacts.copy()
        df_table['Selected_Year_HI'] = [round(projections_dict[s][year_idx], 1) for s in df_table['Station']]
        st.dataframe(df_table[['Station','Region','Altitude_m','Selected_Year_HI','Outdoor_Labor_Capacity_Loss_pct','Hospital_ER_Heat_Surge_Risk_pct']]
                     .sort_values(by='Selected_Year_HI', ascending=False), width='stretch', height=400)
        st.markdown("#### Mann-Kendall / Sen's Slope — all stations")
        st.dataframe(df_trend_stats.sort_values('Sen_slope_C_per_yr', ascending=False), width='stretch', height=250)

DEPED_CSS = """
<style>
.fh-hero{position:relative;background:var(--surface);border:1px solid var(--border);border-radius:18px;padding:28px 32px 22px;overflow:hidden;margin:6px 0 18px;box-shadow:0 18px 44px rgba(0,0,0,.35)}
.fh-hero-bar{position:absolute;top:0;left:0;right:0;height:3px}
.fh-hero-top{display:flex;justify-content:space-between;align-items:center;gap:18px;flex-wrap:wrap}
.fh-eyebrow{font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:var(--text-dim);font-weight:600}
.fh-big{font-family:'JetBrains Mono',monospace;font-size:64px;font-weight:700;line-height:1.05;letter-spacing:-.03em}
.fh-big span{font-size:26px;margin-left:4px;opacity:.7}
.fh-pill{display:inline-flex;align-items:center;gap:10px;border:1px solid;border-radius:999px;padding:10px 22px;font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:22px;letter-spacing:.05em}
.fh-pill em{font-style:normal;font-size:13px;font-weight:500;opacity:.8;letter-spacing:0}
.fh-headline{font-family:'Space Grotesk',sans-serif;font-size:21px;font-weight:600;margin:18px 0 4px;color:var(--text)}
.fh-detail{color:var(--text-dim);font-size:14.5px;line-height:1.6;max-width:880px}
.fh-scale{position:relative;margin-top:46px}
.fh-track{display:flex;height:16px;border-radius:9px;overflow:hidden}
.fh-seg{height:100%;display:flex;align-items:center;justify-content:center;font-size:9px;font-weight:700;letter-spacing:.12em;color:rgba(0,0,0,.62)}
.fh-marker{position:absolute;top:-8px;width:3px;height:32px;background:#fff;border-radius:2px;transform:translateX(-50%);box-shadow:0 0 0 2px rgba(0,0,0,.55)}
.fh-marker b{position:absolute;bottom:36px;left:50%;transform:translateX(-50%);background:#fff;color:#111;font-family:'JetBrains Mono',monospace;font-size:12px;padding:2px 8px;border-radius:6px;white-space:nowrap}
.fh-ticks{position:relative;height:18px;margin-top:8px;font-size:11px;color:var(--text-faint);font-family:'JetBrains Mono',monospace}
.fh-ticks span{position:absolute;transform:translateX(-50%)}
.fh-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:6px 0 18px}
@media(max-width:1100px){.fh-grid{grid-template-columns:repeat(2,1fr)}}
.fh-card{position:relative;background:var(--surface);border:1px solid var(--border);border-top-width:3px;border-radius:14px;padding:20px 18px 16px;opacity:.6;transition:all .2s}
.fh-card.on{opacity:1;transform:translateY(-3px)}
.fh-card .nm{font-family:'Space Grotesk',sans-serif;font-size:27px;font-weight:700;letter-spacing:.02em;line-height:1.1}
.fh-card .rg{font-size:12px;font-weight:600;margin:2px 0 12px;font-family:'JetBrains Mono',monospace}
.fh-card .hd{font-size:13.5px;font-weight:600;color:var(--text);margin-bottom:6px}
.fh-card .ds{font-size:12.5px;color:var(--text-dim);line-height:1.5}
.fh-card ul{list-style:none;padding:0;margin:12px 0 0}
.fh-card li{display:flex;gap:9px;font-size:12.5px;color:var(--text-dim);padding:6px 0;border-top:1px solid var(--border)}
.fh-here{position:absolute;top:12px;right:12px;font-size:9.5px;font-weight:700;letter-spacing:.1em;padding:3px 9px;border-radius:999px;color:#14100D}
.fh-chips{display:grid;grid-template-columns:repeat(auto-fill,minmax(92px,1fr));gap:8px;margin:8px 0 14px}
.fh-chip{border:1px solid;border-radius:10px;padding:8px 10px;display:flex;flex-direction:column;gap:1px}
.fh-chip span{font-size:11px;color:var(--text-dim)}
.fh-chip b{font-family:'JetBrains Mono',monospace;font-size:16px}
.fh-chip em{font-style:normal;font-size:10.5px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:var(--text-dim)}
.fh-sds{background:rgba(79,216,208,0.07);border:1px solid rgba(79,216,208,0.25);border-radius:14px;padding:16px 20px;margin:6px 0 10px;color:var(--text-dim);font-size:14px;line-height:1.55}
.fh-sds b{color:var(--text)}
.fh-other{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:16px 18px;margin:6px 0 12px}
.fh-other-t{font-size:11px;font-weight:700;letter-spacing:.09em;color:var(--teal);margin-bottom:10px}
.fh-factors{display:grid;grid-template-columns:repeat(auto-fill,minmax(130px,1fr));gap:10px}
.fh-factor{display:flex;flex-direction:column;align-items:center;text-align:center;gap:4px;font-size:12px;color:var(--text-dim)}
.fh-factor span{font-size:22px}
.fh-important{background:rgba(255,93,93,0.10);border:1px solid rgba(255,93,93,0.45);border-radius:14px;padding:14px 18px;margin:0 0 8px;color:var(--text);font-size:14px;line-height:1.55}
.fh-source{font-size:11.5px;color:var(--text-faint);margin-bottom:18px}
</style>
"""

with tab_deped:
    st.markdown(DEPED_CSS, unsafe_allow_html=True)
    st.subheader("DepEd 4H Heat Framework")
    st.markdown(
        '<div class="methodnote">Anchored to <b>DepEd Memo No. 064, s. 2026</b> — the 4H framework: '
        '<b>Hayo</b> – Manageable (below 27°C) · <b>Hinay</b> – Caution (27–32°C) · <b>Hinga</b> – Extreme Caution (33–41°C) · <b>Hinto</b> – Danger and Extreme Danger (42°C and above). '
        'Cutoffs are applied at 27, 33 and 42°C. Transcribed from the DepEd infographic (Heat Index Guide for Schools During Super El Niño) — check it against the '
        f'<a href="{DEPED_MEMO_URL}" target="_blank">full memo</a> before citing it in your paper.</div>',
        unsafe_allow_html=True)

    # ---------- 1 · INPUT ----------
    st.markdown("#### 1 · Check a heat index")
    ci1, ci2 = st.columns([1.1, 1])
    with ci1:
        in_mode = st.radio(
            "What do you want to check?",
            ["Projected value for my location", "Enter a heat index", "Enter temperature + humidity"],
            key="fh_mode")
        local_unbearable = st.checkbox("Local conditions feel unbearable (for SDS coordination)", key="fh_local")

    if in_mode == "Enter a heat index":
        with ci2:
            hi_check = st.slider(
                "Heat index (°C)", min_value=15.0, max_value=60.0,
                value=float(np.clip(round(float(daily_target_hi) * 2) / 2, 15.0, 60.0)),
                step=0.5, key="fh_hi")
        basis = "Heat index you entered"
    elif in_mode == "Enter temperature + humidity":
        with ci2:
            t_in = st.number_input("Air temperature (°C)", min_value=20.0, max_value=50.0, value=33.0, step=0.5, key="fh_t")
            rh_in = st.number_input("Relative humidity (%)", min_value=20.0, max_value=100.0, value=70.0, step=1.0, key="fh_rh")
        hi_check = float(compute_heat_index_c(t_in, rh_in))
        basis = f"NOAA Rothfusz from {t_in:.1f}°C and {rh_in:.0f}% RH"
    else:
        hi_check = float(daily_target_hi)
        basis = f"Projected for {search_query} · {target_date.strftime('%b %d, %Y')}"
        with ci2:
            st.caption("Uses the location, year and date set in the sidebar.")

    cat = classify_4h(hi_check)
    idx = [z["key"] for z in FOURH].index(cat["key"])

    AX_MIN, AX_MAX = 20.0, 50.0

    def _pos(v):
        return min(max((v - AX_MIN) / (AX_MAX - AX_MIN), 0.0), 1.0) * 100.0

    segs = "".join(
        f'<div class="fh-seg" style="width:{(min(z["hi"], AX_MAX) - max(z["lo"], AX_MIN)) / (AX_MAX - AX_MIN) * 100:.3f}%;background:{z["color"]}">{z["label"].upper()}</div>'
        for z in FOURH)
    ticks = "".join(f'<span style="left:{_pos(v):.2f}%">{v:.0f}°</span>' for v in (27, 33, 42))

    st.markdown("".join([
        f'<div class="fh-hero" style="border-color:{cat["color"]}66;">',
        f'<div class="fh-hero-bar" style="background:{cat["color"]}"></div>',
        '<div class="fh-hero-top"><div>',
        f'<div class="fh-eyebrow">{basis}</div>',
        f'<div class="fh-big" style="color:{cat["color"]}">{hi_check:.1f}<span>°C</span></div></div>',
        f'<div class="fh-pill" style="background:{cat["color"]}22;border-color:{cat["color"]};color:{cat["color"]}">',
        f'<span>{cat["icon"]}</span>{cat["label"].upper()}<em>{cat["range"]}</em></div></div>',
        f'<div class="fh-headline">{cat["headline"]}</div>',
        f'<div class="fh-detail">{cat["detail"]}</div>',
        f'<div class="fh-scale"><div class="fh-track">{segs}</div>',
        f'<div class="fh-marker" style="left:{_pos(hi_check):.2f}%"><b>{hi_check:.1f}°</b></div></div>',
        f'<div class="fh-ticks">{ticks}</div>',
        '</div>']), unsafe_allow_html=True)

    # ---------- 2 · INTERPRETATION ----------
    st.markdown("#### 2 · Interpretation")
    notes = []
    if idx < len(FOURH) - 1:
        nxt = FOURH[idx + 1]
        notes.append(f"**{nxt['lo'] - hi_check:.1f} °C** below the next level, **{nxt['label']}** (≥ {nxt['lo']:.0f}°C).")
    else:
        notes.append(f"**{hi_check - HINTO_THRESHOLD:.1f} °C** above the Hinto threshold of {HINTO_THRESHOLD:.0f}°C.")
    if idx > 0:
        notes.append(f"**{hi_check - cat['lo']:.1f} °C** above the floor of {cat['label']} ({cat['lo']:.0f}°C).")
    near = min(abs(hi_check - b) for b in (27.0, 33.0, 42.0))
    if near <= 1.0:
        notes.append("This value is within 1°C of a category boundary — treat it as **borderline**, because a small change in temperature or humidity moves it to the next level.")
    if in_mode == "Projected value for my location":
        notes.append(f"Nearest PAGASA station: **{nearest_st}**. The value is a model projection for {target_year}, not an observation.")
    st.markdown("\n".join(f"- {n}" for n in notes))
    if idx < len(FOURH) - 1 and local_unbearable:
        st.markdown(f'<div class="fh-sds"><b>SDS coordination applies.</b> {SDS_NOTE} Record the observed local conditions that justify an early suspension.</div>', unsafe_allow_html=True)
    elif idx < len(FOURH) - 1:
        st.markdown(f'<div class="fh-sds">{SDS_NOTE}</div>', unsafe_allow_html=True)
    else:
        st.error("Hinto: suspend all academic activities, including face-to-face and home-based learning. Maintain communication with learners and families, conduct safety checks, and provide referrals for support.")

    # ---------- 3 · THE FOUR LEVELS ----------
    st.markdown("#### 3 · The four levels")

    def _zone_card(z, active):
        acts = "".join(f"<li><span>{ic}</span><span>{tx}</span></li>" for ic, tx in z["actions"])
        tag = f'<div class="fh-here" style="background:{z["color"]}">YOU ARE HERE</div>' if active else ""
        glow = f"box-shadow:0 0 0 1px {z['color']}66, 0 18px 40px {z['color']}22;" if active else ""
        return (f'<div class="fh-card{" on" if active else ""}" style="border-top-color:{z["color"]};{glow}">{tag}'
                f'<div class="nm" style="color:{z["color"]}">{z["label"].upper()}</div>'
                f'<div class="rg" style="color:{z["color"]}">({z["eie"]}) · {z["range"]}</div>'
                f'<div class="hd">{z["headline"]}</div><div class="ds">{z["detail"]}</div><ul>{acts}</ul></div>')

    st.markdown('<div class="fh-grid">' + "".join(_zone_card(z, z["key"] == cat["key"]) for z in FOURH) + '</div>', unsafe_allow_html=True)

    factors = "".join(f'<div class="fh-factor"><span>{ic}</span><b>{tx}</b></div>' for ic, tx in OTHER_FACTORS)
    st.markdown(
        '<div class="fh-other"><div class="fh-other-t">OTHER FACTORS TO CONSIDER</div>'
        f'<div class="fh-factors">{factors}</div></div>'
        f'<div class="fh-important"><b>⚠️ IMPORTANT:</b> {SDS_NOTE}</div>'
        f'<div class="fh-source">Source: {DEPED_SOURCE}</div>', unsafe_allow_html=True)

    # ---------- 4 · WHAT THE RECORD SHOWS ----------
    st.markdown(f"#### 4 · What the record shows — {nearest_st}")
    if IS_SYNTHETIC:
        st.warning("Running on synthetic placeholder data — the counts below are illustrative only.")
    ser_df = df_daily[["Year", nearest_st]].dropna().copy()
    ser_df["cat"] = pd.cut(ser_df[nearest_st], bins=[-np.inf, 27.0, 33.0, 42.0, np.inf],
                           labels=[z["label"] for z in FOURH], right=False)
    day_counts = ser_df.groupby(["Year", "cat"], observed=False).size().unstack(fill_value=0)
    fig_days = go.Figure()
    for z in FOURH:
        if z["label"] in day_counts.columns:
            fig_days.add_trace(go.Bar(x=day_counts.index.astype(str), y=day_counts[z["label"]],
                                      name=z["label"], marker_color=z["color"]))
    fig_days.update_layout(
        barmode="stack", title=f"Days per year in each 4H level — {nearest_st}",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#ECEDEF"),
        height=360, legend=dict(orientation="h", y=-0.15), margin=dict(t=50, b=30))
    fig_days.update_yaxes(gridcolor="rgba(255,255,255,0.08)", title="Days")
    st.plotly_chart(fig_days, width="stretch")

    n_all = max(len(ser_df), 1)
    n_hinga_up = int((ser_df[nearest_st] >= 33.0).sum())
    n_hinto = int((ser_df[nearest_st] >= HINTO_THRESHOLD).sum())
    yr_lo, yr_hi = int(ser_df["Year"].min()), int(ser_df["Year"].max())
    st.markdown(
        f"Over {yr_lo}–{yr_hi}, **{n_hinga_up / n_all * 100:.0f}%** of days at {nearest_st} reached **Hinga or Hinto** "
        f"(≥ 33°C, when face-to-face classes are suspended under the framework), and **{n_hinto / n_all * 100:.1f}%** reached **Hinto** (≥ 42°C). "
        "These are station-level values after data cleaning and imputation — a school's own conditions can differ.")

    # ---------- 5 · PROJECTION OUTLOOK ----------
    st.markdown(f"#### 5 · Projected peak-day level by year — {nearest_st}")
    counts = {z["key"]: 0 for z in FOURH}
    chips = ""
    for yr, v in zip(range(2026, 2041), projections_dict[nearest_st]):
        zc = classify_4h(float(v))
        counts[zc["key"]] += 1
        chips += (f'<div class="fh-chip" style="border-color:{zc["color"]};background:{zc["color"]}1A">'
                  f'<span>{yr}</span><b style="color:{zc["color"]}">{v:.1f}°</b><em>{zc["label"]}</em></div>')
    st.markdown(f'<div class="fh-chips">{chips}</div>', unsafe_allow_html=True)
    mk_p = float(selected_trend["MK_p"])
    caveat = ("The Mann-Kendall trend is **not statistically significant** (p ≥ 0.05), so treat these levels as indicative, not as a forecast of when suspensions will happen."
              if mk_p >= 0.05 else "The Mann-Kendall trend is statistically significant at the 5% level.")
    st.markdown(
        f"The linear projection puts the **hottest day of the year** in **Hinto in {counts['HINTO']} of 15 years** and in Hinga in {counts['HINGA']} "
        f"(2026–2040). Mann-Kendall for {nearest_st}: *{selected_trend['MK_trend']}* (p = {mk_p:.3f}). {caveat} "
        "Remember this is the single peak day, not the number of suspension days.")

    # ---------- 6 · ALL STATIONS ----------
    st.markdown(f"#### 6 · All stations — projected peak level in {target_year}")
    n_years_rec = max(int(df_daily["Year"].nunique()), 1)
    rows = []
    for s in STATIONS_PRESENT:
        ser = df_daily[s]
        zc = classify_4h(float(projections_dict[s][year_idx]))
        rows.append({
            "Station": s, "Region": STATION_COORDS[s]["region"],
            "Avg days/yr ≥33°C": round(float((ser >= 33.0).sum()) / n_years_rec, 1),
            "Avg days/yr ≥42°C": round(float((ser >= HINTO_THRESHOLD).sum()) / n_years_rec, 1),
            f"Projected peak HI {target_year} (°C)": round(float(projections_dict[s][year_idx]), 1),
            "4H level at projected peak": zc["label"],
        })
    df_4h = pd.DataFrame(rows).sort_values(f"Projected peak HI {target_year} (°C)", ascending=False).reset_index(drop=True)

    def _level_style(v):
        z = next((z for z in FOURH if z["label"] == v), None)
        return f"color:{z['color']};font-weight:700" if z else ""

    styled_4h = style_map(df_4h.style, _level_style, subset=["4H level at projected peak"])
    st.dataframe(styled_4h, width="stretch", height=430, hide_index=True)

    # ---------- 7 · DOLE ----------
    st.markdown("#### 7 · Labor sector (DOLE)")
    st.caption("DOLE cutoffs below are separate from the DepEd memo and have not been checked against the advisory text — verify them before citing.")
    st.markdown("<div class='policy-card'><h4>Department of Labor and Employment (DOLE)</h4><p>DOLE Department Advisory No. 3, s. 2023</p></div>", unsafe_allow_html=True)
    if hi_check >= DOLE_STOPPAGE_THRESHOLD:
        st.error(f"🚨 Work stoppage for heavy outdoor labor, 10AM–3PM (HI ≥ {DOLE_STOPPAGE_THRESHOLD:.0f}°C).")
    elif hi_check >= DOLE_SHADED_REST_THRESHOLD:
        st.warning(f"⚠️ Enforce 15-minute shaded rest cycles per working hour (HI ≥ {DOLE_SHADED_REST_THRESHOLD:.0f}°C).")
    else:
        st.info("ℹ️ Standard operations.")

with tab_pdf:
    st.subheader("Generate Formal A4 PDF Executive Advisory")
    col_pdf1, col_pdf2 = st.columns([1, 1.2])
    with col_pdf1:
        signatory_name = st.text_input("Local Chief Executive / Lead Researcher:", "Sid Ivan R. Piquero")
        custom_notes = st.text_area("Executive Directives:", "All Barangay Disaster Risk Reduction Management Committees (BDRRMC) are instructed to activate emergency cooling centers.")
        st.download_button(label="Download Projections (CSV)", data=df_impacts.to_csv(index=False).encode('utf-8'),
                            file_name=f"HEAT_SHIELD_Projections_{target_year}.csv", mime="text/csv")
        st.download_button(label="Download Trend Statistics (CSV)", data=df_trend_stats.to_csv(index=False).encode('utf-8'),
                            file_name="HEAT_SHIELD_Trend_Stats.csv", mime="text/csv")
        st.download_button(label="Download Validation Metrics (CSV)", data=df_validation.to_csv(index=False).encode('utf-8'),
                            file_name="HEAT_SHIELD_Validation.csv", mime="text/csv")
    with col_pdf2:
        pdf_html = f"""
        <!DOCTYPE html><html><head>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/html2pdf.js/0.10.1/html2pdf.bundle.min.js"></script>
        <style>
            body {{ font-family:'Times New Roman',serif; padding:10px; color:#111; background:#fff; }}
            #report {{ padding:30px; border:1px solid #ccc; background:#fff; width:100%; min-height:9in; margin:auto; box-sizing:border-box; }}
            .header {{ text-align:center; border-bottom:2px solid #000; padding-bottom:10px; margin-bottom:15px; }}
            .title {{ font-size:15px; font-weight:bold; margin:0; text-transform:uppercase; }}
            .section {{ margin-bottom:15px; }}
            .section-title {{ font-size:12px; font-weight:bold; background:#f0f0f0; padding:5px; border-left:4px solid #000; }}
            .cell {{ padding:5px; font-size:11px; border-bottom:1px solid #eee; }}
            .btn-pdf {{ background:#0071E3; color:#fff; border:none; padding:12px 20px; font-size:13px; font-weight:bold; border-radius:980px; cursor:pointer; margin-top:10px; }}
        </style></head><body>
        <div id="report">
            <div class="header">
                <div class="title">REPUBLIC OF THE PHILIPPINES — DEPARTMENT OF EDUCATION</div>
                <div class="subtitle">MOROPORO: LOCAL MUNICIPAL HEAT ADVISORY REPORT</div>
            </div>
            <div class="section"><div class="section-title">I. GEOGRAPHIC & METEOROLOGICAL METADATA</div>
                <table width="100%">
                    <tr><td class="cell"><b>Inspection Date:</b></td><td class="cell">{target_date.strftime('%B %d, %Y')}</td></tr>
                    <tr><td class="cell"><b>Target Location:</b></td><td class="cell">{search_query}</td></tr>
                    <tr><td class="cell"><b>Nearest PAGASA Center:</b></td><td class="cell">{nearest_st}</td></tr>
                    <tr><td class="cell"><b>Projected Daily Heat Index:</b></td><td class="cell"><b style="color:#d9534f;">{daily_target_hi} °C ({target_year})</b></td></tr>
                    <tr><td class="cell"><b>Mann-Kendall trend:</b></td><td class="cell">{selected_trend['MK_trend']} (Z={selected_trend['MK_Z']}, p={selected_trend['MK_p']})</td></tr>
                </table>
            </div>
            <div class="section"><div class="section-title">II. MANDATED SECTORAL PROTOCOLS</div>
                <p style="font-size:11px; line-height:1.4;">
                    1. <b>DepEd Memo No. 064, s. 2026 (4H Framework):</b> Hayo – Manageable (below 27°C) · Hinay – Caution (27–32°C) · Hinga – Extreme Caution (33–41°C) · Hinto – Danger and Extreme Danger (42°C and above).<br/>
                    &nbsp;&nbsp;&nbsp;<b>Level for this location:</b> {deped_cat['label'].upper()} – {deped_cat['eie']} ({deped_cat['range']}) — {deped_cat['headline']}<br/>&nbsp;&nbsp;&nbsp;<b>Note:</b> {SDS_NOTE}<br/>&nbsp;&nbsp;&nbsp;<b>Source:</b> {DEPED_SOURCE}.<br/>
                    2. <b>DOLE Advisory No. 3, s. 2023:</b> Shaded rest cycles at HI ≥ {DOLE_SHADED_REST_THRESHOLD}°C; full work stoppage at HI ≥ {DOLE_STOPPAGE_THRESHOLD}°C.
                </p>
            </div>
            <div class="section"><div class="section-title">III. LOCAL CHIEF EXECUTIVE DIRECTIVE</div>
                <p style="font-size:11px;"><i>"{custom_notes}"</i></p><br/><br/>
                <p style="font-size:11px;"><b>Validated By:</b><br/><br/>___________________________________<br/><b>{signatory_name}</b><br/>Lead Researcher / Project Proponent</p>
            </div>
        </div>
        <button class="btn-pdf" onclick="downloadPDF()">Save as Formatted A4 PDF</button>
        <script>
            function downloadPDF() {{
                var element = document.getElementById('report');
                var opt = {{ margin:0.3, filename:'Heat_Advisory_{target_year}.pdf', image:{{type:'jpeg',quality:0.98}},
                             html2canvas:{{scale:2}}, jsPDF:{{unit:'in',format:'a4',orientation:'portrait'}} }};
                html2pdf().set(opt).from(element).save();
            }}
        </script></body></html>
        """
        components.html(pdf_html, height=600)

st.markdown("---")
st.caption("MOROPORO | Rothfusz HI, k-NN spatiotemporal imputation, Mann-Kendall/Sen's slope, non-stationary GEV, held-out validation, cross-station altitude comparison.")