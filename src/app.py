import streamlit as st
from pathlib import Path
import joblib
import pandas as pd
import sys
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Live Energy Forecast",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add the root directory to Python's path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

# Our wrapper methods
from src.preprocess import preprocess_energy, feature_engineering, xgb_preprocess
from interval_prediction import get_conformal_prediction

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
ACCENT = "#0D9488"          # forecast / interval
ACTUAL = "#475569"          # observed data
ACCENT_FILL = "rgba(13, 148, 136, 0.16)"

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif;
    }

    /* Keep the header (it holds the sidebar open/close arrow) but make it blend in */
    header[data-testid="stHeader"] { background: transparent; }
    [data-testid="stAppDeployButton"] { display: none; }
    footer { visibility: hidden; }

    .block-container { padding-top: 3rem; padding-bottom: 2rem; max-width: 1400px; }

    /* Page header */
    .page-title { font-size: 1.75rem; font-weight: 700; letter-spacing: -0.02em; margin: 0; }
    .page-sub   { font-size: 0.95rem; opacity: 0.65; margin: 0.25rem 0 1.25rem 0; }

    /* Summary cards */
    .kpi {
        border: 1px solid rgba(128, 128, 128, 0.22);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        height: 100%;
        background: rgba(128, 128, 128, 0.04);
    }
    .kpi-label { font-size: 0.8rem; font-weight: 500; opacity: 0.65; margin-bottom: 0.35rem; }
    .kpi-value { font-size: 1.6rem; font-weight: 650; letter-spacing: -0.01em; line-height: 1.2; }
    .kpi-unit  { font-size: 0.9rem; font-weight: 500; opacity: 0.6; margin-left: 0.2rem; }
    .kpi-note  { font-size: 0.8rem; opacity: 0.6; margin-top: 0.3rem; }
    .kpi-value.accent { color: #0D9488; }
    .kpi-date  { font-size: 1.05rem; font-weight: 600; line-height: 1.35; }

    /* Section headings */
    .section-title { font-size: 1.05rem; font-weight: 600; margin: 0 0 0.15rem 0; }
    .section-sub   { font-size: 0.85rem; opacity: 0.6; margin: 0 0 0.75rem 0; }

    /* Sensor table */
    table.sensors { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
    table.sensors th {
        text-align: left; font-weight: 500; font-size: 0.8rem; opacity: 0.6;
        padding: 0.5rem 0.5rem; border-bottom: 1px solid rgba(128,128,128,0.25);
    }
    table.sensors td {
        padding: 0.55rem 0.5rem; border-bottom: 1px solid rgba(128,128,128,0.12);
        font-variant-numeric: tabular-nums;
    }
    table.sensors th.num, table.sensors td.num { text-align: right; }
    table.sensors tr:last-child td { border-bottom: none; }

    /* Forecast range bar */
    .range-bar {
        position: relative; height: 8px; border-radius: 4px;
        background: rgba(128,128,128,0.18); margin-top: 0.6rem;
    }
    .range-fill { position: absolute; top: 0; bottom: 0; border-radius: 4px; background: #0D9488; }

    /* Sidebar */
    section[data-testid="stSidebar"] { border-right: 1px solid rgba(128,128,128,0.2); }

    /* Status pill */
    .pill {
        display: inline-block; padding: 0.15rem 0.65rem; border-radius: 999px;
        font-size: 0.78rem; font-weight: 500;
    }
    .pill.live   { background: rgba(13,148,136,0.14); color: #0D9488; }
    .pill.paused { background: rgba(217,119,6,0.15);  color: #B45309; }
    .pill.done   { background: rgba(128,128,128,0.18); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Constants and session state
# ---------------------------------------------------------------------------
ROOMS = {
    1: "Kitchen", 2: "Living room", 3: "Laundry room",
    4: "Office", 5: "Bathroom", 6: "Outside",
    7: "Ironing room", 8: "Teenager room", 9: "Parents room",
}

DEFAULT_STATE = {
    "current_index": 0,
    "running": True,
    "finished": False,
    "actual_history": {},
    "prediction_history": {},
    "lower_history": {},
    "upper_history": {},
    "last_forecast": None,
}
for key, default in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default


def reset_simulation():
    '''Clear all simulation state and start again'''
    st.session_state.current_index = 0
    st.session_state.running = True
    st.session_state.finished = False
    st.session_state.actual_history = {}
    st.session_state.prediction_history = {}
    st.session_state.lower_history = {}
    st.session_state.upper_history = {}
    st.session_state.last_forecast = None


def toggle_running():
    st.session_state.running = not st.session_state.running


# ---------------------------------------------------------------------------
# Loaders (cached so widget changes don't recompute them)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    '''This is the model and encoder loader'''
    MODEL_PATH = root_dir / "models" / "xgb_best.pkl"
    ENCODER_PATH = root_dir / "models" / "month_encoder.pkl"
    return joblib.load(MODEL_PATH), joblib.load(ENCODER_PATH)


@st.cache_data
def load_data(name):
    '''This is the dataset loader'''
    return pd.read_csv(root_dir / "data" / "processed" / name)


@st.cache_resource
def prepare_calibration(_encoder):
    '''Preprocess the calibration set once for Conformal Prediction'''
    cal_raw = load_data("cal_set.csv")
    cal_df = preprocess_energy(cal_raw.copy())
    cal_df = feature_engineering(cal_df)
    X_cal = xgb_preprocess(cal_df, _encoder)
    y_cal = cal_df["Appliances"]
    ref_max = float(y_cal.quantile(0.99))
    return X_cal, y_cal, ref_max


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def format_date(value):
    '''Readable day and time strings'''
    ts = pd.to_datetime(value)
    return ts.strftime("%A, %d %B %Y"), ts.strftime("%I:%M %p")


def kpi_card(label, value, unit="", note="", accent=False):
    cls = "kpi-value accent" if accent else "kpi-value"
    unit_html = f'<span class="kpi-unit">{unit}</span>' if unit else ""
    note_html = f'<div class="kpi-note">{note}</div>' if note else ""
    return (
        f'<div class="kpi"><div class="kpi-label">{label}</div>'
        f'<div class="{cls}">{value}{unit_html}</div>{note_html}</div>'
    )


def render_sensor_table(row):
    rows = ""
    for n, name in ROOMS.items():
        rows += (
            f"<tr><td>{name}</td>"
            f"<td class='num'>{row[f'T{n}']:.1f} °C</td>"
            f"<td class='num'>{row[f'RH_{n}']:.1f} %</td></tr>"
        )
    return (
        "<table class='sensors'><thead><tr>"
        "<th>Room</th><th class='num'>Temperature</th><th class='num'>Humidity</th>"
        f"</tr></thead><tbody>{rows}</tbody></table>"
    )


def render_forecast_card(forecast, ref_max):
    low, high = forecast["low"], forecast["high"]
    mid = (low + high) / 2
    scale = max(ref_max, high, 1.0)
    left = max(0.0, low / scale * 100)
    width = max(2.0, (high - low) / scale * 100)
    return (
        f'<div class="kpi">'
        f'<div class="kpi-label">Expected range in {forecast["minutes"]} min (95% interval)</div>'
        f'<div class="kpi-value accent">{low:.0f} – {high:.0f}<span class="kpi-unit">Wh</span></div>'
        f'<div class="range-bar"><div class="range-fill" '
        f'style="left:{left:.1f}%; width:{width:.1f}%;"></div></div>'
        f'<div class="kpi-note">Midpoint {mid:.0f} Wh · width {high - low:.0f} Wh</div>'
        f'</div>'
    )


def build_chart(window_points=150):
    '''Plotly figure: actual vs predicted consumption with the conformal band'''
    def to_series(d, name):
        s = pd.Series(d, dtype="float64")
        s.index = pd.to_datetime(s.index)
        return s.rename(name)

    chart_df = pd.concat(
        [
            to_series(st.session_state.actual_history, "Actual"),
            to_series(st.session_state.prediction_history, "Predicted"),
            to_series(st.session_state.lower_history, "Lower"),
            to_series(st.session_state.upper_history, "Upper"),
        ],
        axis=1,
    ).sort_index().tail(window_points)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=chart_df.index, y=chart_df["Lower"], mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=chart_df.index, y=chart_df["Upper"], mode="lines",
        line=dict(width=0), fill="tonexty", fillcolor=ACCENT_FILL,
        name="95% interval", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=chart_df.index, y=chart_df["Actual"], mode="lines",
        name="Actual", line=dict(color=ACTUAL, width=2),
        hovertemplate="Actual: %{y:.1f} Wh<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=chart_df.index, y=chart_df["Predicted"], mode="lines",
        name="Predicted", line=dict(color=ACCENT, width=2, dash="dot"),
        hovertemplate="Predicted: %{y:.1f} Wh<extra></extra>",
    ))

    fig.update_layout(
        height=420,
        hovermode="x unified",
        uirevision="live-chart",          # keep zoom/pan state between updates
        transition=dict(duration=0),      # no animation between frames
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis_title="Energy (Wh)",
        xaxis_title=None,
    )
    fig.update_xaxes(showgrid=False, showline=True, linecolor="rgba(128,128,128,0.3)")
    fig.update_yaxes(
        showgrid=True, gridcolor="rgba(128,128,128,0.15)",
        zeroline=False, rangemode="tozero",
    )
    return fig


# ---------------------------------------------------------------------------
# Load data and calibrate
# ---------------------------------------------------------------------------
model, encoder = load_model()
test_set = load_data("test_set.csv")
cal_set = load_data("cal_set.csv")
X_cal, y_cal, REF_MAX = prepare_calibration(encoder)

# ---------------------------------------------------------------------------
# Sidebar: settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Simulation")
    speed = st.select_slider(
        "Playback speed",
        options=[5, 3, 2, 1, 0.5, 0.25],
        value=0.5,
        format_func=lambda s: f"{s}s per step",
        help="Real-time seconds between 10-minute readings. Lower is faster.",
    )
    horizon = st.select_slider(
        "Forecast horizon",
        options=[30, 60, 90, 120],
        value=30,
        format_func=lambda m: f"{m} min",
    )

    st.divider()
    st.button("Restart simulation", on_click=reset_simulation, use_container_width=True)
    st.caption(
        "Forecasts use an XGBoost model with split conformal prediction "
        "to give a 95% interval around each estimate."
    )

# ---------------------------------------------------------------------------
# Page header with always-visible playback control
# ---------------------------------------------------------------------------
head_left, head_right = st.columns([4, 1])
with head_left:
    st.markdown('<p class="page-title">Live energy forecast</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-sub">Appliance consumption, forecast with calibrated '
        'uncertainty from live home sensor data.</p>',
        unsafe_allow_html=True,
    )
with head_right:
    if st.session_state.finished:
        st.button("Restart", on_click=reset_simulation, use_container_width=True, type="primary")
    elif st.session_state.running:
        st.button("Pause", on_click=toggle_running, use_container_width=True)
    else:
        st.button("Resume", on_click=toggle_running, use_container_width=True, type="primary")

# ---------------------------------------------------------------------------
# One simulation step
# ---------------------------------------------------------------------------
def advance_one_step():
    '''Run the model for the next reading and store results in session state'''
    i = st.session_state.current_index

    available_periods = min(int(horizon / 30), len(test_set) - i - 1)
    if available_periods <= 0:
        st.session_state.current_index = len(test_set)
        st.session_state.finished = True
        st.session_state.running = False
        return True  # just finished

    # History up to and including the current reading
    history_raw = pd.concat([cal_set, test_set.iloc[:i + 1]])
    current_row = history_raw.iloc[-1]
    current_key = pd.to_datetime(current_row["date"]).isoformat()
    st.session_state.actual_history[current_key] = float(current_row["Appliances"])

    # Real-time preprocessing
    sensor_forecast = pd.concat(
        [history_raw, test_set.iloc[i + 1:i + 1 + available_periods]]
    )
    sensor_forecast = preprocess_energy(sensor_forecast)
    sensor_forecast = feature_engineering(sensor_forecast)
    sensor_forecast = xgb_preprocess(sensor_forecast, encoder)
    api_processed = sensor_forecast.tail(available_periods)

    # Conformal prediction
    y_pred, lower_bounds, upper_bounds = get_conformal_prediction(
        model, X_cal, y_cal, api_processed,
    )
    y_pred = pd.Series(y_pred).to_numpy().flatten()
    lower_bounds = pd.Series(lower_bounds).to_numpy().flatten()
    upper_bounds = pd.Series(upper_bounds).to_numpy().flatten()

    forecast_dates = pd.to_datetime(
        test_set.iloc[i + 1:i + 1 + available_periods]["date"]
    ).tolist()

    if not (len(y_pred) == len(lower_bounds) == len(upper_bounds) == len(forecast_dates)):
        raise ValueError(
            "The number of predictions, prediction bounds and forecast timestamps do not match."
        )

    for j, forecast_date in enumerate(forecast_dates):
        key = pd.Timestamp(forecast_date).isoformat()
        st.session_state.prediction_history[key] = float(y_pred[j])
        st.session_state.lower_history[key] = float(lower_bounds[j])
        st.session_state.upper_history[key] = float(upper_bounds[j])

    st.session_state.last_forecast = {
        "low": float(lower_bounds[-1]),
        "high": float(upper_bounds[-1]),
        "minutes": available_periods * 30,
    }
    st.session_state.current_index = i + 1
    return False


# ---------------------------------------------------------------------------
# Live view: a fragment re-runs on its own timer, so only these elements are
# updated (in place) instead of the whole page being redrawn.
# ---------------------------------------------------------------------------
run_every = speed if (st.session_state.running and not st.session_state.finished) else None


@st.fragment(run_every=run_every)
def live_view():
    just_finished = False
    if st.session_state.running and not st.session_state.finished:
        just_finished = advance_one_step()

    total = len(test_set)
    idx = min(st.session_state.current_index, total)

    # Status pill
    if st.session_state.finished:
        pill = '<span class="pill done">Simulation complete</span>'
    elif st.session_state.running:
        pill = '<span class="pill live">● Live</span>'
    else:
        pill = '<span class="pill paused">Paused</span>'
    st.markdown(f'<div style="text-align:right; margin-bottom:0.5rem">{pill}</div>', unsafe_allow_html=True)

    # Nothing to show yet
    if idx == 0:
        st.info("The dashboard will fill in as soon as the simulation starts.")
        return

    current_row = test_set.iloc[idx - 1]

    # Summary cards
    c1, c2, c3, c4 = st.columns(4)
    day_str, time_str = format_date(current_row["date"])
    c1.markdown(
        f'<div class="kpi"><div class="kpi-label">Current time</div>'
        f'<div class="kpi-date">{time_str}</div>'
        f'<div class="kpi-note">{day_str}</div></div>',
        unsafe_allow_html=True,
    )
    c2.markdown(kpi_card("Appliances now", f"{current_row['Appliances']:.0f}", "Wh"), unsafe_allow_html=True)
    c3.markdown(kpi_card("Lights now", f"{current_row['lights']:.0f}", "Wh"), unsafe_allow_html=True)
    if st.session_state.last_forecast:
        c4.markdown(render_forecast_card(st.session_state.last_forecast, REF_MAX), unsafe_allow_html=True)

    # Progress
    progress = min(idx / total, 1.0)
    st.progress(progress, text=f"{progress * 100:.0f}% complete ({idx:,} of {total:,} readings)")

    # Chart
    with st.container(border=True):
        st.markdown('<p class="section-title">Actual vs predicted consumption</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="section-sub">Most recent readings, with the 95% prediction interval shaded.</p>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            build_chart(),
            use_container_width=True,
            key="energy_chart",
            config={"displayModeBar": False},
        )

    # Sensors
    with st.container(border=True):
        st.markdown('<p class="section-title">Home sensors</p>', unsafe_allow_html=True)
        st.markdown('<p class="section-sub">Latest temperature and humidity by room.</p>', unsafe_allow_html=True)
        st.markdown(render_sensor_table(current_row), unsafe_allow_html=True)

    # Switch off the timer by triggering one full rerun when the run ends
    if just_finished:
        st.rerun()


live_view()