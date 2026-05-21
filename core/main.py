import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
import tempfile
import os
import pyarrow.parquet as pq
import scipy.ndimage as ndimage
from scipy.signal import find_peaks, butter, filtfilt, welch
from scipy.ndimage import uniform_filter1d

from core import logic

# =============================================================================
# 1. RADAR DSP (formerly radar_dsp.py)
# =============================================================================

class RecordingSession:
    def __init__(self, filepath: str, cfg: logic.RadarConfig):
        self.filepath = filepath
        self.cfg = cfg
        self.frames, self.timestamps = [], []
        self._load()

    def _load(self):
        df = pq.read_table(self.filepath).to_pandas()
        if self.cfg is None: return
        exp = self.cfg.numRangeBins * self.cfg.numLoops
        for raw_bytes, ts in zip(df['rdhm_bytes'], df['timestamp']):
            raw = np.frombuffer(raw_bytes, dtype=np.uint16)
            if raw.size == exp:
                self.frames.append(raw.astype(np.float32).reshape(self.cfg.numRangeBins, self.cfg.numLoops))
                self.timestamps.append(float(ts))

    @property
    def num_frames(self): return len(self.frames)
    @property
    def duration_s(self): return (self.timestamps[-1] - self.timestamps[0]) if len(self.timestamps) > 1 else 0.0

    def build_spectrogram(self, gate_lo_m, gate_hi_m, smooth_t=2):
        if not self.frames or self.cfg is None: return np.zeros((10,10)), np.zeros(10), np.zeros(10), np.zeros(10)
        cfg, nv = self.cfg, self.cfg.numLoops
        lo, hi = max(0, int(gate_lo_m / cfg.rangeRes)), min(cfg.numRangeBins, max(int(gate_lo_m/cfg.rangeRes)+1, int(gate_hi_m/cfg.rangeRes)))
        v_coarse = np.linspace(-cfg.dopMax, cfg.dopMax, nv, dtype=np.float32)
        spec_lin = np.abs(np.fft.fftshift(np.array(self.frames)[:, lo:hi, :].max(axis=1), axes=1))
        weights = np.maximum(spec_lin - np.percentile(spec_lin, 30, axis=1, keepdims=True), 0.0)
        w_sum = weights.sum(axis=1)
        centroid = np.where(w_sum > 1e-9, (weights * v_coarse[np.newaxis, :]).sum(axis=1) / w_sum, 0.0).astype(np.float32)
        spec_db = 20.0 * np.log10(spec_lin + 1e-9)
        c_idx = nv // 2
        clutter = np.percentile(np.delete(spec_db, [c_idx-1, c_idx, c_idx+1], axis=1), 99.0)
        spec_db[:, c_idx-1:c_idx+2] = np.clip(spec_db[:, c_idx-1:c_idx+2], a_min=None, a_max=clutter)
        if smooth_t > 1: spec_db = ndimage.uniform_filter1d(spec_db, size=smooth_t, axis=0)
        spec_db = ndimage.zoom(spec_db, (1, 8), order=3)
        return spec_db, np.array([t - self.timestamps[0] for t in self.timestamps], dtype=np.float32), np.linspace(-cfg.dopMax, cfg.dopMax, nv * 8, dtype=np.float32), centroid

# =============================================================================
# 2. GAIT INTERFACE (formerly gait.py)
# =============================================================================

@st.cache_data(show_spinner=False)
def process_analysis_data(df_raw):
    ts_df, session_stats = logic.generate_analysis_report(logic.df_to_session(df_raw))
    ts_df['time_sec'] = np.floor(ts_df['timestamp']).astype(int)
    numeric_cols = [c for c in ts_df.columns if c not in ['frame', 'time_sec', 'timestamp', 'time_min']]
    df_sec = ts_df.groupby('time_sec')[numeric_cols].mean().reset_index()
    ts_df['time_min'] = np.floor(ts_df['timestamp'] / 60.0).astype(int)
    df_min = ts_df.groupby('time_min')[numeric_cols].mean().reset_index()
    stats_df = df_sec.drop(columns=['time_sec'], errors='ignore').describe().T
    if len(df_sec) > 1:
        x_mins = df_sec['time_sec'] / 60.0
        for col in numeric_cols:
            mask = ~np.isnan(df_sec[col])
            if mask.sum() > 1: stats_df.loc[col, 'trend/min'], _ = np.polyfit(x_mins[mask], df_sec[col][mask], 1)
    for col in session_stats.columns:
        if col not in stats_df.index: stats_df.loc[col, 'mean'] = session_stats.loc['mean', col]
    return ts_df, df_sec, df_min, stats_df

def create_kinematic_plot(df, x_col, y_cols, names, colors, title, show_env=False, show_trend=False, env_win=5, y_title="Degrees (°)"):
    fig = go.Figure()
    for y_col, name, color in zip(y_cols, names, colors):
        if y_col not in df.columns: continue
        x, y = df[x_col].values, df[y_col].values
        if show_env:
            m, s = df[y_col].rolling(env_win, min_periods=1).mean().values, df[y_col].rolling(env_win, min_periods=1).std().fillna(0).values
            fig.add_trace(go.Scatter(x=x, y=m+s, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
            fig.add_trace(go.Scatter(x=x, y=m-s, mode='lines', line=dict(width=0), fillcolor=f"rgba{tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0,2,4)) + (0.2,)}", fill='tonexty', showlegend=False, hoverinfo='skip'))
            fig.add_trace(go.Scatter(x=x, y=m, mode='lines', name=name, line=dict(color=color, width=2.5)))
        else: fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=name, line=dict(color=color, width=2.5)))
        if show_trend:
            mask = ~np.isnan(y) & ~np.isnan(x)
            if mask.sum() > 1:
                slope, intercept = np.polyfit(x[mask], y[mask], 1)
                fig.add_trace(go.Scatter(x=x, y=slope*x+intercept, mode='lines', name=f"{name} Trend", line=dict(color=color, width=1.5, dash='dash'), hoverinfo='skip'))
    fig.update_layout(title=title, xaxis_title=x_col.capitalize(), yaxis_title=y_title, hovermode="x unified", margin=dict(l=0,r=0,t=40,b=0), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

def render_gait():
    if st.session_state.get('analysis_raw_df') is not None:
        st.write(''); st.button("← Back to Hub", type='tertiary', on_click=lambda: st.session_state.update(current_page="hub"))
        st.markdown("<h2 style='margin-top: -15px;'>Gait Analysis Dashboard</h2>", unsafe_allow_html=True)
        ts_df, df_sec, df_min, stats_df = process_analysis_data(st.session_state.analysis_raw_df)
        with st.sidebar:
            st.markdown("### Dashboard Controls")
            grp = st.radio("Temporal Grouping:", ["Frames (Raw)", "Seconds", "Minutes"], index=1)
            env, env_win, trend, rom = st.checkbox("Show Envelopes", value=True), st.number_input("Window:", 1, 100, 5), st.checkbox("Show Trends", value=True), st.checkbox("Show ROM", value=False)
            exp_df = ts_df if "Frames" in grp else (df_sec if "Seconds" in grp else df_min)
            st.download_button("Download Timeseries", exp_df.to_csv(index=False).encode('utf-8'), f"ts_{grp.lower()}.csv", "text/csv", use_container_width=True)
            if st.button("Clear Workspace", use_container_width=True): st.session_state.update(raw_df=None, clean_df=None, analysis_raw_df=None, current_page="hub"); st.rerun()
        x_col = "frame" if "Frames" in grp else ("time_sec" if "Seconds" in grp else "time_min")
        plot_df = ts_df.iloc[::max(1, len(ts_df)//1500)] if "Frames" in grp else (df_sec if "Seconds" in grp else df_min)
        st.dataframe(stats_df, use_container_width=True, height=250)
        st.plotly_chart(create_kinematic_plot(plot_df, x_col, ['lean_x'], ["Trunk Lean"], [logic.COLOR_RIGHT], "Trunk Lean Dynamics", env, trend, env_win), use_container_width=True)
        st.plotly_chart(create_kinematic_plot(plot_df, x_col, ['drift_x'], ["Position (X)"], [logic.COLOR_CENTER], "Treadmill Drift", env, trend, env_win, "Meters"), use_container_width=True)
        cols = st.columns(2)
        cfgs = [("Knee Flexion", ['l_knee', 'r_knee']), ("Hip Flexion", ['l_hip', 'r_hip']), ("Shoulder Swing", ['l_sho', 'r_sho']), ("Elbow Flexion", ['l_elb', 'r_elb'])]
        for i, (t, ys) in enumerate(cfgs):
            if rom: ys, t = [f"{y}_rom" for y in ys], f"{t} (ROM)"
            with cols[i%2]: st.plotly_chart(create_kinematic_plot(plot_df, x_col, ys, ["Left", "Right"], [logic.COLOR_LEFT, logic.COLOR_RIGHT], t, env, trend, env_win), use_container_width=True)
    else:
        st.write(''); st.button("← Back to Hub", type='tertiary', on_click=lambda: st.session_state.update(current_page="hub"))
        st.markdown("<h2 style='margin-top: -15px;'>Data Preparation</h2>", unsafe_allow_html=True)
        if st.session_state.get('raw_df') is None:
            _, c, _ = st.columns([1,2,1])
            with c:
                with st.container(border=True):
                    st.markdown("<h3 style='text-align: center;'>Import Dataset</h3>", unsafe_allow_html=True)
                    up = st.file_uploader("Upload", type=['parquet', 'csv'], label_visibility="collapsed")
                    if up:
                        st.session_state.raw_df = pd.read_parquet(up) if up.name.endswith('.parquet') else pd.read_csv(up)
                        st.session_state.validation_report, _ = logic.PipelineProcessor.validate(st.session_state.raw_df)
                        st.rerun()
                st.markdown("<h3 style='text-align: center;'>OR</h3>", unsafe_allow_html=True)
                with st.container(border=True):
                    st.markdown("<h3 style='text-align: center;'>Import Cleaned</h3>", unsafe_allow_html=True)
                    cl = st.file_uploader("Upload Clean", type=['csv'], label_visibility="collapsed")
                    if cl: st.session_state.analysis_raw_df = pd.read_csv(cl); st.rerun()
        else:
            with st.sidebar:
                st.markdown("### DSP Controls")
                j_col = st.selectbox("Node:", [c for c in st.session_state.raw_df.columns if c.startswith('j')])
                tel, tel_th, rep, rep_m, rep_l = st.checkbox("Teleport", True), st.number_input("Thresh:", 0.01, 10.0, 0.5), st.checkbox("Interpolate", True), st.selectbox("Method:", ["Linear", "Spline"]), st.number_input("Limit:", 1, 300, 30)
                sm, sm_w = st.checkbox("Smooth", True), st.number_input("Window:", 3, 101, 3)
                if st.button("Apply Pipeline", type="primary", use_container_width=True):
                    df = st.session_state.raw_df.copy()
                    if tel: df, _ = logic.PipelineProcessor.remove_teleportation(df, tel_th)
                    if rep: df = logic.PipelineProcessor.repair(df, rep_m.lower(), rep_l)
                    if sm: df = logic.PipelineProcessor.smooth(df, sm_w if sm_w%2!=0 else sm_w+1)
                    st.session_state.clean_df = df; st.rerun()
                if st.session_state.clean_df is not None:
                    st.download_button("Export Cleaned", st.session_state.clean_df.to_csv(index=False), "clean.csv", "text/csv", use_container_width=True)
                    if st.button("Analyze →", type="primary", use_container_width=True): st.session_state.analysis_raw_df = st.session_state.clean_df; st.rerun()
                if st.button("Clear", use_container_width=True): st.session_state.update(raw_df=None, clean_df=None, validation_report=""); st.rerun()
            if st.session_state.validation_report: st.code(st.session_state.validation_report)
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=st.session_state.raw_df[j_col], name='Raw', line=dict(color=logic.COLOR_RAW_DATA, width=1, dash='dot')))
            if st.session_state.clean_df is not None: fig.add_trace(go.Scatter(y=st.session_state.clean_df[j_col], name='Clean', line=dict(color=logic.COLOR_CLEAN_DATA, width=2.5)))
            fig.update_layout(title=f"Quality Check: {j_col}", height=500, margin=dict(l=0,r=0,t=40,b=0)); st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# 3. RADAR INTERFACE (formerly radar.py)
# =============================================================================
def analyze_gait_radar(time, velocity, cfg):
    v_raw = velocity * cfg["velocity_scale"]
    dt = float(np.mean(np.diff(time))) if len(time) > 1 else 0.1
    if np.isnan(dt) or dt <= 0.0: dt = 0.1
    fs = 1.0 / dt
    try:
        b, a = butter(4, min(cfg["lp_cutoff"]/(fs/2), 0.99), btype='low')
        v_disp = filtfilt(b, a, v_raw)
    except: v_disp = v_raw.copy()
    try:
        b, a = butter(2, max(cfg["hp_cutoff"]/(fs/2), 1e-4), btype='high')
        v_ac = filtfilt(b, a, v_raw)
        b, a = butter(4, min(cfg["lp_cutoff"]/(fs/2), 0.99), btype='low')
        v_ac = filtfilt(b, a, v_ac)
    except: v_ac = v_raw - np.mean(v_raw)
    try:
        f, pxx = welch(v_ac, fs, nperseg=min(len(v_ac), int(fs*8)))
        idx = (f >= cfg["step_freq_min_hz"]/2) & (f <= cfg["step_freq_max_hz"]/2)
        f_spm = f[idx][np.argmax(pxx[idx])] * 2 * 60 if np.any(idx) else 75.0
    except: f_spm = 75.0
    peaks, _ = find_peaks(v_ac, distance=max(1, int(fs*cfg["min_step_gap_s"])), prominence=max(float(np.std(v_ac))*cfg["prominence_factor"], 1e-5))
    t_spm = (len(peaks) / (float(time[-1]-time[0]) if len(time)>1 else 1.0) * 60)
    v_macro = v_disp - np.mean(v_disp)
    disp = np.cumsum(v_macro * dt)
    try:
        b, a = butter(2, max(0.01/(fs/2), 1e-4), btype='high')
        disp = filtfilt(b, a, disp)
    except: disp = disp - np.mean(disp)
    s_disp = uniform_filter1d(disp, size=max(1, int(fs*4.0)))
    d_lim = cfg["drift_thresh_factor"] * float(np.std(s_disp))
    s_a, s_b, cur = [], [], 0
    exp_g = 1.0 / (f_spm/120.0)
    for i, pk in enumerate(peaks):
        if i > 0:
            if round((time[pk] - time[peaks[i-1]]) / exp_g) % 2 == 1: cur ^= 1
        (s_a if cur == 0 else s_b).append(pk)
    a_asy = abs(np.mean(v_ac[s_a]) - np.mean(v_ac[s_b])) / ((abs(np.mean(v_ac[s_a])) + abs(np.mean(v_ac[s_b])))/2) * 100 if s_a and s_b else 0.0
    return {"time": time, "v_ac": v_ac, "s_disp": s_disp, "peaks": peaks, "s_a": s_a, "s_b": s_b, "d_lim": d_lim, "steps": len(peaks), "t_spm": t_spm, "f_spm": f_spm, "asy": a_asy}

def render_radar():
    st.write(''); st.button("← Back to Hub", type="tertiary", on_click=lambda: st.session_state.update(current_page="hub"))
    st.markdown("<h2 style='margin-top: -15px;'>Radar Analysis</h2>", unsafe_allow_html=True)
    if st.session_state.get('radar_bytes') is None:
        _, c, _ = st.columns([1, 2, 1])
        with c:
            with st.container(border=True):
                st.markdown("<h3 style='text-align: center;'>Import Radar Session</h3>", unsafe_allow_html=True)
                up = st.file_uploader("Upload", type=['parquet'], label_visibility="collapsed")
                if up: st.session_state.radar_bytes = up.getvalue(); st.rerun()
        return
    with st.sidebar:
        st.markdown("### Range & DSP")
        r_lo, r_hi = st.number_input("Min (m)", 0.0, 49.0, 0.0), st.number_input("Max (m)", 0.1, 50.0, 5.0)
        cmap, smooth = st.selectbox("Colormap:", ['Jet', 'Inferno', 'Plasma']), st.number_input("Smoothing:", 1, 10, 3)
        st.markdown("### Calibration")
        v_sc = st.number_input("Velocity Scale", 0.001, 1.0, 0.1333, format="%.4f")
        hp, lp = st.number_input("High-pass (Hz)", 0.1, 1.5, 0.5), st.number_input("Low-pass (Hz)", 2.0, 10.0, 3.5)
        st.markdown("### Step & Drift")
        f_min, f_max = st.number_input("Min Freq", 0.5, 2.5, 2.2), st.number_input("Max Freq", 2.0, 6.0, 3.5)
        gap, prom, drift = st.number_input("Min Gap", 0.15, 0.5, 0.28), st.number_input("Prominence", 0.05, 1.0, 0.45), st.number_input("Drift Thresh", 0.5, 5.0, 1.0)
        if st.button("Clear Workspace", use_container_width=True): st.session_state.radar_bytes = None; st.rerun()
    cfg = {"velocity_scale": v_sc, "hp_cutoff": hp, "lp_cutoff": lp, "step_freq_min_hz": f_min, "step_freq_max_hz": f_max, "min_step_gap_s": gap, "prominence_factor": prom, "drift_thresh_factor": drift}
    with st.spinner("Processing..."):
        try: r_cfg = logic.RadarConfig(logic.RADAR_CFG_PATH)
        except: r_cfg = None
        
        if r_cfg is None:
            st.error("Missing or invalid radar configuration. Please ensure 'radar_config.cfg' is present in 'assets/'.")
            return
            
        with tempfile.NamedTemporaryFile(delete=False, suffix='.parquet') as tmp:
            tmp.write(st.session_state.radar_bytes)
            tmp.flush()
            tmp_path = tmp.name
            
        sess = RecordingSession(tmp_path, r_cfg)
        spec, t, v, cent = sess.build_spectrogram(r_lo, r_hi, smooth)
        g = analyze_gait_radar(t, cent, cfg)
        os.remove(tmp_path)
    st.dataframe(pd.DataFrame([{"Steps": g['steps'], "SPM": f"{g['t_spm']:.1f}", "Asym": f"{g['asy']:.1f}%"}]), hide_index=True, use_container_width=True)
    fig = go.Figure(go.Heatmap(z=spec.T, x=t, y=v, colorscale=cmap, zmin=float(np.percentile(spec, 40)), zmax=float(np.percentile(spec, 99.5)), showscale=False))
    fig.add_trace(go.Scatter(x=t, y=cent, mode='lines', line=dict(color='white', width=1.5), name='Centroid'))
    fig.update_layout(title="Spectrogram", height=300, margin=dict(l=0,r=0,t=40,b=0)); st.plotly_chart(fig, use_container_width=True)
    f1 = go.Figure(go.Scatter(x=g['time'], y=g['v_ac'], line=dict(color="#4A90D9", width=1.2), name="Gait AC"))
    f1.add_trace(go.Scatter(x=g['time'][g['peaks']], y=g['v_ac'][g['peaks']], mode='markers', marker=dict(color="red", size=7), name="Steps"))
    st.plotly_chart(f1.update_layout(title="Oscillations", height=250, margin=dict(l=0,r=0,t=40,b=0)), use_container_width=True)
    f2 = go.Figure(go.Scatter(x=g['time'], y=g['s_disp'], line=dict(color="orange", width=3), name="Drift"))
    f2.add_hline(y=g['d_lim'], line_dash="dot", line_color="green"); f2.add_hline(y=-g['d_lim'], line_dash="dot", line_color="red")
    st.plotly_chart(f2.update_layout(title="Positional Drift", height=250, margin=dict(l=0,r=0,t=40,b=0)), use_container_width=True)

# =============================================================================
# 4. HUB & ROUTER (formerly hub.py and router.py)
# =============================================================================

def render_hub():
    st.markdown("""<style>[data-testid="stSidebar"] {display: none;}</style>""", unsafe_allow_html=True)
    _, c, _ = st.columns([1, 4, 1])
    with c:
        st.image(logic.LOGO_PATH, width=200); st.markdown("<p style='font-weight: bold; color: #666; font-size: 0.9rem; margin-top: -10px;'>The Core of Motion</p>", unsafe_allow_html=True)
        st.markdown("#### Welcome to Craton Studio\nA unified workspace for movement analysis.")
        st.markdown("#### Modules")
        m1, m2 = st.columns(2)
        with m1:
            with st.container(border=True):
                st.markdown("##### Gait Analysis"); st.caption("Posture metrics & export.")
                if st.button("Launch", key="g", type="primary", use_container_width=True): st.session_state.current_page = "gait"; st.rerun()
        with m2:
            with st.container(border=True):
                st.markdown("##### Radar Analysis"); st.caption("Micro-Doppler spectrograms.")
                if st.button("Launch", key="r", type="primary", use_container_width=True): st.session_state.current_page = "radar"; st.rerun()
        st.caption("University of Roehampton | Bakhtiyor Sohibnazarov, Jose Paredes & Lisa Haskel")

if __name__ == "__main__":
    st.set_page_config(page_title="Craton Studio", page_icon=logic.ICON_PATH, layout="wide", initial_sidebar_state="expanded")
    st.markdown("<style>.block-container { padding-top: 2rem !important; }</style>", unsafe_allow_html=True)
    for k, v in {"current_page": "hub", "raw_df": None, "clean_df": None, "analysis_raw_df": None, "radar_bytes": None}.items():
        if k not in st.session_state: st.session_state[k] = v
    with st.sidebar: st.image(logic.LOGO_PATH, width=180)
    if st.session_state.current_page == "hub": render_hub()
    elif st.session_state.current_page == "gait": render_gait()
    elif st.session_state.current_page == "radar": render_radar()
