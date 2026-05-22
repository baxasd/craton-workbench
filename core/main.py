import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import tempfile
import os
from core import logic

# =============================================================================
# 1. RESEARCH LOGIC (Restored Original Calculations)
# =============================================================================

@st.cache_resource
def get_radar_session(radar_bytes):
    if not radar_bytes: return None
    with tempfile.NamedTemporaryFile(delete=False, suffix='.parquet') as tmp:
        tmp.write(radar_bytes)
        tmp_path = tmp.name
    r_cfg = logic.RadarConfig(logic.RADAR_CFG_PATH)
    sess = logic.RecordingSession(tmp_path, r_cfg)
    os.remove(tmp_path)
    return sess

@st.cache_data(show_spinner="Performing DSP Analysis...")
def get_radar_analysis(radar_bytes, r_lo, r_hi, v_sc, hp, lp, gap, prom):
    sess = get_radar_session(radar_bytes)
    if not sess: return None, None, None, None, None
    spec, t, v, cent = sess.build_spectrogram(float(r_lo), float(r_hi), 3)
    g = logic.analyze_gait_radar(t, cent, {
        "velocity_scale": float(v_sc), "hp_cutoff": float(hp), "lp_cutoff": float(lp), 
        "step_freq_min_hz": 0.5, "step_freq_max_hz": 4.0, 
        "min_step_gap_s": float(gap), "prominence_factor": float(prom), "drift_thresh_factor": 1.0
    })
    return spec, t, v, cent, g

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
            if mask.sum() > 1:
                slope, _ = np.polyfit(x_mins[mask], df_sec[col][mask], 1)
                stats_df.loc[col, 'trend/min'] = slope
                
    for col in session_stats.columns:
        if col not in stats_df.index:
            stats_df.loc[col, 'mean'] = session_stats.loc['mean', col]
    return ts_df, df_sec, df_min, stats_df

def create_kinematic_plot(df, x_col, y_cols, names, colors, title, show_env=False, show_trend=False, env_win=5, y_title="Degrees (°)"):
    fig = go.Figure()
    for y_col, name, color in zip(y_cols, names, colors):
        if y_col not in df.columns: continue
        x, y = df[x_col].values, df[y_col].values
        if show_env:
            m = df[y_col].rolling(env_win, min_periods=1).mean().values
            s = df[y_col].rolling(env_win, min_periods=1).std().fillna(0).values
            fig.add_trace(go.Scatter(x=x, y=m+s, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
            fig.add_trace(go.Scatter(x=x, y=m-s, mode='lines', line=dict(width=0), fillcolor=f"rgba{tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0,2,4)) + (0.15,)}", fill='tonexty', showlegend=False, hoverinfo='skip'))
            fig.add_trace(go.Scatter(x=x, y=m, mode='lines', name=name, line=dict(color=color, width=2.5)))
        else:
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=name, line=dict(color=color, width=2.5)))
        
        if show_trend:
            mask = ~np.isnan(y) & ~np.isnan(x)
            if mask.sum() > 1:
                slope, intercept = np.polyfit(x[mask], y[mask], 1)
                fig.add_trace(go.Scatter(x=x, y=slope*x+intercept, mode='lines', name=f"{name} Trend", line=dict(color=color, width=1.5, dash='dash'), hoverinfo='skip'))
    
    fig.update_layout(
        template="plotly_white",
        height=350,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title=x_col.capitalize(),
        yaxis_title=y_title,
        hovermode="x unified"
    )
    return fig

# =============================================================================
# 2. MAIN INTERFACE
# =============================================================================

def main():
    st.set_page_config(page_title="Craton Studio Workbench", page_icon=logic.ICON_PATH, layout="wide")
    
    # State-linked Navigation
    tabs = ["Gait Analysis", "Radar Dynamics", "Data Quality"]
    query_params = st.query_params.to_dict()
    default_tab = query_params.get("tab", "Gait Analysis")
    if default_tab not in tabs: default_tab = "Gait Analysis"
    
    # Initialize session state for active tab if not present
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = default_tab


    # Sidebar: Context-Aware Control Center
    with st.sidebar:
        st.subheader("Control Center")
        
        # Display controls based on the current active tab
        nav_mode = st.session_state.active_tab

        if nav_mode == "Gait Analysis":
            grp = st.selectbox("Temporal Grouping", ["Frames", "Seconds", "Minutes"], index=1)
            env = st.checkbox("Show Envelopes", value=True)
            trend = st.checkbox("Show Trends", value=True)
            env_win = st.number_input("Smoothing Window", 1, 100, 5)
            
        elif nav_mode == "Radar Dynamics":
            r_lo = st.number_input("Min Range (m)", 0.0, 10.0, 0.0)
            r_hi = st.number_input("Max Range (m)", 0.1, 20.0, 5.0)
            v_sc = st.number_input("Velocity Scale", 0.001, 1.0, 0.1333, format="%.4f")
            hp = st.number_input("High-pass (Hz)", 0.1, 5.0, 0.5)
            lp = st.number_input("Low-pass (Hz)", 2.0, 10.0, 3.5)
            gap = st.number_input("Min Step Gap (s)", 0.05, 1.0, 0.28)
            prom = st.number_input("Peak Prominence", 0.05, 1.0, 0.45)
            
        elif nav_mode == "Data Quality":
            tel = st.checkbox("Teleport Removal", True)
            tel_th = st.number_input("Jump Threshold (m)", 0.1, 5.0, 0.5)
            rep = st.checkbox("Interpolation", True)
            sm = st.checkbox("Smoothing", True)
            sm_w = st.number_input("Filter Window", 3, 101, 3)
            
            if st.button("Apply Pipeline", type="primary", use_container_width=True):
                if 'raw_df' in st.session_state:
                    df = st.session_state.raw_df.copy()
                    if tel: df, _ = logic.PipelineProcessor.remove_teleportation(df, float(tel_th))
                    if rep: df = logic.PipelineProcessor.repair(df)
                    if sm: df = logic.PipelineProcessor.smooth(df, int(sm_w))
                    st.session_state.clean_df = df
                    st.rerun()

        st.divider()
        if st.button("🗑️ Reset Workspace", use_container_width=True):
            for k in ['raw_df', 'clean_df', 'radar_bytes', 'validation_report']:
                if k in st.session_state: del st.session_state[k]
            st.rerun()

    # Main Header
    st.title("Research Workbench")
    
    # Context-Aware Tab-like Navigation Bar
    # Since st.tabs executes all code blocks and causes loops when setting session state,
    # we use Segmented Control to drive the workbench logic.
    nav_mode = st.segmented_control(
        "Navigation",
        options=["Gait Analysis", "Radar Dynamics", "Data Quality"],
        selection_mode="single",
        default=st.session_state.active_tab,
        label_visibility="collapsed"
    )
    
    # Sync session state if user clicks the segmented control
    if nav_mode != st.session_state.active_tab:
        st.session_state.active_tab = nav_mode
        st.query_params["tab"] = nav_mode
        st.rerun()

    st.markdown("---")

    # Content Area
    if nav_mode == "Gait Analysis":
        if 'raw_df' not in st.session_state:
            gait_file = st.file_uploader("Upload Gait Dataset (CSV/Parquet)", type=['csv', 'parquet'])
            if gait_file:
                df = pd.read_csv(gait_file) if gait_file.name.endswith('.csv') else pd.read_parquet(gait_file)
                st.session_state.raw_df = df
                st.session_state.validation_report, _ = logic.PipelineProcessor.validate(df)
                st.rerun()
        
        if 'raw_df' in st.session_state:
            df = st.session_state.get('clean_df', st.session_state.raw_df)
            ts_df, df_sec, df_min, stats_df = process_analysis_data(df)
            
            # Summary Metrics
            if 'SPM' in stats_df.columns:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Cadence (SPM)", f"{stats_df.loc['mean', 'SPM']:.1f}")
                m2.metric("Trunk Lean (Avg)", f"{stats_df.loc['mean', 'lean_x']:.1f}°")
                m3.metric("Knee ROM (L)", f"{stats_df.loc['mean', 'l_knee_rom']:.1f}°")
                m4.metric("Knee ROM (R)", f"{stats_df.loc['mean', 'r_knee_rom']:.1f}°")

            st.subheader("Research Metrics Overview")
            st.dataframe(stats_df, use_container_width=True, height=250)
            
            plot_df = {"Frames": ts_df.iloc[::max(1, len(ts_df)//1500)], "Seconds": df_sec, "Minutes": df_min}[grp]
            x_col = {"Frames": "frame", "Seconds": "time_sec", "Minutes": "time_min"}[grp]
            
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(create_kinematic_plot(plot_df, x_col, ['l_knee', 'r_knee'], ["Left", "Right"], [logic.COLOR_LEFT, logic.COLOR_RIGHT], "Knee Flexion", env, trend, env_win), use_container_width=True)
                st.plotly_chart(create_kinematic_plot(plot_df, x_col, ['l_sho', 'r_sho'], ["Left", "Right"], [logic.COLOR_LEFT, logic.COLOR_RIGHT], "Shoulder Swing", env, trend, env_win), use_container_width=True)
            with c2:
                st.plotly_chart(create_kinematic_plot(plot_df, x_col, ['l_hip', 'r_hip'], ["Left", "Right"], [logic.COLOR_LEFT, logic.COLOR_RIGHT], "Hip Flexion", env, trend, env_win), use_container_width=True)
                st.plotly_chart(create_kinematic_plot(plot_df, x_col, ['lean_x'], ["Trunk"], [logic.COLOR_RIGHT], "Trunk Lean Dynamics", env, trend, env_win), use_container_width=True)
        else:
            st.warning("Gait dataset required for analysis. Upload a file above.")

    elif nav_mode == "Radar Dynamics":
        if 'radar_bytes' not in st.session_state:
            radar_file = st.file_uploader("Upload Radar Session (Parquet)", type=['parquet'])
            if radar_file:
                st.session_state.radar_bytes = radar_file.getvalue()
                st.rerun()

        if 'radar_bytes' in st.session_state:
            spec, t, v, cent, g = get_radar_analysis(st.session_state.radar_bytes, r_lo, r_hi, v_sc, hp, lp, gap, prom)
            if g:
                st.subheader("Gait Radar Metrics")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Steps", g['steps'])
                m2.metric("Cadence (SPM)", f"{g['t_spm']:.1f}")
                m3.metric("Asymmetry", f"{g['asy']:.1f}%")
                m4.metric("Drift (m)", f"{max(g['s_disp'])-min(g['s_disp']):.2f}")

                # Properly Visualized Spectrogram
                fig_spec = go.Figure(go.Heatmap(
                    z=spec.T, x=t, y=v, 
                    colorscale='Jet', 
                    zmin=float(np.percentile(spec, 40)), 
                    zmax=float(np.percentile(spec, 99.5)),
                    showscale=False
                ))
                fig_spec.add_trace(go.Scatter(x=t, y=cent, mode='lines', line=dict(color='white', width=1.5), name='Centroid'))
                fig_spec.update_layout(title="Micro-Doppler Spectrogram", height=400, template="plotly_dark", margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig_spec, use_container_width=True)
                
                st.plotly_chart(go.Figure(go.Scatter(x=g['time'], y=g['v_ac'], line=dict(color="#4A90D9", width=1.5))).update_layout(title="Oscillation Dynamics", height=250, margin=dict(l=0,r=0,t=40,b=0)), use_container_width=True)
        else:
            st.warning("Radar session required for analysis. Upload a file above.")

    elif nav_mode == "Data Quality":
        if 'raw_df' not in st.session_state:
            g_file = st.file_uploader("Upload Gait Dataset for Quality Check", type=['csv', 'parquet'])
            if g_file:
                df = pd.read_csv(g_file) if g_file.name.endswith('.csv') else pd.read_parquet(g_file)
                st.session_state.raw_df = df
                st.session_state.validation_report, _ = logic.PipelineProcessor.validate(df)
                st.rerun()

        if 'raw_df' in st.session_state:
            st.subheader("Data Integrity Report")
            st.code(st.session_state.get('validation_report', "No data loaded."))
            
            j_col = st.selectbox("Inspect Node", logic.identify_joint_columns(st.session_state.raw_df.columns))
            fig_q = go.Figure()
            fig_q.add_trace(go.Scatter(y=st.session_state.raw_df[j_col], name='Raw', line=dict(color=logic.COLOR_RAW_DATA, width=1, dash='dot')))
            if 'clean_df' in st.session_state:
                fig_q.add_trace(go.Scatter(y=st.session_state.clean_df[j_col], name='Cleaned', line=dict(color=logic.COLOR_CLEAN_DATA, width=2.5)))
            st.plotly_chart(fig_q.update_layout(title=f"Quality Check: {j_col}", height=450, template="plotly_white"), use_container_width=True)
        else:
            st.info("Upload a dataset to evaluate signal quality.")

if __name__ == "__main__":
    main()
