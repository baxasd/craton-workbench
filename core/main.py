import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import tempfile
import os
from core import logic

# =============================================================================
# 1. RESEARCH LOGIC
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
def get_radar_analysis(radar_bytes, r_lo, r_hi, v_sc, apply_mti, mti_weight, snr_th):
    sess = get_radar_session(radar_bytes)
    if not sess: return None, None, None, None
    spec, t, v = sess.build_spectrogram(float(r_lo), float(r_hi), 3, apply_mti, float(mti_weight))
    g = logic.analyze_gait_radar(spec, t, v, {"velocity_scale": float(v_sc), "snr_threshold": float(snr_th)})
    return spec, t, v, g

@st.cache_data(show_spinner="Generating Biomechanical Report...")
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

def create_kinematic_plot(df, x_col, y_cols, names, colors, title, show_env=False, show_trend=False):
    fig = go.Figure()
    # Dynamic window size from eval.py
    env_win = max(1, len(df)//20) if show_env else 1

    for y_col, name, color in zip(y_cols, names, colors):
        if y_col not in df.columns: continue
        x, y = df[x_col].values, df[y_col].values
        
        if show_env:
            m = df[y_col].rolling(env_win, min_periods=1).mean().values
            s = df[y_col].rolling(env_win, min_periods=1).std().fillna(0).values
            fig.add_trace(go.Scatter(x=x, y=m+s, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
            fig.add_trace(go.Scatter(x=x, y=m-s, mode='lines', line=dict(width=0), fillcolor=f"rgba{tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0,2,4)) + (0.2,)}", fill='tonexty', showlegend=False, hoverinfo='skip'))
            fig.add_trace(go.Scatter(x=x, y=m, mode='lines', name=name, line=dict(color=color, width=2.5)))
        else:
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=name, line=dict(color=color, width=2.5)))
        
        if show_trend:
            mask = ~np.isnan(y) & ~np.isnan(x)
            if mask.sum() > 1:
                slope, intercept = np.polyfit(x[mask], y[mask], 1)
                fig.add_trace(go.Scatter(x=x, y=slope*x+intercept, mode='lines', name=f"{name} Trend", line=dict(color=color, width=1.5, dash='dash'), hoverinfo='skip'))
    
    fig.update_layout(
        title=title, template="plotly_white", height=350, margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title=x_col.capitalize(), yaxis_title="Degrees (°)", hovermode="x unified"
    )
    return fig

# =============================================================================
# 2. MAIN INTERFACE
# =============================================================================

def main():
    st.set_page_config(page_title="Workbench", layout="wide")
    
    # Global Styles
    st.markdown("""
        <style>
            .block-container { padding-top: 1rem !important; }
            div[data-testid="stMetric"] {
                border: 1px solid #e9ecef;
                padding: 10px;
                border-radius: 5px;
            }
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)

    # Header Row
    c1, c2 = st.columns([0.8, 0.2], vertical_alignment="bottom")
    with c1:
        st.header("Workbench")
    with c2:
        if st.button("Reset Workspace", type="secondary", use_container_width=True):
            for k in ['raw_df', 'clean_df', 'radar_bytes', 'validation_report']:
                if k in st.session_state: del st.session_state[k]
            st.rerun()
    
    # Navigation
    nav_mode = st.segmented_control(
        "Navigation",
        options=["Data Quality", "Gait Analysis", "Radar Dynamics"],
        selection_mode="single",
        default="Data Quality",
        label_visibility="collapsed"
    )
    st.markdown("---")

    # Content Area
    if nav_mode == "Data Quality":
        if 'raw_df' not in st.session_state:
            g_file = st.file_uploader("Upload Gait Dataset for Quality Check", type=['csv', 'parquet'])
            if g_file:
                with st.spinner("Ingesting and Validating Dataset..."):
                    df = pd.read_csv(g_file) if g_file.name.endswith('.csv') else pd.read_parquet(g_file)
                    st.session_state.raw_df = df
                    st.session_state.validation_report, _ = logic.PipelineProcessor.validate(df)
                st.rerun()

        if 'raw_df' in st.session_state:
            # Inline Quality Controls
            with st.container(border=True):
                st.markdown("**DSP Pipeline Configuration**")
                c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
                with c1:
                    tel = st.checkbox("Teleport Removal", True)
                    tel_th = st.number_input("Jump Threshold", 0.1, 5.0, 0.5)
                with c2:
                    rep = st.checkbox("Interpolation", True)
                with c3:
                    sm = st.checkbox("Smoothing", True)
                    sm_w = st.number_input("Filter Window", 3, 101, 3)
                with c4:
                    if st.button("Apply Clean Pipeline", type="primary", use_container_width=True):
                        df = st.session_state.raw_df.copy()
                        if tel: df, _ = logic.PipelineProcessor.remove_teleportation(df, float(tel_th))
                        if rep: df = logic.PipelineProcessor.repair(df)
                        if sm: df = logic.PipelineProcessor.smooth(df, int(sm_w))
                        st.session_state.clean_df = df
                        st.rerun()
                    
                    if 'clean_df' in st.session_state:
                        csv_clean = st.session_state.clean_df.to_csv(index=False).encode('utf-8')
                        st.download_button("Export Filtered CSV", csv_clean, "cleaned_gait_data.csv", "text/csv", use_container_width=True)

            st.subheader("Data Integrity Report")
            st.code(st.session_state.get('validation_report', "No data loaded."))
            
            j_col = st.selectbox("Inspect Node", logic.identify_joint_columns(st.session_state.raw_df.columns))
            with st.container(border=True):
                fig_q = go.Figure()
                fig_q.add_trace(go.Scatter(y=st.session_state.raw_df[j_col], name='Raw', line=dict(color=logic.COLOR_RAW_DATA, width=1, dash='dot')))
                if 'clean_df' in st.session_state:
                    fig_q.add_trace(go.Scatter(y=st.session_state.clean_df[j_col], name='Cleaned', line=dict(color=logic.COLOR_CLEAN_DATA, width=2.5)))
                st.plotly_chart(fig_q.update_layout(title=f"Quality Check: {j_col}", height=450, template="plotly_white"), use_container_width=True)

    elif nav_mode == "Gait Analysis":
        if 'raw_df' not in st.session_state:
            gait_file = st.file_uploader("Upload Gait Dataset (CSV/Parquet)", type=['csv', 'parquet'])
            if gait_file:
                with st.spinner("Analyzing Movement Patterns..."):
                    df = pd.read_csv(gait_file) if gait_file.name.endswith('.csv') else pd.read_parquet(gait_file)
                    st.session_state.raw_df = df
                    st.session_state.validation_report, _ = logic.PipelineProcessor.validate(df)
                st.rerun()
        
        if 'raw_df' in st.session_state:
            # Data Processing (Spinner will appear here in main area)
            df = st.session_state.get('clean_df', st.session_state.raw_df)
            ts_df, df_sec, df_min, stats_df = process_analysis_data(df)

            # Inline Controls
            with st.container(border=True):
                st.markdown("**Visualization Settings**")
                c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
                with c1:
                    grp = st.selectbox("Temporal Grouping", ["Frames", "Seconds", "Minutes"], index=1)
                with c2:
                    env = st.checkbox("Show Envelopes", False)
                    trend = st.checkbox("Show Trends", True)
                with c3:
                    env_win = st.number_input("Smoothing Window", 1, 100, 5)
                with c4:
                    plot_df = {"Frames": ts_df.iloc[::max(1, len(ts_df)//1500)], "Seconds": df_sec, "Minutes": df_min}[grp]
                    csv_gait = plot_df.to_csv(index=False).encode('utf-8')
                    st.download_button(f"Export {grp} CSV", csv_gait, f"gait_metrics_{grp.lower()}.csv", "text/csv", use_container_width=True)

            # Summary Metrics
            if 'SPM' in stats_df.columns:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Trunk Lean (Avg)", f"{stats_df.loc['mean', 'lean_x']:.1f}°")
                m2.metric("Cadence (SPM)", f"{stats_df.loc['mean', 'SPM']:.1f}")
                m3.metric("Knee ROM (L)", f"{stats_df.loc['mean', 'l_knee_rom']:.1f}°")
                m4.metric("Knee ROM (R)", f"{stats_df.loc['mean', 'r_knee_rom']:.1f}°")

            st.subheader("Research Metrics Overview")
            st.dataframe(stats_df, use_container_width=True, height=250)
            
            x_col = {"Frames": "frame", "Seconds": "time_sec", "Minutes": "time_min"}[grp]
            with st.container(border=True):
                st.plotly_chart(create_kinematic_plot(plot_df, x_col, ['lean_x'], ["Trunk"], [logic.COLOR_RIGHT], "Trunk Lean Dynamics", env, trend), use_container_width=True)
            
            # 2x2 Grid for Limb Flexion
            plots_config = [
                ("Knee Flexion", ['l_knee', 'r_knee'], ["Left Knee", "Right Knee"]),
                ("Hip Flexion", ['l_hip', 'r_hip'], ["Left Hip", "Right Hip"]),
                ("Shoulder Swing", ['l_sho', 'r_sho'], ["Left Shoulder", "Right Shoulder"]),
                ("Elbow Flexion", ['l_elb', 'r_elb'], ["Left Elbow", "Right Elbow"])
            ]
            
            grid_cols = st.columns(2)
            for i, (title, y_cols, names) in enumerate(plots_config):
                with grid_cols[i % 2]:
                    with st.container(border=True):
                        st.plotly_chart(create_kinematic_plot(plot_df, x_col, y_cols, names, [logic.COLOR_LEFT, logic.COLOR_RIGHT], title, env, trend), use_container_width=True)

    elif nav_mode == "Radar Dynamics":
        if 'radar_bytes' not in st.session_state:
            radar_file = st.file_uploader("Upload Radar Session (Parquet)", type=['parquet'])
            if radar_file:
                with st.spinner("Loading Binary Radar Data..."):
                    st.session_state.radar_bytes = radar_file.getvalue()
                st.rerun()

        if 'radar_bytes' in st.session_state:
            # Inline Radar Controls
            with st.container(border=True):
                st.markdown("**Radar Processing Parameters**")
                c1, c2, c3 = st.columns(3)
                with c1:
                    r_lo = st.number_input("Min Range (m)", 0.0, 10.0, 0.0)
                    r_hi = st.number_input("Max Range (m)", 0.1, 20.0, 5.0)
                with c2:
                    v_sc = st.number_input("Velocity Scale", 0.001, 1.0, 0.1333, format="%.4f")
                    snr_th = st.number_input("SNR Quality Threshold (dB)", 5.0, 40.0, 15.0, 1.0)
                with c3:
                    apply_mti = st.checkbox("Apply MTI Filter (Clutter Removal)", True)
                    mti_weight = st.slider("MTI Strength", 0.0, 1.0, 0.8, 0.05, help="1.0 completely removes the median background. Lower values preserve more detail but leave more clutter.")

            spec, t, v, g = get_radar_analysis(st.session_state.radar_bytes, r_lo, r_hi, v_sc, apply_mti, mti_weight, snr_th)
            if spec is not None:
                st.subheader("Signal Quality")
                sq1, sq2 = st.columns(2)
                sq1.metric("Signal-to-Noise Ratio", f"{g.get('snr_db', 0.0):.1f} dB")
                sq2.metric("Quality Assessment", g.get('quality', 'Unknown'))

                with st.container(border=True):
                    fig_spec = go.Figure(go.Heatmap(
                        z=spec.T, x=t, y=v, 
                        colorscale='Jet', 
                        zmin=float(np.percentile(spec, 40)), 
                        zmax=float(np.percentile(spec, 99.5)),
                        showscale=False
                    ))
                    fig_spec.add_trace(go.Scatter(x=t, y=g.get('upper_env', []), mode='lines', line=dict(color='rgba(255, 255, 255, 0.8)', width=1.5, dash='dash'), name='Upper Env'))
                    fig_spec.add_trace(go.Scatter(x=t, y=g.get('lower_env', []), mode='lines', line=dict(color='rgba(255, 255, 255, 0.8)', width=1.5, dash='dash'), name='Lower Env'))
                    fig_spec.update_layout(title="Micro-Doppler Spectrogram", height=400, template="plotly_dark", margin=dict(l=0, r=0, t=50, b=0))
                    st.plotly_chart(fig_spec, use_container_width=True)

if __name__ == "__main__":
    main()
# Force reload
# Force reload
# Force reload 2
