import os
import sys
import math
import logging
import struct
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from datetime import datetime
from scipy.signal import find_peaks

# =============================================================================
# 1. CONSTANTS & THEME
# =============================================================================

COLOR_LEFT = "#005FB8"
COLOR_RIGHT = "#D83B01"
COLOR_CENTER = "#8764B8" 
COLOR_RAW_DATA = "#D83B01"
COLOR_CLEAN_DATA = "#107C10"
PREP_RAW_WIDTH = 1.5
PREP_CLEAN_WIDTH = 2.5
COLOR_JOINT = "#323130"
COLOR_SKELETON_BG = "rgba(0,0,0,0.02)"
COLOR_REF_LINE = "rgba(0,0,0,0.15)"   
COLOR_RADAR_BG = "rgba(0,0,0,1)"       
COLOR_CENTROID_MAIN = "#00E5FF"              
COLOR_CENTROID_SHADOW = "black"             
COLOR_ZERO_LINE = "rgba(255, 255, 255, 0.4)" 

if getattr(sys, 'frozen', False):
    ROOT_DIR = sys._MEIPASS
    BASE_DIR = os.path.dirname(sys.executable)
else:
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.abspath(os.path.join(_current_dir, '..'))
    BASE_DIR = ROOT_DIR

LOGO_PATH = os.path.join(ROOT_DIR, 'assets', 'logo.png')
ICON_PATH = os.path.join(ROOT_DIR, 'assets', 'icon.ico')
COMMAND_ICON = os.path.join(ROOT_DIR, 'assets', 'command.ico')
RADAR_CFG_PATH = os.path.join(ROOT_DIR, 'core', 'config.cfg')
APP_VERSION = "v1.0.0"

# =============================================================================
# 2. DATA STRUCTURES (formerly types.py)
# =============================================================================

POSE_LANDMARKS = {
    0: "pelvis", 1: "spine_navel", 2: "spine_chest", 3: "neck",
    4: "clavicle_left", 5: "shoulder_left", 6: "elbow_left", 7: "wrist_left",
    8: "hand_left", 9: "handtip_left", 10: "thumb_left",
    11: "clavicle_right", 12: "shoulder_right", 13: "elbow_right", 14: "wrist_right",
    15: "hand_right", 16: "handtip_right", 17: "thumb_right",
    18: "hip_left", 19: "knee_left", 20: "ankle_left", 21: "foot_left",
    22: "hip_right", 23: "knee_right", 24: "ankle_right", 25: "foot_right",
    26: "head", 27: "nose", 28: "eye_left", 29: "ear_left",
    30: "eye_right", 31: "ear_right"
}

NAME_TO_ID = {v: k for k, v in POSE_LANDMARKS.items()}

def identify_joint_columns(columns: List[str]) -> List[str]:
    return [c for c in columns if c.endswith('_x') and (c.startswith('j') or c.startswith('joint'))]

@dataclass
class Joint:
    name: str = "unknown"
    metric: Tuple[float, float, float] = (0.0, 0.0, 0.0) 

@dataclass
class Frame:
    timestamp: float
    frame_id: int
    joints: Dict[int, Joint] = field(default_factory=dict) 

@dataclass
class Session:
    frames: List[Frame] = field(default_factory=list)
    @property
    def fps(self):
        if len(self.frames) < 2: return 30.0
        dur = self.duration
        return len(self.frames) / dur if dur > 0.001 else 30.0
    @property
    def duration(self):
        if not self.frames: return 0.0
        return self.frames[-1].timestamp - self.frames[0].timestamp

def df_to_session(df: pd.DataFrame) -> Session:
    sess = Session()
    if df.empty: return sess
    x_cols = identify_joint_columns(df.columns)
    start_time = None
    records = df.to_dict('records')
    parsed_columns = []
    for col in x_cols:
        prefix = col[:-2] 
        idx = int(prefix.split('_')[1]) if 'joint_' in prefix else int(prefix[1:])
        real_name = POSE_LANDMARKS.get(idx, str(idx))
        parsed_columns.append((prefix, idx, real_name))
    for i, row in enumerate(records):
        raw_ts = row.get('timestamp') or row.get('Timestamp') or row.get('time') or 0.0
        try:
            if isinstance(raw_ts, (int, float)):
                if start_time is None: start_time = float(raw_ts)
                ts = float(raw_ts) - start_time
            else:
                dt = pd.to_datetime(raw_ts)
                if start_time is None: start_time = dt
                ts = (dt - start_time).total_seconds()
        except: ts = float(i) * 0.033 
        f = Frame(timestamp=ts, frame_id=int(i))
        for prefix, idx, real_name in parsed_columns:
            f.joints[idx] = Joint(name=real_name, metric=(float(row.get(f'{prefix}_x', 0.0)), float(row.get(f'{prefix}_y', 0.0)), float(row.get(f'{prefix}_z', 0.0))))
        sess.frames.append(f)
    return sess

# =============================================================================
# 3. DSP FILTERS (formerly filter.py)
# =============================================================================

class PipelineProcessor:
    @staticmethod
    def _get_all_joint_cols(df: pd.DataFrame) -> list:
        x_cols = identify_joint_columns(df.columns)
        all_cols = []
        for c in x_cols:
            base = c[:-2]
            all_cols.extend([f"{base}_x", f"{base}_y", f"{base}_z"])
        return [c for c in all_cols if c in df.columns]

    @staticmethod
    def validate(df: pd.DataFrame):
        x_cols = identify_joint_columns(df.columns)
        if not x_cols: return "CRITICAL: No joint data found.", False
        existing_cols = PipelineProcessor._get_all_joint_cols(df)
        zeros = (df[existing_cols] == 0.0).sum().sum()
        nans = df[existing_cols].isna().sum().sum()
        report = []
        if zeros > 0: report.append(f"• Tracking Loss: {(zeros / df[existing_cols].size) * 100:.1f}% zeros.")
        if nans > 0: report.append(f"• Data Gaps: {nans} NaNs.")
        return ("DATA INTEGRITY: PASS", False) if not report else ("ISSUES:\n" + "\n".join(report), True)

    @staticmethod
    def remove_teleportation(df: pd.DataFrame, threshold=0.5):
        df_clean = df.copy()
        x_cols = identify_joint_columns(df_clean.columns)
        teleports = 0
        for c in x_cols:
            base = c[:-2]
            cols = [f"{base}_x", f"{base}_y", f"{base}_z"]
            if not all(k in df_clean.columns for k in cols): continue
            dists = np.sqrt((df_clean[cols].diff()**2).sum(axis=1))
            jump_idx = dists > threshold
            teleports += jump_idx.sum()
            df_clean.loc[jump_idx, cols] = np.nan
        return df_clean, teleports

    @staticmethod
    def repair(df: pd.DataFrame, method='linear', limit=30):
        df_clean = df.copy()
        valid_cols = PipelineProcessor._get_all_joint_cols(df_clean)
        df_clean[valid_cols] = df_clean[valid_cols].replace(0.0, np.nan).interpolate(method=method, limit=limit, limit_direction='both').fillna(0.0)
        return df_clean

    @staticmethod
    def smooth(df: pd.DataFrame, window=5):
        df_proc = df.copy()
        valid_cols = PipelineProcessor._get_all_joint_cols(df_proc)
        if valid_cols: df_proc[valid_cols] = df_proc[valid_cols].rolling(window=window, min_periods=1, center=True).mean()
        return df_proc

# =============================================================================
# 4. KINEMATICS (formerly motion.py)
# =============================================================================

def _get_vec(frame: Frame, name_or_id):
    idx = NAME_TO_ID.get(name_or_id) if isinstance(name_or_id, str) else name_or_id
    if idx is None or idx not in frame.joints: return None
    j = frame.joints[idx]
    return np.array([j.metric[0], j.metric[1], j.metric[2]])

def _get_trunk_midpoints(f: Frame):
    pts = [_get_vec(f, n) for n in ["hip_right", "hip_left", "shoulder_right", "shoulder_left"]]
    if any(v is None for v in pts): return None, None
    return (pts[0] + pts[1]) / 2.0, (pts[2] + pts[3]) / 2.0

def calculate_joint_angle(f: Frame, p1: str, p2: str, p3: str) -> float:
    a, b, c = _get_vec(f, p1), _get_vec(f, p2), _get_vec(f, p3)
    if any(v is None for v in [a, b, c]): return 0.0
    ba, bc = a - b, c - b
    na, nc = np.linalg.norm(ba), np.linalg.norm(bc)
    if na < 1e-4 or nc < 1e-4: return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / (na * nc), -1.0, 1.0))))

def calculate_trunk_lean(f: Frame) -> tuple[float, float]:
    m_hip, m_sho = _get_trunk_midpoints(f)
    leans = []
    for h, s in [(m_hip, m_sho), (_get_vec(f, "hip_left"), _get_vec(f, "shoulder_left")), (_get_vec(f, "hip_right"), _get_vec(f, "shoulder_right"))]:
        if h is not None and s is not None: leans.append(np.degrees(np.arctan2(s[0]-h[0], -(s[1]-h[1]))))
    lx = float(np.mean(leans)) if leans else 0.0
    lz = np.degrees(np.arctan2(m_sho[2]-m_hip[2], -(m_sho[1]-m_hip[1]))) if m_hip is not None and m_sho is not None else 0.0
    return lx, float(lz)

def compute_all_metrics(f: Frame) -> dict:
    lx, _ = calculate_trunk_lean(f)
    la, ra = _get_vec(f, "ankle_left"), _get_vec(f, "ankle_right")
    m_hip, _ = _get_trunk_midpoints(f)
    return {
        'lean_x': lx, 'ankle_dist': float(np.linalg.norm(la - ra)) if la is not None and ra is not None else 0.0,
        'l_knee': calculate_joint_angle(f, "hip_left", "knee_left", "ankle_left"),
        'r_knee': calculate_joint_angle(f, "hip_right", "knee_right", "ankle_right"),
        'l_hip':  calculate_joint_angle(f, "pelvis", "hip_left", "knee_left"),
        'r_hip':  calculate_joint_angle(f, "pelvis", "hip_right", "knee_right"),
        'l_sho':  calculate_joint_angle(f, "spine_chest", "shoulder_left", "elbow_left"),
        'r_sho':  calculate_joint_angle(f, "spine_chest", "shoulder_right", "elbow_right"),
        'l_elb':  calculate_joint_angle(f, "shoulder_left", "elbow_left", "wrist_left"),
        'r_elb':  calculate_joint_angle(f, "shoulder_right", "elbow_right", "wrist_right"),
        'com_y': m_hip[1] if m_hip is not None else 0.0, 'drift_x': m_hip[0] if m_hip is not None else 0.0
    }

def generate_analysis_report(session):
    data = []
    for f in session.frames:
        m = compute_all_metrics(f)
        m.update({'timestamp': f.timestamp, 'frame': f.frame_id})
        data.append(m)
    df = pd.DataFrame(data)
    for col in ['l_knee', 'r_knee', 'l_hip', 'r_hip', 'l_sho', 'r_sho', 'l_elb', 'r_elb']:
        if col in df.columns: df[f'{col}_rom'] = df[col].rolling(30, center=True).max() - df[col].rolling(30, center=True).min()
    stats = df.drop(columns=['timestamp', 'frame']).describe()
    if 'ankle_dist' in df.columns and len(df) > 60:
        y = df['ankle_dist'].rolling(10, center=True).mean().fillna(0).values
        peaks, _ = find_peaks(y - np.mean(y), distance=10)
        stats.loc['mean', 'SPM'] = (len(peaks) / ((df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]) / 60.0)) if len(peaks) > 1 else 0.0
    return df, stats

# =============================================================================
# 5. RADAR CONFIG (formerly radar_parse.py)
# =============================================================================

class RadarConfig:
    def __init__(self, file_path: str):
        self.file_path = file_path
        with open(file_path) as f:
            lines = [l.split() for l in f if l.strip() and not l.startswith("%")]
        c, fr, rx, tx = {}, {}, 0, 0
        for v in lines:
            if not v: continue
            if v[0] == "channelCfg": rx, tx = int(v[1]), int(v[2])
            elif v[0] == "profileCfg" and int(v[1]) == 0:
                c = {"startFreq": float(v[2]), "idleTime": float(v[3]), "rampEndTime": float(v[5]), "freqSlope": float(v[8]), "numADCsamples": int(v[10]), "sampleRate": float(v[11])}
            elif v[0] == "frameCfg":
                fr = {"chirpStartInd": int(v[1]), "chirpEndInd": int(v[2]), "numLoops": int(v[3]), "periodicity": int(v[5])}
        if not c or not fr: raise ValueError("Invalid radar config")
        self.rxAntennas, self.txAntennas, self.ADCsamples = bin(rx).count("1"), bin(tx).count("1"), c["numADCsamples"]
        self.numRangeBins = 2 ** math.ceil(math.log2(self.ADCsamples)) if self.ADCsamples > 0 else 1
        self.BW = c["freqSlope"] * self.ADCsamples / c["sampleRate"] * 1e9
        self.rangeRes, self.rangeMax = 3e8 / (2 * self.BW), (3e8 / (2 * self.BW)) * self.numRangeBins
        self.numLoops, Tc, fc = fr["numLoops"], (c["idleTime"] + c["rampEndTime"]) * 1e-6, c["startFreq"] * 1e9
        nc = (fr["chirpEndInd"] - fr["chirpStartInd"] + 1) * self.numLoops
        self.dopRes = 3e8 / (2 * fc * Tc * nc)
        self.dopMax, self.T = nc * self.dopRes / 2, fr["periodicity"]
        self.frameRate = 1e3 / self.T
