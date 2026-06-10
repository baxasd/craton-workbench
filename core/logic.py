import os
import sys
import math
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from scipy.signal import find_peaks, butter, filtfilt, welch
from scipy.ndimage import uniform_filter1d

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
    # In one-directory mode with contents_directory='libs', 
    # the actual files are in the 'libs' folder.
    root_base = sys._MEIPASS
    libs_path = os.path.join(root_base, 'libs')
    if os.path.exists(libs_path):
        ROOT_DIR = libs_path
    else:
        ROOT_DIR = root_base
    BASE_DIR = os.path.dirname(sys.executable)
else:
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.abspath(os.path.join(_current_dir, '..'))
    BASE_DIR = ROOT_DIR

LOGO_PATH = os.path.join(ROOT_DIR, 'assets', 'logo.png')
ICON_PATH = os.path.join(ROOT_DIR, 'assets', 'icon.ico')
COMMAND_ICON = os.path.join(ROOT_DIR, 'assets', 'command.ico')
RADAR_CFG_PATH = os.path.join(ROOT_DIR, 'core', 'config.cfg')
APP_VERSION = "0.1.0"

# =============================================================================
# 2. DATA STRUCTURES (MediaPipe Topology mapped to Math Engine Names)
# =============================================================================

POSE_LANDMARKS = {
    0: "nose", 1: "left_eye_inner", 2: "left_eye", 3: "left_eye_outer",
    4: "right_eye_inner", 5: "right_eye", 6: "right_eye_outer",
    7: "left_ear", 8: "right_ear", 9: "mouth_left", 10: "mouth_right",
    11: "left_shoulder", 12: "right_shoulder", 13: "left_elbow", 14: "right_elbow",
    15: "left_wrist", 16: "right_wrist", 17: "left_pinky", 18: "right_pinky",
    19: "left_index", 20: "right_index", 21: "left_thumb", 22: "right_thumb",
    23: "left_hip", 24: "right_hip", 25: "left_knee", 26: "right_knee",
    27: "left_ankle", 28: "right_ankle", 29: "left_heel", 30: "right_heel",
    31: "left_foot_index", 32: "right_foot_index"
}

NAME_TO_ID = {v: k for k, v in POSE_LANDMARKS.items()}

def identify_joint_columns(columns: List[str]) -> List[str]:
    return [c for c in columns if c.endswith('_x') and (c.startswith('joint_') or (c.startswith('j') and c[1:2].isdigit()))]

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
        try:
            if 'joint_' in prefix: idx = int(prefix.split('_')[1])
            else: idx = int(prefix[1:])
        except: continue
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
            f.joints[idx] = Joint(name=real_name, metric=(
                float(row.get(f'{prefix}_x', 0.0)), 
                float(row.get(f'{prefix}_y', 0.0)), 
                float(row.get(f'{prefix}_z', 0.0))
            ))
        sess.frames.append(f)
    return sess

# =============================================================================
# 3. DSP FILTERS
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
# 4. KINEMATICS (Math Engine Restoration from motion.py)
# =============================================================================

def _get_vec(frame: Frame, name_or_id):
    idx = NAME_TO_ID.get(name_or_id) if isinstance(name_or_id, str) else name_or_id
    if idx is None or idx not in frame.joints: return None
    j = frame.joints[idx]
    return np.array([j.metric[0], j.metric[1], j.metric[2]])

def _get_trunk_midpoints(f: Frame):
    rh, lh = _get_vec(f, "right_hip"), _get_vec(f, "left_hip")
    rs, ls = _get_vec(f, "right_shoulder"), _get_vec(f, "left_shoulder")
    if any(v is None for v in [rh, lh, rs, ls]): return None, None
    return (rh + lh) / 2.0, (rs + ls) / 2.0

def calculate_joint_angle(f: Frame, p1: str, p2: str, p3: str) -> float:
    a, b, c = _get_vec(f, p1), _get_vec(f, p2), _get_vec(f, p3)
    if any(v is None for v in [a, b, c]): return 0.0
    ba, bc = a - b, c - b
    na, nc = np.linalg.norm(ba), np.linalg.norm(bc)
    if na == 0 or nc == 0: return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / (na * nc), -1.0, 1.0))))

def calculate_frontal_lean(f: Frame) -> float:
    """Forward/Backward Lean (X-Y Plane)"""
    mid_hip, mid_shoulder = _get_trunk_midpoints(f)
    if mid_hip is None: return 0.0
    dx = mid_shoulder[0] - mid_hip[0]
    dy = mid_shoulder[1] - mid_hip[1]
    return float(np.degrees(np.arctan2(dx, -dy)))

def calculate_sagittal_lean(f: Frame) -> float:
    """Side-to-Side Lean (Z-Y Plane) using the Right side"""
    rh, rs = _get_vec(f, "right_hip"), _get_vec(f, "right_shoulder")
    if rh is None or rs is None: return 0.0
    dz = rs[2] - rh[2]
    dy = rs[1] - rh[1]
    return float(np.degrees(np.arctan2(dz, -dy)))

def compute_all_metrics(f: Frame) -> dict:
    la, ra = _get_vec(f, "left_ankle"), _get_vec(f, "right_ankle")
    m_hip, _ = _get_trunk_midpoints(f)
    
    return {
        'lean_x': calculate_frontal_lean(f),
        'step_width': abs(la[0] - ra[0]) if la is not None and ra is not None else 0.0,
        'ankle_dist': float(np.linalg.norm(la - ra)) if la is not None and ra is not None else 0.0,
        
        'l_knee': calculate_joint_angle(f, "left_hip", "left_knee", "left_ankle"),
        'r_knee': calculate_joint_angle(f, "right_hip", "right_knee", "right_ankle"),
        
        'l_hip':  calculate_joint_angle(f, "left_shoulder", "left_hip", "left_knee"),
        'r_hip':  calculate_joint_angle(f, "right_shoulder", "right_hip", "right_knee"),
        
        'l_sho':  calculate_joint_angle(f, "left_hip", "left_shoulder", "left_elbow"),
        'r_sho':  calculate_joint_angle(f, "right_hip", "right_shoulder", "right_elbow"),
        
        'l_elb':  calculate_joint_angle(f, "left_shoulder", "left_elbow", "left_wrist"),
        'r_elb':  calculate_joint_angle(f, "right_shoulder", "right_elbow", "right_wrist"),
        
        'com_y': m_hip[1] if m_hip is not None else 0.0, 
        'drift_x': m_hip[0] if m_hip is not None else 0.0
    }

def generate_analysis_report(session):
    data = []
    for f in session.frames:
        m = compute_all_metrics(f)
        m.update({'timestamp': f.timestamp, 'frame': f.frame_id})
        data.append(m)
    df = pd.DataFrame(data)
    
    angle_cols = ['l_knee', 'r_knee', 'l_hip', 'r_hip', 'l_sho', 'r_sho', 'l_elb', 'r_elb']
    for col in angle_cols:
        if col in df.columns: 
            df[f'{col}_rom'] = df[col].rolling(30, center=True).max() - df[col].rolling(30, center=True).min()
            
    stats = df.drop(columns=['timestamp', 'frame']).describe()
    
    if 'ankle_dist' in df.columns and len(df) > 60:
        y = df['ankle_dist'].rolling(10, center=True).mean().fillna(0).values
        peaks, _ = find_peaks(y - np.mean(y), distance=10, prominence=0.05)
        duration_min = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]) / 60.0
        stats.loc['mean', 'SPM'] = (len(peaks) / duration_min) if duration_min > 0 and len(peaks) > 1 else 0.0
        
    return df, stats

# =============================================================================
# 5. RADAR PROCESSING
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

class RecordingSession:
    def __init__(self, filepath: str, cfg: RadarConfig):
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

    def build_spectrogram(self, gate_lo_m, gate_hi_m, smooth_t=2, apply_mti=True, mti_weight=0.8):
        from scipy.ndimage import uniform_filter1d, zoom
        if not self.frames or self.cfg is None: return np.zeros((10,10)), np.zeros(10), np.zeros(10)
        cfg, nv = self.cfg, self.cfg.numLoops
        lo, hi = max(0, int(gate_lo_m / cfg.rangeRes)), min(cfg.numRangeBins, max(int(gate_lo_m/cfg.rangeRes)+1, int(gate_hi_m/cfg.rangeRes)))
        
        raw_frames = np.array(self.frames)
        if apply_mti:
            # Proper Background Subtraction (MTI Filter) with adjustable weight
            bg = np.median(raw_frames, axis=0)
            clean_frames = np.maximum(raw_frames - (bg * mti_weight), 0)
        else:
            clean_frames = raw_frames

        spec_lin = np.abs(np.fft.fftshift(clean_frames[:, lo:hi, :].max(axis=1), axes=1))
        spec_db = 20.0 * np.log10(spec_lin + 1e-9)
        
        # Softly cap the exact center bin instead of aggressive deletion
        c_idx = nv // 2
        spec_db[:, c_idx] = np.clip(spec_db[:, c_idx], a_min=None, a_max=np.percentile(spec_db, 95))
        
        if smooth_t > 1: spec_db = uniform_filter1d(spec_db, size=smooth_t, axis=0)
        spec_db = zoom(spec_db, (1, 8), order=3)
        return spec_db, np.array([t - self.timestamps[0] for t in self.timestamps], dtype=np.float32), np.linspace(-cfg.dopMax, cfg.dopMax, nv * 8, dtype=np.float32)

def analyze_gait_radar(spec_db, time, velocities, cfg):
    """
    Analyzes gait from the micro-Doppler spectrogram.
    Extracts foundational signal quality metrics.
    """
    if spec_db is None or spec_db.size == 0: return {"snr_db": 0.0, "quality": "Poor"}
    
    # Calculate Signal-to-Noise Ratio (SNR)
    # spec_db is already in logarithmic scale (dB), so ratio is subtraction
    noise_floor = float(np.median(spec_db))
    signal_peak = float(np.max(spec_db))
    snr_db = max(0.0, signal_peak - noise_floor)
    
    return {
        "snr_db": snr_db,
        "quality": "Good" if snr_db > 15.0 else "Poor"
    }
# Force reload
