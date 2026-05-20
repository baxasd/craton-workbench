import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from datetime import datetime

# ================================================
# KINECT AZURE CONSTANTS (32 Joints)
# ================================================
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

# ── 1. Skeleton Configuration (From render.py) ──
VISIBLE_NAMES = [
    "pelvis", "spine_navel", "spine_chest", "neck", "head",
    "shoulder_left", "shoulder_right",
    "elbow_left", "elbow_right", 
    "wrist_left", "wrist_right",
    "hip_left", "hip_right", 
    "knee_left", "knee_right", 
    "ankle_left", "ankle_right"
]

BONES_LIST = [
    # Spine
    ("pelvis", "spine_navel"),
    ("spine_navel", "spine_chest"),
    ("spine_chest", "neck"),
    ("neck", "head"),
    
    # Left Arm
    ("spine_chest", "clavicle_left"),
    ("clavicle_left", "shoulder_left"),
    ("shoulder_left", "elbow_left"),
    ("elbow_left", "wrist_left"),
    
    # Right Arm
    ("spine_chest", "clavicle_right"),
    ("clavicle_right", "shoulder_right"),
    ("shoulder_right", "elbow_right"),
    ("elbow_right", "wrist_right"),
    
    # Left Leg
    ("pelvis", "hip_left"),
    ("hip_left", "knee_left"),
    ("knee_left", "ankle_left"),
    ("ankle_left", "foot_left"),
    
    # Right Leg
    ("pelvis", "hip_right"),
    ("hip_right", "knee_right"),
    ("knee_right", "ankle_right"),
    ("ankle_right", "foot_right")
]

NAME_TO_ID = {v: k for k, v in POSE_LANDMARKS.items()}

def identify_joint_columns(columns: List[str]) -> List[str]:
    return [c for c in columns if c.endswith('_x') and (c.startswith('j') or c.startswith('joint'))]

# ================================================
# Core Data Structures
# ================================================

@dataclass
class Joint:
    name: str = "unknown"
    pixel: Tuple[int, int] = (0, 0)       
    metric: Tuple[float, float, float] = (0.0, 0.0, 0.0) 
    visibility: float = 0.0

@dataclass
class Frame:
    timestamp: float
    frame_id: int
    joints: Dict[int, Joint] = field(default_factory=dict) 

@dataclass
class Session:
    subject_id: str = "Anonymous"
    date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    frames: List[Frame] = field(default_factory=list)

    @property
    def fps(self):
        if len(self.frames) < 2: return 30.0
        dur = self.duration
        # Prevent divide-by-zero if timestamps are corrupted
        return len(self.frames) / dur if dur > 0.001 else 30.0

    @property
    def duration(self):
        if not self.frames: return 0.0
        return self.frames[-1].timestamp - self.frames[0].timestamp

# ================================================
# Converters
# ================================================

def df_to_session(df: pd.DataFrame) -> Session:
    # Notice we removed pd.Timestamp.now(). The dataclass default_factory handles the date automatically!
    sess = Session(subject_id="Processed")
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
        # Safely grab the timestamp regardless of how Pandas formatted the column name
        raw_ts = row.get('timestamp') or row.get('Timestamp') or row.get('time') or 0.0
        ts = 0.0
        
        try:
            # FIX: If it is already a raw float (Unix epoch), do math normally
            if isinstance(raw_ts, (int, float)):
                if start_time is None: start_time = float(raw_ts)
                ts = float(raw_ts) - start_time
            # FIX: If it is a string OR a pd.Timestamp object, let Pandas safely extract the seconds
            else:
                dt = pd.to_datetime(raw_ts)
                if start_time is None: start_time = dt
                ts = (dt - start_time).total_seconds()
                
        except Exception as e:
            # We only hit this fallback if the data is genuinely missing/corrupted
            ts = float(i) * 0.033 

        f = Frame(timestamp=ts, frame_id=int(i))
        
        for prefix, idx, real_name in parsed_columns:
            mx = float(row.get(f'{prefix}_x', 0.0))
            my = float(row.get(f'{prefix}_y', 0.0))
            mz = float(row.get(f'{prefix}_z', 0.0))
            
            f.joints[idx] = Joint(name=real_name, metric=(mx, my, mz))
            
        sess.frames.append(f)
        
    return sess