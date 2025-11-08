"""
config.py
儲存所有專案的常數、路徑、設定和超參數。
"""

from pathlib import Path
from typing import Dict

from xgboost import XGBClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier
)
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from catboost import CatBoostClassifier

# --- 基礎路徑 ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "Logs"
MODELS_DIR = BASE_DIR / "Models"
FIGURES_DIR = BASE_DIR / "Figures"
PERFORMANCES_DIR = BASE_DIR / "Performances"
CONTRIBUTIONS_DIR = BASE_DIR / "Contributions"

# --- 資料階段路徑 ---
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURED_DATA_DIR = DATA_DIR / "featured"
PREPARED_DATA_DIR = DATA_DIR / "prepared"

# --- OS 標籤對應 (Single Source of Truth) ---
# 將 OS_MAPPING_DISPLAY 移至頂部，作為唯一來源
OS_MAPPING_DISPLAY = {
    1: "Windows,Windows CE,Windows Phone",
    2: "Mac OS,Mac OS X,Darwin",
    3: "Linux,Ubuntu,Fedora",
    4: "Android",
    5: "iOS",
}


# 從 OS_MAPPING_DISPLAY 動態產生 OS_MAPPING
def _create_os_mapping(display_map: Dict[int, str]) -> Dict[str, int]:
    """從顯示用的 map 逆向建立 MAPPING 字典"""
    mapping = {}
    for int_label, string_list in display_map.items():
        os_names = string_list.split(",")
        for name in os_names:
            mapping[name.strip()] = int_label
    return mapping


# --- 1. 資料前處理 (Data Preprocessing) ---
PREPROCESS_COLUMNS = ["ts", "te", "sa", "da", "sp", "dp", "os", "win", "syn", "ttl"]

LASTO18_COLS = [
    "date_first_seen",
    "date_last_seen",
    "src_IP_addr",
    "dst_IP_addr",
    "src_pt",
    "dst_Pt",
    "HTTP_Host_OS",
    "TCP_window_size",
    "TCP_syn_size",
    "TCP_TTL",
]

LASTO20_COLS = [
    "Date flow start",
    "Date flow end",
    "Src IPv4",
    "Dst IPv4",
    "sPort",
    "dPort",
    "Ground Truth OS",
    "TCP win",
    "SYN size",
    "TCP SYN TTL",
]

LASTO23_COLS = [
    "start",
    "end",
    "SRC IP",
    "DST IP",
    "SRC port",
    "DST port",
    "UA OS family",
    "TCP Win Size",
    "TCP SYN Size",
    "TCP SYN TTL",
]

# --- 2. 特徵工程 (Feature Engineering) ---
# OS_MAPPING 現在從上面的 OS_MAPPING_DISPLAY 動態產生
OS_MAPPING = _create_os_mapping(OS_MAPPING_DISPLAY)

# 聚合後需要標準化的特徵
FEATURE_COLS_TO_SCALE = ["nsp", "csa", "stdtd", "stdp", "enp"]

# 埠類型劃分
PORT_BINS = [0, 1024, 49152, 65536]
PORT_BIN_LABELS = ["00", "01", "10"]

# --- 3. 模型訓練 (Training) ---
TRAIN_SIZE = 0.1
TEST_SIZE = 0.2
RANDOM_STATE = 42

# 要從訓練中移除的非特徵欄位
NON_FEATURE_COLS = ["ts", "te", "da", "sa", "td", "sp", "dp", "spwin"]

# 要從資料集中移除的標籤（如果樣本數過少）
MIN_LABEL_COUNT = 100

# --- 4. 分析 (Analysis) ---

# SHAP 分析中每個類別要繪製的樣本數
SHAP_TOP_N_SAMPLES = 10

# --- 5. 模型配置 (Model Configuration) ---

MODEL_CONFIGS = {
    "xgboost": {
        "class": XGBClassifier,
        "params": {"n_estimators": 100, "verbosity": 0, "n_jobs": -1},
        "tree_friendly": True
    },
    "randomforest": {
        "class": RandomForestClassifier,
        "params": {"n_estimators": 100, "verbose": 0, "n_jobs": -1},
        "tree_friendly": True
    },
    "extratrees": {
        "class": ExtraTreesClassifier,
        "params": {"n_estimators": 100, "verbose": 0, "n_jobs": -1},
        "tree_friendly": True
    },
    "decisiontree": {
        "class": DecisionTreeClassifier,
        "params": {},
        "tree_friendly": True
    },
    "gradientboosting": {
        "class": GradientBoostingClassifier,
        "params": {"n_estimators": 100, "verbose": 0},
        "tree_friendly": True
    },
    "histgradientboosting": {
        "class": HistGradientBoostingClassifier,
        "params": {"verbose": 0},
        "tree_friendly": True # 通常支援
    },
    "catboost": {
        "class": CatBoostClassifier,
        "params": {"n_estimators": 100, "verbose": 0, "allow_writing_files": False},
        "tree_friendly": True
    },
    "adaboost": {
        "class": AdaBoostClassifier,
        "params": {"n_estimators": 100},
        "tree_friendly": True # 假設基底為樹
    },
    "extratree": {
        "class": ExtraTreeClassifier,
        "params": {},
        "tree_friendly": True
    },
    # --- 以下為非樹模型或複雜組合 ---
    "bagging": {
        "class": BaggingClassifier,
        "params": {"n_estimators": 100, "n_jobs": -1},
        "tree_friendly": False 
    },
    "gaussiannb": {
        "class": GaussianNB,
        "params": {},
        "tree_friendly": False
    },
    "kneighbors": {
        "class": KNeighborsClassifier,
        "params": {"n_neighbors": 3, "n_jobs": -1},
        "tree_friendly": False
    },
    "svc": {
        "class": svm.SVC,
        "params": {"C": 1, "kernel": "rbf", "probability": True, "verbose": False},
        "tree_friendly": False
    },
}
