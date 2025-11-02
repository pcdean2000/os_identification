"""
config.py
儲存所有專案的常數、路徑、設定和超參數。
"""

from pathlib import Path

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
OS_MAPPING = {
    "Windows": 1,
    "Windows CE": 1,
    "Windows Phone": 1,
    "Mac OS": 2,
    "Mac OS X": 2,
    "Darwin": 2,
    "Linux": 3,
    "Ubuntu": 3,
    "Fedora": 3,
    "Android": 4,
    "iOS": 5,
}

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
OS_MAPPING_DISPLAY = {
    1: "Windows,Windows-CE,Windows-Phone",
    2: "Mac-OS,Mac-OS-X,Darwin",
    3: "Linux,Ubuntu,Fedora",
    4: "Android",
    5: "iOS",
}

# SHAP 分析中每個類別要繪製的樣本數
SHAP_TOP_N_SAMPLES = 10
