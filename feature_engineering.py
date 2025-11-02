"""
feature_engineering.py
負責第二階段：從 processed 資料中抽取特徵。
應用策略模式 (Strategy Pattern) 來進行特徵彙總 (Aggregation)。
"""
import logging
import warnings
from pathlib import Path
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk
from scipy.stats.mstats import winsorize
from sklearn import preprocessing

import config
import utils

warnings.filterwarnings("ignore")
utils.setup_logging(__file__)

# --- 特徵彙總的策略模式 (Strategy Pattern) ---

class AggregationStrategy(ABC):
    """特徵彙總策略的抽象基礎類別"""
    @abstractmethod
    def aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        對 DataFrame 進行特徵彙總並返回更新後的 DataFrame。
        """
        pass

class CoreNetworkFeatures(AggregationStrategy):
    """計算核心網路特徵：csa, nsp"""
    def aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.debug("應用 CoreNetworkFeatures 策略")
        df["csa"] = df.groupby("sa")["da"].transform(lambda x: x.nunique())
        df["nsp"] = df.groupby("sa")["sp"].transform(lambda x: x.nunique())
        return df

class PortStatisticsFeatures(AggregationStrategy):
    """計算埠統計特徵：spwin, maxp, minp, prg, stdp"""
    def aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.debug("應用 PortStatisticsFeatures 策略")
        df["spwin"] = df.groupby("sa")["sp"].transform(
            lambda x: winsorize(x, limits=[0.1, 0.1])
        )
        df["maxp"] = df.groupby("sa")["spwin"].transform("max")
        df["minp"] = df.groupby("sa")["spwin"].transform("min")
        df["prg"] = df["maxp"] - df["minp"]
        df["stdp"] = df.groupby("sa")["sp"].transform(lambda x: x.std(ddof=0))
        return df

class DurationStatisticsFeatures(AggregationStrategy):
    """計算持續時間統計特徵：stdtd"""
    def aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.debug("應用 DurationStatisticsFeatures 策略")
        df["stdtd"] = df.groupby("sa")["td"].transform(lambda x: x.std(ddof=0))
        return df

class EntropyFeatures(AggregationStrategy):
    """計算埠熵特徵：enp"""
    def aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.debug("應用 EntropyFeatures 策略")
        pf = df.groupby(["sa", "sp"]).size().reset_index(name="fsp")
        flows = pf.groupby("sa")["fsp"].sum()
        
        # 避免除以零
        if flows.empty:
            logging.warning("EntropyFeatures: 'flows' 為空，無法計算 enp")
            df["enp"] = np.nan
            return df
            
        pf["pb"] = pf.apply(lambda x: x["fsp"] / flows[x["sa"]] if x["sa"] in flows else 0, axis=1)
        
        # 避免 log(0)
        pf['pb_log'] = np.log2(pf['pb'].replace(0, 1)) 
        entropy = -pf.groupby("sa").apply(lambda x: np.sum(x["pb"] * x["pb_log"]))
        
        df = pd.merge(df, entropy.rename("enp"), on="sa", how="left")
        return df

class PortTypeFeatures(AggregationStrategy):
    """計算埠類型特徵：typ (拆分為 typ1, typ2)"""
    def aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.debug("應用 PortTypeFeatures 策略")
        temp = pd.cut(
            df["sp"],
            bins=config.PORT_BINS,
            labels=config.PORT_BIN_LABELS,
            right=False,
        ).reset_index(name="typ")
        df = df.merge(temp["typ"], left_index=True, right_index=True, how="left")
        df["typ"] = df["typ"].astype(str)
        
        # 拆分 typ 為 typ1 和 typ2
        temp_typ = df["typ"].str.split("", expand=True).iloc[:, 1:-1]
        if temp_typ.shape[1] == 2:
            temp_typ.columns = ["typ1", "typ2"]
            df[["typ1", "typ2"]] = temp_typ.astype(int)
        else:
            logging.warning("PortTypeFeatures: 'typ' 欄位拆分失敗")
            df["typ1"] = np.nan
            df["typ2"] = np.nan
            
        df = df.drop(columns=["typ"], errors="ignore")
        return df

# --- 特徵抽取主類別 ---

class FeatureExtraction:
    def __init__(self, dataset_path: Path, freq: int = 24):
        if not (1 <= freq):
            raise ValueError("Frequency 必須大於等於 1 小時")
            
        logging.info(f"初始化 FeatureExtraction: {dataset_path.stem}, Freq={freq}h")
        self.freq = freq
        self.input_dir = dataset_path.parent
        self.output_dir = config.FEATURED_DATA_DIR
        self.dataset_name = dataset_path.stem
        
        try:
            self.dataset = load_from_disk(dataset_path)
            self.df = self.dataset.to_pandas()
        except FileNotFoundError:
            logging.error(f"找不到已處理的資料集: {dataset_path}")
            raise
            
        self.scaler = preprocessing.StandardScaler()
        self.os_mapping = config.OS_MAPPING
        
        # 註冊所有彙總策略
        self.strategies: list[AggregationStrategy] = [
            CoreNetworkFeatures(),
            PortStatisticsFeatures(),
            DurationStatisticsFeatures(),
            EntropyFeatures(),
            PortTypeFeatures(),
        ]

    def aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        依序執行所有已註冊的彙總策略。
        """
        logging.debug(f"在 {len(df)} 筆記錄上執行 {len(self.strategies)} 個彙總策略")
        for strategy in self.strategies:
            df = strategy.aggregate(df)
        return df

    def extract_features(self):
        """
        執行完整的特徵抽取流程。
        """
        logging.info("開始抽取特徵...")

        # 1. OS 標籤對應
        self.df["os"] = self.df["os"].map(self.os_mapping)
        self.df = self.df.dropna(subset=["os"], how="any").reset_index(drop=True)
        self.df["os"] = self.df["os"].astype(int)

        # 2. 基本特徵正規化
        self.df["ttl"] = (2 ** np.ceil(np.log2(self.df["ttl"].replace(0, 1)))) / 128
        self.df["syn"] = self.df["syn"] / 64
        self.df["win"] = self.df["win"] / 65535

        logging.debug(
            "欄位 (時間分組處理前): %s", self.df.columns
        )

        # 3. 時間分組特徵處理
        self.df = self.process_time_grouped_features(self.df, self.freq)

        self.dataset = Dataset.from_pandas(self.df, preserve_index=False)
        logging.info("特徵抽取完成")

    def process_time_grouped_features(self, df: pd.DataFrame, freq: int) -> pd.DataFrame:
        df["ts"] = pd.to_datetime(df["ts"])
        df = df.sort_values("ts")

        logging.info(f"處理時間分組特徵，頻率: {freq} 小時")
        
        # 儲存原始欄位，以便後續區分
        exists_cols = set(df.columns)

        grouped = df.groupby(pd.Grouper(key="ts", freq=f"{freq}h"))
        temp_df_list = []

        for _, group in grouped:
            if group.shape[0] > 0:
                group = self.aggregate(group)
                temp_df_list.append(group)

        if not temp_df_list:
            logging.warning("沒有資料可供彙總")
            return df
            
        temp_df = pd.concat(temp_df_list, ignore_index=True)

        # 4. 特徵標準化 (僅限聚合特徵)
        cols_to_scale = [col for col in config.FEATURE_COLS_TO_SCALE if col in temp_df.columns]
        if cols_to_scale:
            logging.debug(f"正在標準化欄位: {cols_to_scale}")
            scaled_data = self.scaler.fit_transform(temp_df[cols_to_scale])
            temp_df[cols_to_scale] = scaled_data

        # 5. 其他特徵正規化 (範圍)
        if "prg" in temp_df.columns:
            temp_df["prg"] = temp_df["prg"] / 65536
            temp_df["maxp"] = temp_df["maxp"] / 65536
            temp_df["minp"] = temp_df["minp"] / 65536

        # 6. 為新特徵加上頻率後綴
        new_cols = set(temp_df.columns) - exists_cols
        rename_map = {col: f"{col}_{freq}h" for col in new_cols}
        df = temp_df.rename(columns=rename_map)

        logging.debug("欄位 (時間分組處理後): %s", df.columns)
        return df

    def save_to_disk(self):
        logging.info("儲存已抽取特徵的資料集")
        output_name = f"{self.dataset_name}_freq{self.freq}h"
        path = self.output_dir / output_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset.save_to_disk(path)
        logging.info(f"已儲存至: {path}")


# --- 模組協調函式 ---

def merging_datasets(dataset_name: str, src_dir: Path = config.FEATURED_DATA_DIR) -> Dataset:
    """
    合併來自不同頻率 (freq) 的同名資料集特徵 (axis=1)。
    """
    logging.info(f"正在合併 {dataset_name} 的特徵資料集")
    files = [file for file in src_dir.iterdir() if file.stem.startswith(dataset_name)]
    logging.debug(f"找到的檔案: {files}")

    if not files:
        logging.warning(f"在 {src_dir} 中找不到 {dataset_name} 的檔案")
        raise FileNotFoundError(f"找不到 {dataset_name} 的特徵檔案")

    datasets = [load_from_disk(file) for file in files]

    # 取得所有資料集的共通欄位
    if not datasets:
        logging.error("沒有載入任何資料集可供合併")
        raise ValueError("沒有資料集可合併")

    common_columns = set(datasets[0].column_names)
    for dataset in datasets[1:]:
        common_columns.intersection_update(dataset.column_names)

    logging.debug(f"共通欄位: {common_columns}")

    # 合併 (Concatenate)
    merged_dataset = datasets[0]
    logging.debug(f"合併前 shape: {merged_dataset.shape}")
    
    for dataset in datasets[1:]:
        # 移除共通欄位，只保留該頻率特有的新特徵
        unique_cols_to_add = set(dataset.column_names) - common_columns
        if not unique_cols_to_add:
            logging.warning("某個資料集沒有獨特的欄位可供合併，跳過")
            continue
            
        cols_to_remove = set(dataset.column_names) - unique_cols_to_add
        to_merge_dataset = dataset.remove_columns(list(cols_to_remove))
        
        # 確保行數一致
        if to_merge_dataset.shape[0] != merged_dataset.shape[0]:
            logging.error(f"Shape 不匹配: {merged_dataset.shape[0]} vs {to_merge_dataset.shape[0]}")
            # 這裡可能需要更複雜的對齊邏輯，但暫時先報錯
            raise ValueError("資料集合併時行數不一致")

        merged_dataset = concatenate_datasets([merged_dataset, to_merge_dataset], axis=1)
        logging.debug(f"合併後 shape: {merged_dataset.shape}")

    return merged_dataset


def eliminate_outliers(dataset: Dataset) -> Dataset:
    """
    (此功能在原始碼中被註解，暫時保留)
    使用 IsolationForest 移除異常值。
    """
    logging.warning("注意：異常值移除功能 (eliminate_outliers) 目前未啟用。")
    return dataset
    # logging.info("Eliminate outliers")
    # ... (原始邏輯)


def save_merged_dataset(merged_dataset: Dataset, dataset_name: str, dst_dir: Path = config.PREPARED_DATA_DIR):
    dst_dir.mkdir(parents=True, exist_ok=True)
    output_path = dst_dir / dataset_name
    merged_dataset.save_to_disk(output_path)
    logging.info(f"已合併的資料集儲存至: {output_path}")


def run_feature_engineering(freqs: list[int] = [24]):
    """
    執行所有資料集的特徵工程流程。
    """
    logging.info("===== 開始執行特徵工程階段 =====")
    
    if not config.PROCESSED_DATA_DIR.exists():
        logging.warning(f"已處理資料目錄不存在: {config.PROCESSED_DATA_DIR}")
        return

    for dataset_path in config.PROCESSED_DATA_DIR.iterdir():
        if not dataset_path.is_dir():
            continue
            
        logging.info(f"--- 正在處理資料集: {dataset_path.stem} ---")
        
        # 1. 針對每個頻率抽取特徵
        for freq in freqs:
            fx = FeatureExtraction(dataset_path, freq=freq)
            # 檢查是否已存在 (可選)
            output_name = f"{fx.dataset_name}_freq{freq}h"
            if (config.FEATURED_DATA_DIR / output_name).exists():
                logging.info(f"特徵檔案 {output_name} 已存在，跳過抽取。")
                continue
            
            try:
                fx.extract_features()
                fx.save_to_disk()
            except Exception as e:
                logging.error(f"為 {dataset_path.stem} (freq={freq}) 抽取特徵時失敗: {e}", exc_info=True)
                continue

        # 2. 合併不同頻率的特徵
        try:
            merged_dataset = merging_datasets(dataset_path.stem)
            
            # 3. 移除異常值 (可選)
            # merged_dataset = eliminate_outliers(merged_dataset)
            
            # 4. 儲存最終備妥的資料
            save_merged_dataset(merged_dataset, dataset_path.stem)
        except FileNotFoundError:
             logging.warning(f"找不到 {dataset_path.stem} 的任何特徵檔案可供合併，跳過。")
        except Exception as e:
            logging.error(f"合併或儲存 {dataset_path.stem} 時失敗: {e}", exc_info=True)

    logging.info("===== 特徵工程階段結束 =====")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Data feature engineering script")
    parser.add_argument(
        "freq", type=int, nargs='+', default=[24],
        help="用於特徵抽取的頻率 (小時)，可多選，以空格分隔"
    )
    args = parser.parse_args()
    logging.debug(f"Arguments: {args.freq}")
    run_feature_engineering(args.freq)
