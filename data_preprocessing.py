"""
data_preprocessing.py
負責第一階段：載入原始資料、清理、轉換並儲存為 processed 格式。
包含 ABCDataPreprocess 介面、DatasetPreprocess 基礎類別、
具體的資料集子類，以及 DataPreparationFactory 工廠。
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd
from datasets import Dataset  # type: ignore

import config
import utils

# 設定此模組的日誌
utils.setup_logging(__file__)

class ABCDataPreprocess(ABC):
    """資料前處理的抽象基礎類別 (ABC)"""
    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def save_data(self, data: pd.DataFrame):
        pass

    @abstractmethod
    def run_pipeline(self):
        pass


class DatasetPreprocess(ABCDataPreprocess):
    """
    資料前處理的基礎類別，定義了通用的處理流程。
    """
    def __init__(self, dataset_name: str, usecols: list, sep: str = ","):
        self.check_column = config.PREPROCESS_COLUMNS
        self.input_dir = config.RAW_DATA_DIR
        self.output_dir = config.PROCESSED_DATA_DIR
        self.dataset_name = dataset_name
        self.usecols = usecols
        self.sep = sep
        self.directory = self.input_dir / self.dataset_name
        self.filenames = self.directory.glob("*.csv")

        logging.info(f"準備處理 {self.dataset_name} 資料集")
        logging.debug(f"Input directory: {self.directory}")

    def load_data(self) -> pd.DataFrame:
        logging.info(f"正在載入 {self.dataset_name} 資料集...")
        df = pd.concat(
            [
                pd.read_csv(file, usecols=self.usecols, sep=self.sep)
                for file in self.filenames
            ]
        )
        logging.debug(f"已載入 {self.dataset_name} 資料集，欄位: {df.columns}")
        return df

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"正在清理 {self.dataset_name} 資料集...")
        
        # 檢查欄位數量是否匹配
        if len(self.usecols) != len(self.check_column):
            logging.error(
                f"欄位數量不匹配: 原始 {len(self.usecols)} vs 目標 {len(self.check_column)}"
            )
            raise ValueError("usecols 和 check_column 長度必須相同")

        data = data.rename(
            columns={
                key: value
                for key, value in zip(self.usecols, self.check_column)
            }
        )
        return data.dropna(subset=self.check_column)

    def transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"正在轉換 {self.dataset_name} 資料集...")
        dtype = {
            "ts": "datetime64[ns]", "te": "datetime64[ns]",
            "sa": "str", "da": "str", "sp": "int", "dp": "int",
            "os": "str", "win": "int", "syn": "int", "ttl": "int",
        }
        data = data.astype(dtype)

        start_time = data["ts"]
        end_time = data["te"]
        duration = end_time - start_time
        duration_seconds = duration.dt.total_seconds()
        data["td"] = duration_seconds

        return data.loc[:, self.check_column + ["td"]]

    def save_data(self, data: pd.DataFrame):
        logging.info(f"正在儲存 {self.dataset_name} 資料集...")
        output_path = self.output_dir / self.dataset_name
        dataset = Dataset.from_pandas(data, preserve_index=False)
        dataset.save_to_disk(output_path)
        logging.info(f"資料集已儲存至: {output_path}")

    def run_pipeline(self):
        """執行完整的資料前處理流程"""
        data = self.load_data()
        data = self.clean_data(data)
        data = self.transform_data(data)
        self.save_data(data)


class Lasto18DatasetPreprocess(DatasetPreprocess):
    def __init__(self):
        super().__init__(
            dataset_name="lasto18",
            usecols=config.LASTO18_COLS,
            sep=";"
        )


class Lasto20DatasetPreprocess(DatasetPreprocess):
    def __init__(self):
        super().__init__(
            dataset_name="lasto20",
            usecols=config.LASTO20_COLS,
            sep=","
        )


class Lasto23DatasetPreprocess(DatasetPreprocess):
    def __init__(self):
        super().__init__(
            dataset_name="lasto23",
            usecols=config.LASTO23_COLS,
            sep=";"
        )


class DataPreparationFactory:
    """
    工廠模式：根據資料集名稱建立對應的前處理器物件。
    """
    def __init__(self):
        self.dataset_factory = {
            "lasto18": Lasto18DatasetPreprocess,
            "lasto20": Lasto20DatasetPreprocess,
            "lasto23": Lasto23DatasetPreprocess,
        }
        logging.info("資料準備工廠已初始化")
        logging.debug(f"可用的處理器: {list(self.dataset_factory.keys())}")

    def get_data_preparation(self, dataset: str) -> ABCDataPreprocess:
        if dataset in self.dataset_factory:
            logging.info(f"正在建立 {dataset} 的前處理器")
            return self.dataset_factory[dataset]()
        else:
            logging.error(f"無效的資料集名稱: {dataset}")
            raise ValueError("無效的資料集名稱")


def run_preprocessing():
    """
    執行所有資料集的前處理流程。
    """
    logging.info("===== 開始執行資料前處理階段 =====")
    factory = DataPreparationFactory()
    
    if not config.RAW_DATA_DIR.exists():
        logging.warning(f"原始資料目錄不存在: {config.RAW_DATA_DIR}")
        return

    for dataset_path in config.RAW_DATA_DIR.iterdir():
        if not dataset_path.is_dir():
            continue
            
        dataset_name = dataset_path.stem
        output_path = config.PROCESSED_DATA_DIR / dataset_name
        
        if output_path.exists():
            logging.info(f"{dataset_name} 資料集似乎已經處理過，跳過。")
            continue
            
        try:
            data_preparation = factory.get_data_preparation(dataset_name)
            data_preparation.run_pipeline()
            logging.info(f"成功處理 {dataset_name} 資料集")
        except ValueError:
            logging.warning(f"找不到 {dataset_name} 的處理器，跳過。")
        except Exception as e:
            logging.error(f"處理 {dataset_name} 時發生錯誤: {e}", exc_info=True)
            
    logging.info("===== 資料前處理階段結束 =====")

if __name__ == "__main__":
    run_preprocessing()
