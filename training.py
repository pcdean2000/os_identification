"""
training.py
負責第三階段：模型訓練與評估。
包含 ModelTrainer 類別，專注於資料準備、訓練、評估和儲存。
"""
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from datasets import load_from_disk
from joblib import dump
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, recall_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import config
import utils

# utils.setup_logging(__file__)

class ModelTrainer:
    """
    模型訓練器類別。
    負責：準備資料、訓練、評估、儲存模型和報告。
    """
    def __init__(
        self,
        dataset_path: Path,
        estimator: Any,
        eliminate_features: List[str] = [],
    ):
        logging.info(f"初始化 ModelTrainer: {dataset_path.stem}, Estimator: {type(estimator).__name__}")
        self.dataset = load_from_disk(dataset_path)
        self.df = self.dataset.to_pandas()
        self.estimator = estimator
        self.eliminate_features = eliminate_features
        self.dataset_name = dataset_path.stem
        
        self.model_prefix = (
            f"{type(self.estimator).__name__}_{self.dataset_name}_"
            f"cut-{len(self.eliminate_features)}-features"
        )
        logging.debug(f"模型前綴: {self.model_prefix}")

        self.feature_names: List[str] = []
        self.target_name: str = "os"
        self.X_raw = None
        self.y_raw = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.label_encoder = LabelEncoder()

    def get_model_prefix(self) -> str:
        return self.model_prefix

    def get_test_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        if self.X_test is None or self.y_test is None:
            logging.error("測試資料尚未準備好")
            raise ValueError("必須先執行 prepare_data()")
        return (
            pd.DataFrame(self.X_test, columns=self.feature_names), 
            pd.Series(self.y_test, name=self.target_name)
        )
    
    def get_feature_names(self) -> List[str]:
        return self.feature_names
        
    def get_label_encoder(self) -> LabelEncoder:
        return self.label_encoder

    def prepare_data(self):
        """準備訓練和測試資料"""
        logging.info("正在準備資料...")
        
        # 1. 移除不必要的欄位
        cols_to_drop = config.NON_FEATURE_COLS + self.eliminate_features
        # 確保只移除存在的欄位
        cols_to_drop_existing = [col for col in cols_to_drop if col in self.df.columns]
        self.df = self.df.drop(columns=cols_to_drop_existing, errors="ignore")

        # 2. 移除樣本數過少的標籤
        logging.debug(f"原始標籤分佈: {self.df['os'].value_counts()}")
        mask = self.df["os"].value_counts() > config.MIN_LABEL_COUNT
        self.df = self.df.loc[self.df["os"].isin(mask[mask].index)]
        logging.debug(f"清理後標籤分佈: {self.df['os'].value_counts()}")

        # 3. 標籤編碼
        self.df["os"] = self.label_encoder.fit_transform(self.df["os"])
        logging.info(f"標籤編碼: {list(self.label_encoder.classes_)}")

        # 4. 分離特徵和標籤
        _X_raw = self.df.drop(columns=[self.target_name])
        _y_raw = self.df[self.target_name]

        self.X_raw = _X_raw.to_numpy()
        self.y_raw = _y_raw.to_numpy()
        self.feature_names = list(_X_raw.columns)
        self.target_name = _y_raw.name

        logging.debug(f"X_raw shape: {self.X_raw.shape}")
        logging.debug(f"Y_raw unique: {np.unique(self.y_raw, return_counts=True)}")

        # 5. 分割資料集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_raw,
            self.y_raw,
            random_state=config.RANDOM_STATE,
            train_size=config.TRAIN_SIZE,
            test_size=config.TEST_SIZE,
            stratify=self.y_raw  # 確保分層抽樣
        )
        logging.info(f"資料準備完畢: Train={self.X_train.shape}, Test={self.X_test.shape}")

    def train(self):
        """訓練模型"""
        logging.info(f"開始訓練 {type(self.estimator).__name__}...")
        self.estimator.fit(self.X_train, self.y_train)
        logging.info("模型訓練完成")

    def evaluate(self) -> Dict[str, Any]:
        """評估模型並返回指標"""
        logging.info("開始評估模型...")
        y_pred = self.estimator.predict(self.X_test)

        report_str = classification_report(self.y_test, y_pred, zero_division=0)
        logging.info(f"\n{report_str}")

        accuracy = accuracy_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(self.y_test, y_pred, average="macro", zero_division=0)
        balanced_acc = balanced_accuracy_score(self.y_test, y_pred)
        cm = confusion_matrix(self.y_test, y_pred)

        logging.info(f"Accuracy: {accuracy:.4f}")
        logging.info(f"Recall (Macro): {recall:.4f}")
        logging.info(f"F1-score (Macro): {f1:.4f}")
        logging.info(f"Balanced Accuracy: {balanced_acc:.4f}")
        logging.debug(f"Confusion Matrix:\n{cm}")

        metrics = {
            "accuracy": accuracy,
            "recall_macro": recall,
            "f1_macro": f1,
            "balanced_accuracy": balanced_acc,
            "classification_report": report_str,
            "confusion_matrix": cm,
        }
        return metrics

    def save_report(self, metrics: Dict[str, Any]):
        """將評估報告儲存到檔案"""
        config.PERFORMANCES_DIR.mkdir(parents=True, exist_ok=True)
        report_path = config.PERFORMANCES_DIR / f"{self.model_prefix}.txt"
        
        logging.info(f"儲存效能報告至: {report_path}")
        with open(report_path, "w") as f:
            f.write(f"--- 效能報告: {self.model_prefix} ---\n\n")
            f.write(metrics["classification_report"])
            f.write("\n--- 總體指標 ---\n")
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Recall (Macro): {metrics['recall_macro']:.4f}\n")
            f.write(f"F1-score (Macro): {metrics['f1_macro']:.4f}\n")
            f.write(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}\n")
            f.write("\n--- 混淆矩陣 ---\n")
            f.write(f"{metrics['confusion_matrix']}\n")

    def save_model(self):
        """儲存訓練好的模型"""
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = config.MODELS_DIR / f"{self.model_prefix}.joblib"
        logging.info(f"儲存模型至: {model_path}")
        dump(self.estimator, model_path)

    def save_train_test_data(self):
        """儲存訓練/測試資料樣本以供分析"""
        dir_path = config.MODELS_DIR / f"Sample_{self.model_prefix}"
        dir_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"儲存樣本資料至: {dir_path}")
        
        np.save(dir_path / f"{self.model_prefix}_X_test.npy", self.X_test)
        np.save(dir_path / f"{self.model_prefix}_y_test.npy", self.y_test)
        np.save(dir_path / f"{self.model_prefix}_feature.npy", np.array(self.feature_names))
        np.save(dir_path / f"{self.model_prefix}_target.npy", np.array(self.target_name))
        dump(self.label_encoder, dir_path / f"{self.model_prefix}_label_encoder.joblib")

    def run(self) -> Dict[str, Any]:
        """執行完整的訓練和評估流程"""
        try:
            self.prepare_data()
            self.train()
            metrics = self.evaluate()
            self.save_report(metrics)
            return metrics
        except Exception as e:
            logging.error(f"ModelTrainer.run() 失敗: {e}", exc_info=True)
            raise
