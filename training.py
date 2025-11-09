"""
training.py
負責第三階段：模型訓練與評估。
包含 ModelTrainer 類別，專注於資料準備、訓練、評估和儲存。
"""
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from datasets import load_from_disk
from joblib import dump
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, recall_score
)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
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
        """獲取測試集 (用於 RFE / analysis.py)"""
        if self.X_test is None or self.y_test is None:
            logging.error("測試資料尚未準備好")
            raise ValueError("必須先執行 prepare_data(split=True)")
        return (
            pd.DataFrame(self.X_test, columns=self.feature_names), 
            pd.Series(self.y_test, name=self.target_name)
        )
    
    def get_feature_names(self) -> List[str]:
        return self.feature_names
        
    def get_label_encoder(self) -> LabelEncoder:
        return self.label_encoder

    def prepare_data(self, split: bool = True):
        """
        準備資料。
        如果 split=True (預設)，則分割為 X_train, X_test (用於 RFE)。
        如果 split=False，則只準備 X_raw, y_raw (用於 K-Fold CV)。
        """
        logging.info("正在準備資料...")
        
        # 1. 移除不必要的欄位
        cols_to_drop = config.NON_FEATURE_COLS + self.eliminate_features
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
        if split:
            logging.info(f"分割資料集 (train_size={config.TRAIN_SIZE})...")
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X_raw,
                self.y_raw,
                random_state=config.RANDOM_STATE,
                train_size=config.TRAIN_SIZE,
                test_size=config.TEST_SIZE,
                stratify=self.y_raw  # 確保分層抽樣
            )
            logging.info(f"資料準備完畢: Train={self.X_train.shape}, Test={self.X_test.shape}")
        else:
            logging.info(f"資料準備完畢: X_raw shape: {self.X_raw.shape}")


    def train(self, use_all_data: bool = False):
        """
        訓練模型。
        如果 use_all_data=True，使用 X_raw (用於最終模型)。
        如果 use_all_data=False (預設)，使用 X_train (用於 RFE)。
        """
        if use_all_data:
            if self.X_raw is None:
                raise ValueError("X_raw 為空。請先執行 prepare_data(split=False)")
            logging.info(f"開始訓練 {type(self.estimator).__name__} (on all data)...")
            self.estimator = clone(self.estimator) # 確保是新模型
            self.estimator.fit(self.X_raw, self.y_raw)
            logging.info("最終模型訓練完成")
        else:
            if self.X_train is None:
                raise ValueError("X_train 為空。請先執行 prepare_data(split=True)")
            logging.info(f"開始訓練 {type(self.estimator).__name__} (on train split)...")
            self.estimator.fit(self.X_train, self.y_train)
            logging.info("模型訓練完成")

    def evaluate(self) -> Dict[str, Any]:
        """評估模型 (on self.X_test)"""
        logging.info("開始評估模型 (on test split)...")
        if self.X_test is None:
            raise ValueError("X_test 為空。請先執行 prepare_data(split=True)")
            
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
        """將 (單次) 評估報告儲存到檔案"""
        config.PERFORMANCES_DIR.mkdir(parents=True, exist_ok=True)
        report_path = config.PERFORMANCES_DIR / f"{self.model_prefix}.txt"
        
        logging.info(f"儲存 (單次) 效能報告至: {report_path}")
        with open(report_path, "w", encoding="utf-8") as f:
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
        """儲存 (單次) 訓練/測試資料樣本"""
        dir_path = config.MODELS_DIR / f"Sample_{self.model_prefix}"
        dir_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"儲存樣本資料至: {dir_path}")
        
        np.save(dir_path / f"{self.model_prefix}_X_test.npy", self.X_test)
        np.save(dir_path / f"{self.model_prefix}_y_test.npy", self.y_test)
        np.save(dir_path / f"{self.model_prefix}_feature.npy", np.array(self.feature_names))
        np.save(dir_path / f"{self.model_prefix}_target.npy", np.array(self.target_name))
        dump(self.label_encoder, dir_path / f"{self.model_prefix}_label_encoder.joblib")

    def run(self) -> Dict[str, Any]:
        """執行 (單次) 的訓練和評估流程 (用於 RFE)"""
        try:
            self.prepare_data(split=True)
            self.train(use_all_data=False)
            metrics = self.evaluate()
            self.save_report(metrics)
            return metrics
        except Exception as e:
            logging.error(f"ModelTrainer.run() 失敗: {e}", exc_info=True)
            raise

    # --- 5-Fold CV 方法 ---
    
    def run_cross_validation(self, n_splits: int = 5) -> Dict[str, Any]:
        """執行 K-Fold 交叉驗證並返回平均指標"""
        
        # 1. 準備 *完整* 資料
        self.prepare_data(split=False) 
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)

        f1_scores, accuracy_scores, balanced_acc_scores, recall_scores = [], [], [], []
        class_labels = np.unique(self.y_raw)
        total_cm = np.zeros((len(class_labels), len(class_labels)))
        
        logging.info("="*50)
        logging.info(f"開始使用分層 {n_splits}-Fold 交叉驗證評估 {type(self.estimator).__name__} 模型")
        logging.info("="*50)

        total_train_time = 0.0
        last_fold_report = ""

        for fold, (train_index, test_index) in enumerate(skf.split(self.X_raw, self.y_raw)):
            logging.info(f"--- 第 {fold + 1}/{n_splits} 次摺疊 ---")
            
            X_train, X_test = self.X_raw[train_index], self.X_raw[test_index]
            y_train, y_test = self.y_raw[train_index], self.y_raw[test_index]
            
            # 複製估算器以確保每次都是乾淨的訓練
            model = clone(self.estimator) 
            
            fold_start_time = time.time()
            model.fit(X_train, y_train)
            total_train_time += (time.time() - fold_start_time)

            y_pred = model.predict(X_test)
            
            f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
            accuracy = accuracy_score(y_test, y_pred)
            balanced_acc = balanced_accuracy_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred, average='macro', zero_division=0)

            f1_scores.append(f1)
            accuracy_scores.append(accuracy)
            balanced_acc_scores.append(balanced_acc)
            recall_scores.append(recall)
            
            total_cm += confusion_matrix(y_test, y_pred, labels=class_labels)
            
            # 取得原始標籤名稱
            target_names_original = self.label_encoder.inverse_transform(class_labels)
            last_fold_report = classification_report(
                y_test, y_pred, 
                labels=class_labels, 
                target_names=[f"OS_{n}" for n in target_names_original], 
                zero_division=0
            )
            logging.info(f"此次 F1-score (macro): {f1:.4f}, Accuracy: {accuracy:.4f}\n")

        # 計算平均指標
        metrics = {
            "n_splits": n_splits,
            "total_training_time": total_train_time,
            "avg_f1_macro": np.mean(f1_scores),
            "std_f1_macro": np.std(f1_scores),
            "avg_accuracy": np.mean(accuracy_scores),
            "std_accuracy": np.std(accuracy_scores),
            "avg_balanced_accuracy": np.mean(balanced_acc_scores),
            "std_balanced_accuracy": np.std(balanced_acc_scores),
            "avg_recall_macro": np.mean(recall_scores),
            "std_recall_macro": np.std(recall_scores),
            "last_fold_report": last_fold_report,
            "average_confusion_matrix": total_cm / n_splits
        }
        
        logging.info("="*50)
        logging.info("交叉驗證最終評估結果")
        logging.info("="*50)
        logging.info(f"總訓練時間: {metrics['total_training_time']:.2f} 秒")
        logging.info(f"平均 F1-score (macro): {metrics['avg_f1_macro']:.4f} ± {metrics['std_f1_macro']:.4f}")
        logging.info(f"平均 Accuracy: {metrics['avg_accuracy']:.4f} ± {metrics['std_accuracy']:.4f}")
        logging.info(f"平均 Balanced Accuracy: {metrics['avg_balanced_accuracy']:.4f} ± {metrics['std_balanced_accuracy']:.4f}\n")
        
        return metrics

    def save_cv_report(self, metrics: Dict[str, Any]):
        """將 CV 評估報告儲存到檔案"""
        config.PERFORMANCES_DIR.mkdir(parents=True, exist_ok=True)
        # 報告檔名加上 _CV 以區分
        cv_model_prefix = (
            f"{type(self.estimator).__name__}_{self.dataset_name}_"
            f"cut-0-features"
        )
        report_path = config.PERFORMANCES_DIR / f"{cv_model_prefix}_CV.txt"
        
        logging.info(f"儲存 CV 效能報告至: {report_path}")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"--- 交叉驗證效能報告: {cv_model_prefix} ---\n\n")
            f.write(f"N_SPLITS: {metrics['n_splits']}\n")
            f.write(f"總訓練時間: {metrics['total_training_time']:.2f} 秒\n\n")
            
            f.write(f"平均 Accuracy: {metrics['avg_accuracy']:.4f} ± {metrics['std_accuracy']:.4f}\n")
            f.write(f"平均 Recall (Macro): {metrics['avg_recall_macro']:.4f} ± {metrics['std_recall_macro']:.4f}\n")
            f.write(f"平均 F1-score (Macro): {metrics['avg_f1_macro']:.4f} ± {metrics['std_f1_macro']:.4f}\n")
            f.write(f"平均 Balanced Accuracy: {metrics['avg_balanced_accuracy']:.4f} ± {metrics['std_balanced_accuracy']:.4f}\n\n")
            
            f.write("--- 最後一次摺疊的詳細分類報告 ---\n")
            f.write(metrics["last_fold_report"])
            f.write("\n--- 平均混淆矩陣 ---\n")
            f.write(f"{np.array(metrics['average_confusion_matrix'])}\n")

    def run_cv_and_train_final(self, n_splits: int = 5) -> Dict[str, Any]:
        """
        執行 CV 評估，然後訓練並儲存最終模型。
        """
        try:
            # 1. 執行 CV 評估 (這會呼叫 prepare_data(split=False))
            # 確保 eliminate_features 為空
            if len(self.eliminate_features) > 0:
                logging.warning(f"CV 模式下 eliminate_features ({self.eliminate_features}) 將被忽略。")
                self.eliminate_features = []
                # 重設 model_prefix
                self.model_prefix = (
                    f"{type(self.estimator).__name__}_{self.dataset_name}_"
                    f"cut-0-features"
                )

            cv_metrics = self.run_cross_validation(n_splits)
            
            # 2. 儲存 CV 評估報告
            self.save_cv_report(cv_metrics)
            
            # 3. 使用所有資料訓練最終模型
            # (prepare_data(split=False) 已經被 CV 呼叫過, X_raw/y_raw 已備妥)
            self.train(use_all_data=True)
            
            # 4. 儲存最終模型 (使用 cut-0-features 的 prefix)
            self.save_model()
            
            return cv_metrics
        except Exception as e:
            logging.error(f"ModelTrainer.run_cv_and_train_final() 失敗: {e}", exc_info=True)
            raise
