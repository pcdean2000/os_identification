"""
main.py
專案的主入口點 (Orchestrator)。
負責：
1. 解析命令列參數 (要執行的階段、使用的模型)。
2. 協調執行 data_preprocessing, feature_engineering。
3. 執行實驗流程 (應用策略模式)，支援透過參數指定模型。
"""

import logging
import argparse
import time
import sys
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

import config
import utils
import data_preprocessing
import feature_engineering
from training import ModelTrainer
from analysis import ShapAnalyzer

# 設定主日誌
# utils.setup_logging(__file__)


# --- 實驗流程策略模式 ---

class ExperimentStrategy(ABC):
    """實驗策略的抽象基礎類別"""
    @abstractmethod
    def execute(self):
        """執行實驗流程"""
        pass


class RecursiveErrorContributionElimination(ExperimentStrategy):
    """
    遞歸特徵消除 (RFE) 策略。
    基於 SHAP 的「錯誤貢獻度」來逐步消除特徵。
    """
    def __init__(
        self,
        dataset_paths: List[Path],
        estimator_class: Any,
        estimator_params: Dict,
        n_features_to_eliminate: int = 13,
        model_name: str = "model"
    ):
        self.dataset_paths = dataset_paths
        self.estimator_class = estimator_class
        self.estimator_params = estimator_params
        self.n_features_to_eliminate = n_features_to_eliminate
        self.model_name = model_name
        self.map_eliminate_features: Dict[str, List[str]] = {
            dataset.stem: [] for dataset in self.dataset_paths
        }
        logging.info(f"[{self.model_name}] 初始化 RecursiveErrorContributionElimination 策略")
        logging.debug(f"將消除 {self.n_features_to_eliminate} 個特徵")

    def execute(self):
        logging.info(f"===== [{self.model_name}] 開始執行「遞歸錯誤貢獻消除」實驗 =====")

        for i in range(self.n_features_to_eliminate):
            logging.info(f"--- [{self.model_name}] 實驗疊代 {i+1} / {self.n_features_to_eliminate} ---")
            features_to_add_to_elim_list: Dict[str, str] = {}

            for dataset_path in self.dataset_paths:
                dataset_name = dataset_path.stem
                current_elim_features = self.map_eliminate_features[dataset_name]
                
                logging.info(f"[{self.model_name}] 處理資料集: {dataset_name}")
                logging.info(f"目前已消除 {len(current_elim_features)} 個特徵: {current_elim_features}")

                try:
                    # 1. 訓練
                    estimator = self.estimator_class(**self.estimator_params)
                    trainer = ModelTrainer(
                        dataset_path=dataset_path,
                        estimator=estimator,
                        eliminate_features=current_elim_features
                    )
                    trainer.run()
                    trainer.save_model()
                    trainer.save_train_test_data()

                    # 2. 分析
                    model_prefix = trainer.get_model_prefix()
                    
                    # 獲取剛訓練好的模型和資料
                    X_test_df, y_test_s = trainer.get_test_data()
                    label_encoder = trainer.get_label_encoder()
                    
                    analyzer = ShapAnalyzer(
                        model=trainer.estimator,
                        X_test=X_test_df,
                        y_test=y_test_s,
                        label_encoder=label_encoder,
                        model_prefix=model_prefix
                    )
                    
                    # 執行所有分析並獲取貢獻度
                    contribution_df = analyzer.run_all_analyses()
                    
                    # 3. 找到貢獻度最高的特徵
                    if contribution_df.empty:
                        logging.warning(f"[{self.model_name}] {dataset_name} 的貢獻度為空，無法消除特徵")
                        continue

                    top_error_feature = contribution_df.index[0]
                    features_to_add_to_elim_list[dataset_name] = top_error_feature
                    logging.info(f"[{self.model_name}] {dataset_name} 的下一個消除特徵: {top_error_feature}")

                except Exception as e:
                    logging.error(f"[{self.model_name}] 處理 {dataset_name} (疊代 {i}) 時失敗: {e}", exc_info=True)
            
            # 4. 更新待消除列表
            if not features_to_add_to_elim_list:
                logging.warning(f"[{self.model_name}] 無特徵可消除，實驗終止。")
                break
                
            for dataset_name, feature_to_elim in features_to_add_to_elim_list.items():
                if feature_to_elim not in self.map_eliminate_features[dataset_name]:
                    self.map_eliminate_features[dataset_name].append(feature_to_elim)
                else:
                    logging.warning(f"特徵 {feature_to_elim} 已在 {dataset_name} 的消除列表中")

        logging.info(f"===== [{self.model_name}] 實驗流程執行完畢 =====")


def run_experiment_stage(dataset_paths: List[Path], n_features_to_eliminate: int, model_name_arg: str):
    """
    執行實驗階段：
    根據指定的模型名稱執行 RFE 策略。
    """
    
    # 延遲載入: 僅在此處呼叫函式以載入重型模型庫
    MODEL_CONFIGS = config.load_model_configs()
    
    model_key = model_name_arg.lower()

    # 1. 檢查模型是否存在於註冊表
    if model_key not in MODEL_CONFIGS: 
        available_models = list(MODEL_CONFIGS.keys())
        error_msg = f"找不到指定的模型: '{model_name_arg}'。可用模型: {available_models}"
        logging.error(error_msg)
        print(f"Error: {error_msg}")
        sys.exit(1)

    model_cfg = MODEL_CONFIGS[model_key] 
    estimator_class = model_cfg["class"]
    
    # 2. 檢查模型類別是否可用
    if estimator_class is None:
        error_msg = f"模型 '{model_name_arg}' 對應的套件未安裝，無法執行。"
        logging.error(error_msg)
        print(f"Error: {error_msg}")
        sys.exit(1) # 錯誤時結束程式

    # 3. SHAP 相容性檢查與警告
    if not model_cfg["tree_friendly"]:
        warn_msg = (
            f"\n{'!'*40}\n"
            f"!!! 警告: SHAP 相容性問題 !!!\n"
            f"您選擇的模型 '{model_name_arg}' 並非原生的樹模型。\n"
            f"目前的實驗流程依賴 SHAP TreeExplainer。\n"
            f"強制執行可能會導致分析階段失敗，或效率極低。\n"
            f"{'!'*40}\n"
        )
        logging.warning(warn_msg.replace("\n", " "))
        print(warn_msg, flush=True)
        time.sleep(2)
        print("... 2秒後繼續執行 ...\n")

    # 4. 執行實驗
    logging.info(f"準備以模型 [{model_key}] 開始實驗")
    try:
        strategy = RecursiveErrorContributionElimination(
            dataset_paths=dataset_paths,
            estimator_class=estimator_class,
            estimator_params=model_cfg["params"],
            n_features_to_eliminate=n_features_to_eliminate,
            model_name=model_key
        )
        strategy.execute()
    except Exception as e:
        logging.error(f"模型 [{model_key}] 實驗執行失敗: {e}", exc_info=True)
        sys.exit(1) # 實驗失敗時結束程式


def main():
    parser = argparse.ArgumentParser(description="MLOps 專案主執行腳本")
    parser.add_argument(
        "--stage",
        type=str,
        nargs='+',
        choices=['preprocess', 'feature', 'experiment', 'all'],
        default=['all'],
        help="要執行的階段 (可多選)。 'all' 代表所有階段。"
    )
    parser.add_argument(
        "--freq",
        type=int,
        nargs='+',
        default=[24],
        help="特徵工程的頻率 (小時)"
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=13,
        help="實驗階段要消除的特徵總數"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="xgboost",
        help=f"實驗階段使用的模型。預設: xgboost。可用: {config.SUPPORTED_MODEL_NAMES}" 
    )
    
    args = parser.parse_args()
    
    # 設定主日誌
    utils.setup_logging(__file__)
    
    stages = args.stage

    if 'all' in stages:
        stages = ['preprocess', 'feature', 'experiment']

    logging.info(f"將執行以下階段: {stages}")
    logging.info(f"特徵頻率: {args.freq}")

    # --- 階段 1: 資料前處理 ---
    if 'preprocess' in stages:
        try:
            data_preprocessing.run_preprocessing()
        except Exception as e:
            logging.error(f"資料前處理階段失敗: {e}", exc_info=True)
            sys.exit(1) # 錯誤時結束程式

    # --- 階段 2: 特徵工程 ---
    if 'feature' in stages:
        try:
            feature_engineering.run_feature_engineering(args.freq)
        except Exception as e:
            logging.error(f"特徵工程階段失敗: {e}", exc_info=True)
            sys.exit(1) # 錯誤時結束程式

    # --- 階段 3: 實驗 (訓練與分析) ---
    if 'experiment' in stages:
        try:
            # 獲取備妥的資料集
            if not config.PREPARED_DATA_DIR.exists():
                logging.error(f"找不到備妥的資料目錄: {config.PREPARED_DATA_DIR}")
                sys.exit(1) # 錯誤時結束程式

            dataset_paths = [p for p in config.PREPARED_DATA_DIR.iterdir() if p.is_dir()]
            
            if not dataset_paths:
                logging.error("在 PREPARED_DATA_DIR 中找不到任何資料集")
                sys.exit(1) # 錯誤時結束程式

            run_experiment_stage(
                dataset_paths=dataset_paths,
                n_features_to_eliminate=args.n_features,
                model_name_arg=args.model
            )
            
        except Exception as e:
            logging.error(f"實驗階段失敗: {e}", exc_info=True)
            sys.exit(1) # 錯誤時結束程式

    logging.info("===== 所有請求的階段均已執行完畢 =====")


if __name__ == "__main__":
    main()