"""
main.py
專案的主入口點 (Orchestrator)。
負責：
1. 解析命令列參數 (要執行的階段)。
2. 協調執行 data_preprocessing, feature_engineering。
3. 執行實驗流程 (應用策略模式)。
"""

import logging
import argparse
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Dict, Any

from xgboost import XGBClassifier
from joblib import load

import config
import utils
import data_preprocessing
import feature_engineering
from training import ModelTrainer
from analysis import ShapAnalyzer

# 設定主日誌
utils.setup_logging(__file__)


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
        n_features_to_eliminate: int = 13
    ):
        self.dataset_paths = dataset_paths
        self.estimator_class = estimator_class
        self.estimator_params = estimator_params
        self.n_features_to_eliminate = n_features_to_eliminate
        
        # 初始化每個資料集的「待消除特徵列表」
        self.map_eliminate_features: Dict[str, List[str]] = {
            dataset.stem: [] for dataset in self.dataset_paths
        }
        logging.info("初始化 RecursiveErrorContributionElimination 策略")
        logging.debug(f"將消除 {self.n_features_to_eliminate} 個特徵")

    def execute(self):
        logging.info("===== 開始執行「遞歸錯誤貢獻消除」實驗 =====")

        for i in range(self.n_features_to_eliminate):
            logging.info(f"--- 實驗疊代 {i+1} / {self.n_features_to_eliminate} ---")
            
            # 儲存此次疊代中每個資料集要移除的特徵
            features_to_add_to_elim_list: Dict[str, str] = {}

            for dataset_path in self.dataset_paths:
                dataset_name = dataset_path.stem
                current_elim_features = self.map_eliminate_features[dataset_name]
                
                logging.info(f"處理資料集: {dataset_name}")
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
                    model = trainer.estimator
                    X_test_df, y_test_s = trainer.get_test_data()
                    label_encoder = trainer.get_label_encoder()
                    
                    analyzer = ShapAnalyzer(
                        model=model,
                        X_test=X_test_df,
                        y_test=y_test_s,
                        label_encoder=label_encoder,
                        model_prefix=model_prefix
                    )
                    
                    # 執行所有分析並獲取貢獻度
                    contribution_df = analyzer.run_all_analyses()
                    
                    # 3. 找到貢獻度最高的特徵
                    if contribution_df.empty:
                        logging.warning(f"{dataset_name} 的貢獻度為空，無法消除特徵")
                        continue
                        
                    top_error_feature = contribution_df.index[0]
                    features_to_add_to_elim_list[dataset_name] = top_error_feature
                    logging.info(f"{dataset_name} 的下一個消除特徵: {top_error_feature}")

                except Exception as e:
                    logging.error(f"處理 {dataset_name} (疊代 {i}) 時失敗: {e}", exc_info=True)
            
            # 4. 更新待消除列表
            if not features_to_add_to_elim_list:
                logging.warning("沒有找到任何特徵可供消除，實驗提前終止。")
                break
                
            for dataset_name, feature_to_elim in features_to_add_to_elim_list.items():
                if feature_to_elim not in self.map_eliminate_features[dataset_name]:
                    self.map_eliminate_features[dataset_name].append(feature_to_elim)
                else:
                    logging.warning(f"特徵 {feature_to_elim} 已在 {dataset_name} 的消除列表中")

        logging.info("===== 實驗流程執行完畢 =====")


def main():
    """
    主函式：解析參數並協調執行不同階段。
    """
    parser = argparse.ArgumentParser(description="MLOps 專案主執行腳本")
    parser.add_argument(
        "--stage",
        type=str,
        nargs='+',
        choices=['preprocess', 'feature', 'experiment', 'all'],
        default=['all'],
        help="要執行的階段 (可多選)"
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
    
    args = parser.parse_args()
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
            return # 如果第一階段失敗，後續無法進行

    # --- 階段 2: 特徵工程 ---
    if 'feature' in stages:
        try:
            feature_engineering.run_feature_engineering(args.freq)
        except Exception as e:
            logging.error(f"特徵工程階段失敗: {e}", exc_info=True)
            return # 如果第二階段失敗，實驗無法進行

    # --- 階段 3: 實驗 (訓練與分析) ---
    if 'experiment' in stages:
        try:
            # 1. 設定模型
            estimator_class = XGBClassifier
            estimator_params = {"n_estimators": 100, "verbosity": 0, "n_jobs": -1}
            
            # 2. 獲取備妥的資料集
            if not config.PREPARED_DATA_DIR.exists():
                logging.error(f"找不到備妥的資料目錄: {config.PREPARED_DATA_DIR}")
                return
                
            dataset_paths = list(config.PREPARED_DATA_DIR.iterdir())
            dataset_paths = [p for p in dataset_paths if p.is_dir()] # 確保只選目錄
            
            if not dataset_paths:
                logging.error("在 PREPARED_DATA_DIR 中找不到任何資料集")
                return

            # 3. 實例化並執行策略
            strategy = RecursiveErrorContributionElimination(
                dataset_paths=dataset_paths,
                estimator_class=estimator_class,
                estimator_params=estimator_params,
                n_features_to_eliminate=args.n_features
            )
            strategy.execute()
            
        except Exception as e:
            logging.error(f"實驗階段失敗: {e}", exc_info=True)

    logging.info("===== 所有請求的階段均已執行完畢 =====")


if __name__ == "__main__":
    main()
