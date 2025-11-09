"""
analysis.py
負責第三階段：模型分析。
包含 ShapAnalyzer 類別，專注於 SHAP 分析、繪圖和特徵貢獻度計算。
"""
import logging
import re
from pathlib import Path
from typing import List, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from joblib import load
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

import config
import utils

# utils.setup_logging(__file__)

class ShapAnalyzer:
    """
    SHAP 分析器。
    負責載入模型和資料，執行 SHAP 分析、繪製圖表和計算貢獻度。
    """
    def __init__(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        label_encoder: LabelEncoder,
        model_prefix: str,
    ):
        logging.info(f"初始化 ShapAnalyzer: {model_prefix}")
        self.model = model
        
        self.original_feature_names = list(X_test.columns) # 儲存原始欄位名稱
        # 建立一個清理過的特徵名稱列表 (例如移除 '_24h')
        # 使用正則表達式來移除 _<數字>h 這樣的後綴
        self.clean_feature_names = [re.sub(r'_\d+h$', '', col) for col in X_test.columns]
        
        self.X_test_df = X_test.copy() # 複製以避免 InplaceWarning
        self.X_test_df.columns = self.clean_feature_names
        self.feature_names = self.clean_feature_names # 更新 self.feature_names

        self.y_test = y_test
        self.label_encoder = label_encoder
        self.model_prefix = model_prefix
        
        self.fig_base_dir = config.FIGURES_DIR / self.model_prefix
        self.fig_base_dir.mkdir(parents=True, exist_ok=True)
        
        self.explainer = None
        self.shap_values = None
        self.y_pred = None
        self.y_pred_proba = None
        self.display_labels = self._get_display_labels()

    def _get_display_labels(self) -> List[str]:
        """從 LabelEncoder 獲取用於圖表顯示的標籤名稱"""
        try:
            decoded_labels_numeric = self.label_encoder.classes_
            
            # 獲取完整標籤名稱
            full_labels = [config.OS_MAPPING_DISPLAY.get(l, str(l)) for l in decoded_labels_numeric]
            
            # 僅保留第一個逗號前的名稱作為代表
            short_labels = [label.split(',')[0] for label in full_labels]
            
            return short_labels
        except Exception as e:
            logging.error(f"獲取 display_labels 失敗: {e}. 將使用原始編碼標籤。")
            return [str(l) for l in self.label_encoder.classes_]

    def run_predictions(self):
        """執行預測以供分析使用"""
        logging.debug("正在執行模型預測...")
        self.y_pred = self.model.predict(self.X_test_df)
        self.y_pred_proba = self.model.predict_proba(self.X_test_df)
        logging.debug("預測完成")

    def plot_correlation_heatmap(self):
        """繪製特徵相關性熱力圖"""
        logging.info("繪製相關性熱力圖...")
        fig_dir = self.fig_base_dir / "correlation_heatmap"
        fig_dir.mkdir(parents=True, exist_ok=True)
        
        corr = self.X_test_df.corr()
        plt.figure(figsize=(len(self.feature_names), len(self.feature_names)))
        sns.heatmap(corr, annot=True, cmap="vlag", fmt=".2f")
        plt.title("Feature Correlation Heatmap")
        plt.savefig(fig_dir / "correlation_heatmap.png", bbox_inches='tight')
        plt.close()

    def plot_confusion_matrix(self):
        """繪製並儲存標準化的混淆矩陣熱力圖"""
        if self.y_pred is None:
            self.run_predictions()
            
        logging.info("繪製混淆矩陣...")
        fig_dir = self.fig_base_dir / "confusion_matrix"
        fig_dir.mkdir(parents=True, exist_ok=True)

        cm = confusion_matrix(self.y_test, self.y_pred, labels=self.label_encoder.classes_)
        # 按總數標準化
        cm_normalized = cm.astype("float") / cm.sum()

        fig_size = max(8, len(self.display_labels) * 1.5)
        plt.figure(figsize=(fig_size, fig_size * 0.8))

        s = sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".2%",
            cmap="Blues",
            xticklabels=self.display_labels,
            yticklabels=self.display_labels,
        )
        s.set(xlabel="Predicted Label", ylabel="True Label")
        plt.title("Confusion Matrix (Normalized by Total)")
        plt.tight_layout()
        plt.savefig(fig_dir / "confusion_matrix.png")
        plt.close()

    def run_shap_analysis(self):
        """初始化 Explainer 並計算 SHAP 值"""
        logging.info("執行 SHAP 分析 (計算 SHAP 值)...")
        # 假設模型是樹模型 (XGBoost)
        self.explainer = shap.TreeExplainer(self.model)
        self.shap_values = self.explainer(self.X_test_df)
        logging.info("SHAP 值計算完畢")

    def _get_sample_indices(self) -> dict:
        """獲取每個類別中預測機率最高的 N 個樣本索引"""
        if self.y_pred is None:
            self.run_predictions()

        sample_indices = {}
        y_pred_df = pd.DataFrame(self.y_pred, columns=["pred"])
        
        for encoded_label in self.label_encoder.classes_:
            # 找到所有被預測為此類別的樣本索引
            target_indices = y_pred_df[y_pred_df["pred"] == encoded_label].index
            if len(target_indices) == 0:
                continue
                
            # 獲取這些樣本對應此類別的機率，並排序
            probs_for_label = self.y_pred_proba[target_indices, encoded_label]
            top_n_indices = np.argsort(probs_for_label)[::-1][:config.SHAP_TOP_N_SAMPLES]
            
            # 轉換回原始 X_test_df 的索引
            sample_indices[encoded_label] = target_indices[top_n_indices]
            
        return sample_indices

    def plot_waterfall_plots(self):
        """繪製 SHAP 瀑布圖"""
        if self.shap_values is None: self.run_shap_analysis()
        
        logging.info("繪製 SHAP 瀑布圖...")
        fig_dir = self.fig_base_dir / "waterfall_plots"
        fig_dir.mkdir(parents=True, exist_ok=True)
        sample_indices = self._get_sample_indices()

        for encoded_label, indices in sample_indices.items():
            try:
                # 這裡使用 self.display_labels (已經被縮短的標籤)
                label_name = self.display_labels[
                    np.where(self.label_encoder.classes_ == encoded_label)[0][0]
                ]
            except Exception:
                label_name = f"Class_{encoded_label}"
                
            for index in indices:
                plt.figure() # 確保每次都是新圖
                shap.plots.waterfall(self.shap_values[index, :, encoded_label], show=False)
                plt.title(f"Waterfall (Sample {index} - Pred: {label_name})")
                plt.savefig(fig_dir / f"waterfall_{label_name}_{index}.png", bbox_inches='tight')
                plt.close()

    def plot_summary_plots(self):
        """繪製 SHAP 總結圖 (beeswarm)"""
        if self.shap_values is None: self.run_shap_analysis()

        logging.info("繪製 SHAP 總結圖...")
        fig_dir = self.fig_base_dir / "summary_plots"
        fig_dir.mkdir(parents=True, exist_ok=True)

        # 總體總結圖
        plt.figure()
        shap.summary_plot(self.shap_values, self.X_test_df, show=False)
        plt.title("SHAP Summary Plot (All Classes)")
        plt.savefig(fig_dir / "summary_plot_all_classes.png", bbox_inches='tight')
        plt.close()

        # 分類別總結圖
        for c in self.label_encoder.classes_:
            try:
                # 這裡的 c 是原始標籤 (例如 1, 2, 4, 5)，而不是編碼後的 (0, 1, 2, 3)
                # 因此我們需要先找到 c 在 le.classes_ 中的索引
                encoded_c_index = np.where(self.label_encoder.classes_ == c)[0][0]
                label_name = self.display_labels[encoded_c_index]
            except Exception:
                 label_name = f"Class_{c}"
            
            plt.figure()
            shap.summary_plot(self.shap_values[:, :, encoded_c_index], self.X_test_df, show=False)
            plt.title(f"SHAP Summary Plot (Class: {label_name})")
            plt.savefig(fig_dir / f"summary_plot_class_{label_name}.png", bbox_inches='tight')
            plt.close()

    def plot_absolute_mean_shap(self):
        """繪製 SHAP 平均絕對值條形圖 (特徵重要性)"""
        if self.shap_values is None: self.run_shap_analysis()
        if self.y_pred is None: self.run_predictions()

        logging.info("繪製 SHAP 平均絕對值條形圖...")
        fig_dir = self.fig_base_dir / "absolute_mean_shap_plot"
        fig_dir.mkdir(parents=True, exist_ok=True)

        # 獲取每個樣本「被預測類別」的 SHAP 值
        new_shap_values = []
        for i, pred_class_encoded in enumerate(self.y_pred):
            # y_pred 是編碼後的 [0, 1, 2, 3]
            new_shap_values.append(self.shap_values.values[i][:, pred_class_encoded])
        
        # 建立一個新的 SHAP Explanation 物件
        pred_class_shap_values = shap.Explanation(
            values=np.array(new_shap_values),
            base_values=self.shap_values.base_values[np.arange(len(self.y_pred)), self.y_pred],
            data=self.X_test_df.values,
            feature_names=self.feature_names
        )

        plt.figure()
        shap.plots.bar(pred_class_shap_values, show=False)
        plt.title("Mean Absolute SHAP (Based on Predicted Class)")
        plt.savefig(fig_dir / "absolute_mean_shap.png", bbox_inches='tight')
        plt.close()

    def calculate_feature_contributions(self) -> pd.DataFrame:
        """
        計算特徵對「預測貢獻度」和「錯誤貢獻度」。
        這是原始碼中 `task_once` 結尾的複雜邏輯。
        """
        if self.shap_values is None: self.run_shap_analysis()
        if self.explainer is None: self.run_shap_analysis()
            
        logging.info("計算特徵貢獻度...")

        # 1. 獲取每個樣本「真實類別」的 SHAP 值
        new_shap_values = []
        new_base_values = []
        for i, truth_class_encoded in enumerate(self.y_test):
            # y_test 是編碼後的 [0, 1, 2, 3]
            new_shap_values.append(self.shap_values.values[i][:, truth_class_encoded])
            new_base_values.append(self.explainer.expected_value[truth_class_encoded])
        
        y_truth_shap_values = np.array(new_shap_values) # (n_samples, n_features)
        y_truth_base_values = np.array(new_base_values) # (n_samples,)

        # 2. 計算「預測貢獻度」 (Prediction Contribution)
        # = 真實類別 SHAP 值的平均絕對值
        prediction_contribution = np.abs(y_truth_shap_values).mean(0)
        
        # 3. 計算「錯誤貢獻度」 (Error Contribution)
        # 總 SHAP 值 (真實類別)
        y_truth_shap_total_value = y_truth_shap_values.sum(axis=1) + y_truth_base_values
        
        # 假設「缺少」某特徵 i 時的總 SHAP 值
        y_truth_wo_feature = (
            np.tile(y_truth_shap_total_value, (len(self.feature_names), 1)).T 
            - y_truth_shap_values
        )
        
        # 總 SHAP 值 (所有類別加總)
        total_shap_effect_per_feature = self.shap_values.values.sum(axis=2)

        error_diff = total_shap_effect_per_feature - np.abs(y_truth_shap_values)
        error_contribution = error_diff.mean(0)


        # 4. 建立貢獻度 DataFrame
        contribution_df = pd.DataFrame(
            {
                "Feature": self.original_feature_names, # 這裡使用原始特徵名稱
                "Prediction_Contribution": prediction_contribution,
                "Error_Contribution": error_contribution,
            }
        )
        contribution_df.set_index("Feature", inplace=True)
        # 按照「錯誤貢獻度」降序排序
        contribution_df.sort_values("Error_Contribution", ascending=False, inplace=True)

        # 5. 儲存貢獻度 CSV
        config.CONTRIBUTIONS_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = config.CONTRIBUTIONS_DIR / f"{self.model_prefix}_contributions.csv"
        contribution_df.to_csv(csv_path)
        logging.info(f"特徵貢獻度已儲存至: {csv_path}")

        # 6. 繪製貢獻度散點圖
        self.plot_contribution_scatter(contribution_df)

        return contribution_df

    def plot_contribution_scatter(self, contribution_df: pd.DataFrame):
        """繪製 預測貢獻度 vs 錯誤貢獻度 散點圖"""
        logging.info("繪製貢獻度散點圖...")
        fig_dir = self.fig_base_dir / "error_contribution_plot"
        fig_dir.mkdir(parents=True, exist_ok=True)
        
        plt.figure()
        sns.scatterplot(
            data=contribution_df, 
            x="Prediction_Contribution", 
            y="Error_Contribution"
        )
        
        # 標記前 5 個錯誤貢獻特徵
        # 這裡我們使用清理過的名稱來標註圖表，避免擁擠
        clean_name_map = dict(zip(self.original_feature_names, self.clean_feature_names))
        
        for feature_name_orig in contribution_df.index[:5]:
            clean_name = clean_name_map.get(feature_name_orig, feature_name_orig)
            plt.annotate(
                clean_name, # 使用清理過的名稱標註
                (
                    contribution_df.loc[feature_name_orig, "Prediction_Contribution"], 
                    contribution_df.loc[feature_name_orig, "Error_Contribution"]
                )
            )
            
        plt.xlabel("Prediction Contribution")
        plt.ylabel("Error Contribution")
        plt.title("Prediction Contribution vs Error Contribution")
        plt.savefig(fig_dir / "error_contribution_plot.png", bbox_inches='tight')
        plt.close()

    def run_all_analyses(self) -> pd.DataFrame:
        """
        執行所有分析步驟並返回貢獻度 DataFrame。
        """
        logging.info(f"===== 開始分析: {self.model_prefix} =====")
        try:
            self.run_predictions()
            # self.plot_correlation_heatmap()
            self.plot_confusion_matrix()
            self.run_shap_analysis()
            # self.plot_waterfall_plots()
            # self.plot_summary_plots()
            # self.plot_absolute_mean_shap()
            # self.plot_force_plots()
            # self.plot_decision_plots()
            
            contribution_df = self.calculate_feature_contributions()
            logging.info(f"===== 分析完成: {self.model_prefix} =====")
            return contribution_df
            
        except Exception as e:
            logging.error(f"分析 {self.model_prefix} 時失敗: {e}", exc_info=True)
            raise
