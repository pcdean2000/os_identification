"""
utils.py
提供共用功能，例如日誌設定。
"""
import logging
import os
from pathlib import Path
import config

def setup_logging(script_file: str):
    """
    設定日誌，同時輸出到控制台和檔案。
    日誌檔案儲存在 config.LOGS_DIR 中。
    """
    log_filename = f"{Path(script_file).stem}.log"
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    log_filepath = config.LOGS_DIR / log_filename

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_filepath),
            logging.StreamHandler(),
        ],
    )
    
    logging.info(f"日誌已設定。儲存至： {log_filepath}")
