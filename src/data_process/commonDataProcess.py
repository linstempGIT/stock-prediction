from pathlib import Path
import logging
import joblib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator

class CommonDataProcessor:

    def __init__(self, ts_code, start_date=None, end_date=None):
        self.ts_code = ts_code
        self.start_date = start_date if start_date else 'all'
        self.end_date = end_date if end_date else 'now'

        self.project_root = Path(__file__).resolve().parent.parent.parent

        self.df = None
        self.logger = self.setup_logger()

    def setup_logger(self):
        """设置日志记录器"""
        logger = logging.getLogger('CommonDataProcessor')
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        return logger
    
    def daily_data_clean(self):

        if self.df is None:
            self.logger.info("请先加载数据")
            return None
        
        df = self.df

        self.logger.info("开始数据清洗...")
        original_count = len(self.df)

        # 删除包含缺失值的行
        df = df.dropna()
        
        # 删除重复行
        df = df.drop_duplicates(subset=['trade_date'])
        
        # 过滤异常值
        # 过滤所有价格列和前收盘价为正数
        price_columns = ['open', 'high', 'low', 'close', 'pre_close']
        df = df[(df[price_columns] > 0).all(axis=1)]
        
        # 确保日期列格式正确
        df['trade_date'] = pd.to_numeric(df['trade_date'], errors='coerce')
        df = df.dropna(subset=['trade_date'])
        df['trade_date'] = df['trade_date'].astype(int)
        
        cleaned_count = len(self.df)
        self.logger.info(f"数据清洗完成，原始数据: {original_count}条，清洗后: {cleaned_count}条")

        # 按日期排序
        df.sort_values('trade_date', inplace=True)

        return df
    
    def load_raw_data(self):
        
        # 构建数据文件路径
        data_filename = f"{self.ts_code}_{self.start_date}_{self.end_date}.csv"
        data_dir = self.project_root / 'data' / 'dailyInfo'
        data_path = data_dir / data_filename

        self.logger.info(f"数据文件路径: {data_path}")

        # 检查文件是否存在
        if not data_path.exists():
            self.logger.info(f"错误：文件 {data_path} 不存在")
        try:
            self.df = pd.read_csv(data_path)
            return True
        except Exception as e:
            self.logger.info(f"处理过程中发生错误: {str(e)}")
            return False
        
    def save_processed_data(self, processed_data, dir, *suffix):
        save_dir = self.project_root / 'data' / 'processed' / dir
        save_dir.mkdir(parents=True, exist_ok=True)

        base_filename = f"{self.ts_code}_{self.start_date}_{self.end_date}"
        suffix_str = '_'.join(suffix) if suffix else ''
        final_filename_stem = f"{base_filename}_{suffix_str}"

        if isinstance(processed_data, pd.DataFrame):
            filename = f"{final_filename_stem}.csv"
            save_path = save_dir / filename
            processed_data.to_csv(save_path, index=False)
        elif isinstance(processed_data, np.ndarray):
            filename = f"{final_filename_stem}.npy"
            save_path = save_dir / filename
            np.save(save_path, processed_data)
        # Add a check for scikit-learn scalers/estimators
        elif isinstance(processed_data, BaseEstimator): 
            filename = f"{final_filename_stem}.joblib" # Save as .joblib
            save_path = save_dir / filename
            joblib.dump(processed_data, save_path)
        else:
            self.logger.error(f"不支持的数据类型进行保存: {type(processed_data)}")
            raise TypeError(f"Unsupported data type for saving: {type(processed_data)}. Only pandas DataFrame, numpy ndarray, or scikit-learn estimators are currently supported.")

        self.logger.info(f"处理后的数据已保存至: {save_path}")