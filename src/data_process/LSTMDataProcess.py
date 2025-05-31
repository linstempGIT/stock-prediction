import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sys
import argparse
import logging
from typing import Tuple
from src.data_process.commonDataProcess import CommonDataProcessor
from src.utils.tsCodeUtils import normalize_ts_code, is_valid_ts_code

class LSTMAttentionDataProcessor:
    def __init__(self, df, seq_length: int = 60):
        """
        初始化LSTM+Attention数据处理器
        
        参数:
            df: 包含股票数据的DataFrame
            seq_length: 时间序列长度 (默认60)
        """
        self.df = self.pre_opera_data(df)
        self.seq_length = seq_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.logger = self.setup_logger()

    def setup_logger(self):
        """设置日志记录器"""
        logger = logging.getLogger('LSTMAttentionDataProcessor')
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        return logger

    def pre_opera_data(self, df) -> pd.DataFrame:
        """清洗并预处理原始数据"""
        # 保留核心特征
        features = ['open', 'high', 'low', 'close', 'vol']
        df = df[features]
        return df

    def create_sequences(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """创建LSTM训练序列"""
        scaled_data = self.scaler.fit_transform(data)
        sequences = []
        targets = []
        
        for i in range(len(scaled_data) - self.seq_length):
            seq = scaled_data[i:i+self.seq_length]
            target = scaled_data[i+self.seq_length, 3]  # 预测下一个时间步的收盘价
            sequences.append(seq)
            targets.append(target)
        
        return np.array(sequences), np.array(targets)

    def process_data(self):
        """
        处理清洗后的数据，生成sequences
        """

        if len(self.df) == 0:
            self.logger.error("清洗后数据为空，无法处理")
            return np.array(), np.array()
        
        self.logger.info("开始数据处理...")

        try:
            X, y = self.create_sequences(self.df)
            self.logger.info(f"数据处理完成，生成{len(X)}条样本")
            return X, y, self.scaler
        except Exception as e:
            self.logger.error(f"数据处理失败: {str(e)}")
            raise

def main():
    
    parser = argparse.ArgumentParser(description='LSTM+Attention模型数据处理')
    parser.add_argument('-c', '--ts_code', required=True, help='股票代码（如：000001.SZ）')
    parser.add_argument('--start_date', help='开始日期(YYYYMMDD)')
    parser.add_argument('--end_date', help='结束日期(YYYYMMDD)')
    parser.add_argument('--seq_length', type=int, default=60, help='时间序列长度（默认60）')
    
    args = parser.parse_args()
    
    # 标准化和验证股票代码
    normalized_ts_code = normalize_ts_code(args.ts_code)
    if not is_valid_ts_code(normalized_ts_code):
        print(f"股票代码格式错误: {args.ts_code}，应为'XXXXXX.XX'格式")
        exit(1)
    args.ts_code = normalized_ts_code

    commonDataProcessor = CommonDataProcessor(args.ts_code, args.start_date, args.end_date)

    # 加载数据
    if not commonDataProcessor.load_raw_data():
        sys.exit(1)

    # 清理数据
    df = commonDataProcessor.daily_data_clean()
    if df is None:
        sys.exit(1)

    try:
        processor = LSTMAttentionDataProcessor(df, args.seq_length)

        processed_data_X, processed_data_y, scaler_object = processor.process_data()

        if processed_data_X.size == 0 or processed_data_y.size == 0 or scaler_object is None:
            print("数据处理失败，请检查日志")
            sys.exit(1)
        
        commonDataProcessor.save_processed_data(processed_data_X, 'lstm', f'seq{args.seq_length}', 'X')
        commonDataProcessor.save_processed_data(processed_data_y, 'lstm', f'seq{args.seq_length}', 'y')
        commonDataProcessor.save_processed_data(scaler_object, 'lstm', f'seq{args.seq_length}', 'scaler')
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
