import pandas as pd
import argparse
import sys
import logging
from src.data_process.commonDataProcess import CommonDataProcessor
from src.utils.tsCodeUtils import normalize_ts_code, is_valid_ts_code

class NNDataProcessor:
    def __init__(self, df, step = 30):
        """
        初始化NN数据处理类
        
        参数:
            df: 包含股票数据的DataFrame
            step: 滑动窗口大小 (默认30)
        """
        self.df = self.pre_opera_data(df)
        self.step = step
        self.logger = self.setup_logger()
    
    def setup_logger(self):
        """设置日志记录器"""
        logger = logging.getLogger('NNDataProcessor')
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        return logger
    
    def pre_opera_data(self, df):
        """预处理原始数据"""
        # 保留核心特征
        features = ['close']
        df = df[features]
        return df
    
    def process_data(self):
        """
        处理清洗后的数据，生成滑动窗口数据集
        """
        if len(self.df) == 0:
            self.logger.error("清洗后数据为空，无法处理")
            return pd.DataFrame()
        
        self.logger.info("开始数据处理...")
        
        # 确保有足够的数据
        if len(self.df) <= self.step:
            self.logger.error(f"数据不足，需要至少{self.step+1}条数据，当前只有{len(self.df)}条")
            return pd.DataFrame()
        
        # 创建滑动窗口数据集
        processed_data = []
        for i in range(len(self.df) - self.step):
            # 提取step天的收盘价
            window = self.df['close'].iloc[i:i+self.step].values.tolist()
            
            # 判断第step+1天是否上涨
            current_close = self.df['close'].iloc[i+self.step-1]
            next_close = self.df['close'].iloc[i+self.step]
            is_up = next_close > current_close
            
            # 添加标签
            window.append(is_up)
            processed_data.append(window)
        
        # 创建DataFrame
        columns = [f'day_{i+1}' for i in range(self.step)] + ['next_day_up']
        result_df = pd.DataFrame(processed_data, columns=columns)
        
        self.logger.info(f"数据处理完成，生成{len(result_df)}条样本")
        return result_df

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='每日股票数据处理')
    parser.add_argument('-c', '--ts_code', required=True, help='股票代码（如：000001.SZ）')
    parser.add_argument('--start_date', help='开始日期(YYYYMMDD)')
    parser.add_argument('--end_date', help='结束日期(YYYYMMDD)')
    parser.add_argument('--step', type=int, default=30, help='滑动窗口大小（默认30）')
    
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
        # 创建处理器实例
        processor = NNDataProcessor(df, args.step)
        
        # 处理数据
        processed_data = processor.process_data()
        
        if processed_data.empty:
            print("数据处理失败，请检查日志")
            sys.exit(1)
        
        # 保存结果到CSV文件
        commonDataProcessor.save_processed_data(processed_data, 'dailyInfo', str(args.step))
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
