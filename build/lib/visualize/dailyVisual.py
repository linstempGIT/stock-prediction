import os
import argparse
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path

def main():
    # 设置中文字体支持
    try:
        # 尝试查找更通用的中文字体
        font_path = None
        for font_name in ['Noto Sans CJK SC', 'SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']:
            try:
                font_path = fm.findfont(font_name)
                if font_path:
                    plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
                    break
            except:
                continue
        
        if font_path:
            print(f"使用字体: {font_path}")
        else:
            print("警告: 未找到中文字体，图表标签将使用英文")
            
        # 解决负号显示问题
        plt.rcParams['axes.unicode_minus'] = False
        
    except Exception as e:
        print(f"字体设置警告: {e}")

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Stock Daily Data Visualization')
    parser.add_argument('-c', '--ts_code', required=True, help='Stock code (required)')
    parser.add_argument('--start_date', help='Start date (YYYYMMDD)')
    parser.add_argument('--end_date', help='End date (YYYYMMDD)')
    
    args = parser.parse_args()
    
    # 构建预期的文件名
    start_str = args.start_date or 'all'
    end_str = args.end_date or 'now'
    data_filename = f"{args.ts_code}_{start_str}_{end_str}.csv"
    data_dir = Path(__file__).parent.parent / 'data' / 'dailyInfo'
    data_path = data_dir / data_filename
    
    # 检查数据文件是否存在
    if not data_path.exists():
        print(f"Data file {data_path} not found, fetching data...")
        # 构建命令调用 getDaily.py
        cmd = [
            'python', 
            str(Path(__file__).parent.parent / 'utils' / 'getDaily.py'),
            '-c', args.ts_code
        ]
        if args.start_date:
            cmd.extend(['--start_date', args.start_date])
        if args.end_date:
            cmd.extend(['--end_date', args.end_date])
        
        # 执行命令
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Data fetch failed: {result.stderr}")
            return
    
    # 读取数据
    try:
        df = pd.read_csv(data_path)
        # 确保有日期和收盘价列
        if 'trade_date' not in df.columns or 'close' not in df.columns:
            print("Data format error: Missing required columns 'trade_date' or 'close'")
            return
            
        # 转换日期格式
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        df.sort_values('trade_date', inplace=True)
        
        # 绘制图表
        plt.figure(figsize=(12, 6))
        plt.plot(df['trade_date'], df['close'], label='Closing Price')
        
        # 添加标题和标签
        if font_path:
            plt.title(f'{args.ts_code} 股票价格走势')
            plt.xlabel('日期')
            plt.ylabel('价格')
        else:
            plt.title(f'{args.ts_code} Stock Price Trend')
            plt.xlabel('Date')
            plt.ylabel('Price')
            
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # 创建可视化图像目录
        img_dir = Path(__file__).parent.parent / 'data' / 'visualImage'
        img_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成图像文件名
        img_filename = f"{args.ts_code}_{start_str}_{end_str}.png"
        img_path = img_dir / img_filename
        
        # 保存图表
        plt.savefig(img_path, dpi=300)
        print(f"Chart saved to: {img_path}")
        
    except Exception as e:
        print(f"Data processing failed: {e}")

if __name__ == "__main__":
    main()
