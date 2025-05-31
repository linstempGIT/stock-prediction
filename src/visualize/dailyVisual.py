#!/usr/bin/env python3

import os
import sys
import argparse
from src.utils.tsCodeUtils import normalize_ts_code, is_valid_ts_code
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 导入中文字体设置模块
from src.utils.chineseFont import setup_chinese_font

def main():
    # 设置中文字体支持
    font_path = setup_chinese_font()
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='股票日线数据可视化')
    parser.add_argument('-c', '--ts_code', required=True, help='股票代码（必填）')
    parser.add_argument('--start_date', help='开始日期(YYYYMMDD)')
    parser.add_argument('--end_date', help='结束日期(YYYYMMDD)')
    
    args = parser.parse_args()
    
    # 标准化和验证股票代码
    normalized_ts_code = normalize_ts_code(args.ts_code)
    if not is_valid_ts_code(normalized_ts_code):
        print(f"股票代码格式错误: {args.ts_code}，应为'XXXXXX.XX'格式")
        exit(1)
    args.ts_code = normalized_ts_code
    
    # 构建预期的文件名
    start_str = args.start_date or 'all'
    end_str = args.end_date or 'now'
    data_filename = f"{args.ts_code}_{start_str}_{end_str}.csv"
    
    # 使用相对路径访问数据目录
    project_root = Path(__file__).resolve().parent.parent.parent
    data_dir = project_root / 'data' / 'dailyInfo'
    data_path = data_dir / data_filename
    
    print(f"数据文件路径: {data_path}")
    
    # 检查数据文件是否存在
    if not data_path.exists():
        print(f"数据文件 {data_path} 不存在，正在获取数据...")
        # 构建命令调用 getDaily.py
        cmd = [
            'python', 
            str(project_root / 'src' / 'utils' / 'getDaily.py'),
            '-c', args.ts_code,
            '-s'  # 跳过文件覆盖确认
        ]
        if args.start_date:
            cmd.extend(['--start_date', args.start_date])
        if args.end_date:
            cmd.extend(['--end_date', args.end_date])
        
        # 执行命令
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)
        if result.returncode != 0:
            print(f"获取数据失败: {result.stderr}")
            return
        # 再次检查数据文件是否存在
        if not data_path.exists():
            print("获取数据后，文件仍未创建")
            return
    
    # 读取数据
    try:
        df = pd.read_csv(data_path)
        # 确保有日期和收盘价列
        if 'trade_date' not in df.columns or 'close' not in df.columns:
            print("数据格式错误: 缺少必要列 'trade_date' 或 'close'")
            return
            
        # 转换日期格式
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        df.sort_values('trade_date', inplace=True)
        
        # 绘制图表
        plt.figure(figsize=(12, 6))
        plt.plot(df['trade_date'], df['close'], label='收盘价')
        
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
        # 使用相对路径访问图像目录
        img_dir = project_root / 'data' / 'visualImage'
        print(f"图表保存目录: {img_dir}")
        img_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成图像文件名
        img_filename = f"{args.ts_code}_{start_str}_{end_str}.png"
        img_path = img_dir / img_filename
        
        # 保存图表
        plt.savefig(img_path, dpi=300)
        print(f"图表已保存至: {img_path}")
        
    except Exception as e:
        print(f"数据处理失败: {e}")

if __name__ == "__main__":
    main()
