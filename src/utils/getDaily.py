import argparse
import tushare as ts
import os
from pathlib import Path
import pandas as pd

def get_daily_data(ts_code, token, start_date=None, end_date=None):
    """
    获取股票历史日线行情数据
    :param ts_code: 股票代码（必填）
    :param token: Tushare API Token
    :param start_date: 开始日期(YYYYMMDD)
    :param end_date: 结束日期(YYYYMMDD)
    :return: DataFrame 包含日线数据
    """
    try:
        ts.set_token(token)
        pro = ts.pro_api()
        df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        return df
    except Exception as e:
        print(f"获取数据失败: {e}")
        return None

def save_data(df, ts_code, start_date, end_date, skip_confirm=False):
    """
    保存数据到CSV文件
    :param df: 包含数据的DataFrame
    :param ts_code: 股票代码
    :param start_date: 开始日期
    :param end_date: 结束日期
    :param skip_confirm: 是否跳过文件覆盖确认
    """
    # 创建输出目录
    output_dir = Path(__file__).parent.parent / 'data' / 'dailyInfo'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成文件名
    date_range = f"{start_date or 'all'}_{end_date or 'now'}"
    filename = f"{ts_code}_{date_range}.csv"
    output_path = output_dir / filename
    
    # 检查文件是否存在并确认覆盖
    if output_path.exists() and not skip_confirm:
        try:
            confirm = input(f"文件 {output_path} 已存在，是否覆盖？[y/N]: ").strip().lower()
            if confirm not in ('y', 'yes'):
                print("操作已取消")
                return None
        except EOFError:
            print("检测到非交互式环境，使用 -s 参数跳过确认")
            return None
    
    # 保存数据
    try:
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"数据已保存至: {output_path}")
        return output_path
    except Exception as e:
        print(f"保存数据失败: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='获取股票历史日线行情数据')
    parser.add_argument('-c', '--ts_code', required=True, help='股票代码（必填）')
    parser.add_argument('--start_date', help='开始日期(YYYYMMDD)')
    parser.add_argument('--end_date', help='结束日期(YYYYMMDD)')
    parser.add_argument('-t', '--token', help='Tushare API Token（优先级高于环境变量）')
    parser.add_argument('-s', '--skip-confirm', action='store_true', help='跳过文件覆盖确认提示')
    
    args = parser.parse_args()
    
    # 获取token（优先命令行参数，其次环境变量）
    token = args.token or os.getenv('TUSHARE_TOKEN')
    if not token:
        print("错误: 未提供Tushare API Token")
        print("请通过 -t 参数提供token或设置 TUSHARE_TOKEN 环境变量")
        exit(1)
    
    data = get_daily_data(args.ts_code, token, args.start_date, args.end_date)
    if data is not None:
        saved_path = save_data(data, args.ts_code, args.start_date, args.end_date, args.skip_confirm)
        if saved_path:
            print(f"数据文件已成功创建: {saved_path}")
        else:
            print("数据保存失败")
    else:
        print("获取数据失败，没有数据可保存")
