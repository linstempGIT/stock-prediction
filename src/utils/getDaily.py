import argparse
from src.utils.tsCodeUtils import normalize_ts_code
import tushare as ts
import os
from pathlib import Path
import pandas as pd
from src.utils.loadConfig import load_config  # 导入配置加载模块
from typing import Optional, Tuple

def get_daily_data(ts_code: str, token: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    获取股票历史日线行情数据
    :param ts_code: 股票代码（必填）
    :param token: Tushare API Token
    :param start_date: 开始日期(YYYYMMDD)
    :param end_date: 结束日期(YYYYMMDD)
    :return: DataFrame 包含日线数据，失败时返回 None
    """
    try:
        ts.set_token(token)
        pro = ts.pro_api()
        df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        return df
    except Exception as e:
        print(f"获取数据失败: {e}")
        return None

def save_data(df: pd.DataFrame, ts_code: str, start_date: Optional[str], end_date: Optional[str], skip_confirm: bool = False) -> Optional[Path]:
    """
    保存数据到CSV文件
    :param df: 包含数据的DataFrame
    :param ts_code: 股票代码
    :param start_date: 开始日期
    :param end_date: 结束日期
    :param skip_confirm: 是否跳过文件覆盖确认
    :return: 保存的文件路径，失败时返回 None
    """
    # 创建输出目录（数据目录已移动到项目根目录）
    output_dir = Path(__file__).parent.parent.parent / 'data' / 'dailyInfo'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 确保股票代码是大写格式
    normalized_ts_code = normalize_ts_code(ts_code)
    
    # 生成文件名
    date_range = f"{start_date or 'all'}_{end_date or 'now'}"
    filename = f"{normalized_ts_code}_{date_range}.csv"
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

def fetch_and_save_daily_data(
    ts_code: str, 
    token: str, 
    start_date: Optional[str] = None, 
    end_date: Optional[str] = None, 
    skip_confirm: bool = False
) -> Tuple[bool, Optional[Path]]:
    """
    核心函数：获取并保存股票日线数据
    
    :param ts_code: 股票代码
    :param token: Tushare API Token
    :param start_date: 开始日期(YYYYMMDD)
    :param end_date: 结束日期(YYYYMMDD)
    :param skip_confirm: 是否跳过文件覆盖确认
    :return: (成功状态, 文件路径) 元组
    """
    # 检查token是否有效
    if not token:
        print("⚠️ 未提供有效的Tushare Token，无法获取数据")
        return False, None
    
    data = get_daily_data(ts_code, token, start_date, end_date)
    if data is not None:
        saved_path = save_data(data, ts_code, start_date, end_date, skip_confirm)
        if saved_path:
            return True, saved_path
    return False, None

def main():
    """命令行接口主函数"""
    parser = argparse.ArgumentParser(description='获取股票历史日线行情数据')
    parser.add_argument('-c', '--ts_code', required=True, help='股票代码（必填）')
    parser.add_argument('--start_date', help='开始日期(YYYYMMDD)')
    parser.add_argument('--end_date', help='结束日期(YYYYMMDD)')
    parser.add_argument('-t', '--token', help='Tushare API Token（优先级高于环境变量）')
    parser.add_argument('-s', '--skip-confirm', action='store_true', help='跳过文件覆盖确认提示')
    
    args = parser.parse_args()
    
    # 加载配置
    load_config()
    
    # 获取token（优先命令行参数，其次环境变量）
    token = args.token or os.getenv('TUSHARE_TOKEN')
    
    # 打印token来源信息
    if args.token:
        print(f"✅ 使用命令行提供的 Tushare Token")
    elif token:
        print(f"✅ TUSHARE_TOKEN 已从环境变量加载")
    else:
        print("⚠️ 未提供 Tushare Token，无法获取数据")
        return
    
    # 调用核心业务逻辑
    success, saved_path = fetch_and_save_daily_data(
        ts_code=args.ts_code,
        token=token,
        start_date=args.start_date,
        end_date=args.end_date,
        skip_confirm=args.skip_confirm
    )
    
    if success and saved_path:
        print(f"数据文件已成功创建: {saved_path}")
    else:
        print("数据获取或保存失败")

if __name__ == "__main__":
    main()
