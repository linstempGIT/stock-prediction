import pandas as pd
from pathlib import Path
import akshare as ak
import tushare as ts
import os
from datetime import datetime

import argparse

def get_companies_info():
    """从公开数据源获取全量上市公司基本信息"""
    try:
        # 解析命令行参数
        parser = argparse.ArgumentParser(description='上市公司基本信息采集工具',
                                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('-t', '--token', 
                          help='Tushare API Token（优先级高于环境变量）')
        parser.add_argument('-s', '--skip-confirm', 
                          action='store_true',
                          help='跳过文件覆盖确认提示')
        args = parser.parse_args()
        
        # 初始化输出路径
        output_path = Path(__file__).parent.parent / 'data' / 'baseInfo' / 'companies_info.csv'
        
        # 提前检查文件存在性
        if output_path.exists():
            if not args.skip_confirm:
                try:
                    confirm = input(f"文件 {output_path} 已存在，是否覆盖？[y/N]: ").strip().lower()
                    if confirm not in ('y', 'yes'):
                        print("操作已取消")
                        return None
                except EOFError:
                    print("检测到非交互式环境，使用 -s 参数跳过确认")
                    return None
            else:
                print(f"覆盖已存在的文件: {output_path}")
                
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 获取基础数据
        stock_info_a_code_name_df = ak.stock_info_a_code_name()
        
        # 获取Tushare token（优先命令行参数）
        tushare_token = args.token or os.getenv('TUSHARE_TOKEN')
        if not tushare_token and not args.skip_confirm:
            try:
                tushare_token = input("请输入Tushare API Token（无token直接回车跳过详细信息获取）：")
            except EOFError:
                tushare_token = ''
        
        # 初始化基础列
        df = pd.DataFrame({
            '股票代码': stock_info_a_code_name_df['code'],
            '公司简称': stock_info_a_code_name_df['name']
        })
        
        # 初始化包含TS代码的默认列
        default_cols = {
            'TS代码': '',  # 新增首列
            '公司全称': '',
            '所在地区': '',
            '所属行业': '',
            '市场类型': '',
            '上市日期': ''
        }
        df = df.assign(**default_cols)
        
        # 如果有有效token则获取详细信息
        if tushare_token.strip():
            try:
                ts.set_token(tushare_token)
                pro = ts.pro_api()
                
                # 获取主板上市公司列表
                df_base = pro.stock_basic(exchange='', 
                                        fields='ts_code,symbol,name,area,industry,market,list_date,list_status')
                # 确保原始数据包含所需字段
                if 'ts_code' not in df_base.columns:
                    raise ValueError("Tushare响应数据缺少ts_code字段")
                # 处理并增强Tushare数据
                tushare_df = df_base.rename(columns={
                    'ts_code': 'TS代码',
                    'name': '公司全称',
                    'area': '所在地区',
                    'industry': '所属行业', 
                    'market': '市场类型',
                    'list_date': '上市日期',
                    'list_status': '上市状态'
                })[['TS代码', 'symbol', '公司全称', '所在地区', '所属行业', '市场类型', '上市日期', '上市状态']]
                
                # 精确合并并重组列顺序
                merged_df = pd.merge(
                    df[['股票代码', '公司简称']],
                    tushare_df,
                    left_on='股票代码',
                    right_on='symbol',
                    how='left'
                ).drop(columns=['symbol'], errors='ignore')
                
                # 确保合并后存在TS代码列
                if 'TS代码' not in merged_df.columns:
                    merged_df['TS代码'] = ''  # 回退处理
                
                # 调整列顺序，保证TS代码作为首列
                # 包含上市状态并调整列顺序
                df = merged_df[['TS代码', '股票代码', '公司简称', '公司全称', 
                              '所在地区', '所属行业', '市场类型', '上市日期', '上市状态']]
                print("成功获取详细上市公司信息")
            except Exception as e:
                print(f"Tushare接口调用失败: {str(e)} 仅保存基本信息")
        else:
            print("未提供Tushare token，仅保存股票代码和名称等基本信息")
        
        # 保存数据（覆盖模式）
        # 最终的字段定义（移除了更新时间，添加了上市状态）
        final_columns = [
            'TS代码', '股票代码', '公司简称', '公司全称',
            '所在地区', '所属行业', '市场类型', '上市日期', '上市状态'
        ]
        
        # 处理字段缺失问题并复制DataFrame避免警告
        df = df.copy()
        for col in final_columns:
            if col not in df.columns:
                df[col] = ''
        
        df = df[final_columns]
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        return output_path
        
    except Exception as e:
        print(f"数据获取失败: {str(e)}")
        return None

if __name__ == "__main__":
    path = get_companies_info()
    if path:
        print(f"上市公司基本信息已保存至：{path}")
