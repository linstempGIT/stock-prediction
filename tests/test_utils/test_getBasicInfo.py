import unittest
from unittest.mock import patch, MagicMock, call
from pathlib import Path
import pandas as pd
import os
from src.utils.getBasicInfo import fetch_and_save_companies_info

class TestGetCompaniesInfo(unittest.TestCase):
    
    def setUp(self):
        # 创建测试输出目录
        self.test_output = Path(__file__).parent / 'test_output'
        self.test_output.mkdir(exist_ok=True)
        self.output_path = self.test_output / 'companies_info.csv'
        
        # 如果测试文件已存在则删除
        if self.output_path.exists():
            self.output_path.unlink()
        
        # 模拟股票信息数据
        self.mock_stock_info = pd.DataFrame({
            'code': ['000001', '600000'],
            'name': ['平安银行', '浦发银行']
        })
        
        # 模拟 Tushare 返回数据
        self.mock_tushare_data = pd.DataFrame({
            'ts_code': ['000001.SZ', '600000.SH'],
            'symbol': ['000001', '600000'],
            'name': ['平安银行股份有限公司', '上海浦东发展银行股份有限公司'],
            'area': ['深圳', '上海'],
            'industry': ['银行', '银行'],
            'market': ['主板', '主板'],
            'list_date': ['19910403', '19991110'],
            'list_status': ['L', 'L']
        })
    
    def tearDown(self):
        # 清理测试文件
        if self.output_path.exists():
            self.output_path.unlink()
        
        # 清理测试目录
        try:
            self.test_output.rmdir()
        except OSError:
            pass

    @patch('src.utils.getBasicInfo.ak.stock_info_a_code_name')
    @patch('src.utils.getBasicInfo.load_config')
    @patch('src.utils.getBasicInfo.ts.pro_api')
    def test_with_valid_token(self, mock_pro_api, mock_load_config, mock_ak):
        """测试带有效 Tushare token 的情况"""
        # 设置模拟返回值
        mock_ak.return_value = self.mock_stock_info
        mock_pro = MagicMock()
        mock_pro.stock_basic.return_value = self.mock_tushare_data
        mock_pro_api.return_value = mock_pro
        
        # 调用函数
        result = fetch_and_save_companies_info(
            tushare_token='valid_token',
            skip_confirm=True,
            output_path=self.output_path
        )
        
        # 验证结果
        self.assertEqual(result, self.output_path)
        self.assertTrue(self.output_path.exists())
        
        # 验证文件内容
        df = pd.read_csv(self.output_path, dtype={'股票代码': str})
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]['股票代码'], '000001')
        self.assertEqual(df.iloc[0]['公司简称'], '平安银行')
        self.assertEqual(df.iloc[0]['公司全称'], '平安银行股份有限公司')
        
        # 验证 Tushare API 被调用
        mock_pro_api.assert_called_once()
        mock_pro.stock_basic.assert_called_once_with(
            exchange='', 
            fields='ts_code,symbol,name,area,industry,market,list_date,list_status'
        )

    @patch('src.utils.getBasicInfo.ak.stock_info_a_code_name')
    @patch('src.utils.getBasicInfo.load_config')
    @patch('src.utils.getBasicInfo.ts.pro_api')
    def test_without_token(self, mock_pro_api, mock_load_config, mock_ak):
        """测试不带 Tushare token 的情况"""
        # 设置模拟返回值
        mock_ak.return_value = self.mock_stock_info
        
        # 调用函数
        result = fetch_and_save_companies_info(
            tushare_token='',
            skip_confirm=True,
            output_path=self.output_path
        )
        
        # 验证结果
        self.assertEqual(result, self.output_path)
        self.assertTrue(self.output_path.exists())
        
        # 验证文件内容
        df = pd.read_csv(self.output_path, dtype={'股票代码': str})
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]['股票代码'], '000001')
        self.assertEqual(df.iloc[0]['公司简称'], '平安银行')
        self.assertTrue(pd.isna(df.iloc[0]['公司全称']) or df.iloc[0]['公司全称'] == '')
        
        # 验证 Tushare API 未被调用
        mock_pro_api.assert_not_called()

    @patch('builtins.input', return_value='y')
    @patch('src.utils.getBasicInfo.ak.stock_info_a_code_name')
    @patch('src.utils.getBasicInfo.load_config')
    def test_file_overwrite_confirm_yes(self, mock_load_config, mock_ak, mock_input):
        """测试文件覆盖确认（用户选择是）"""
        # 创建测试文件
        self.output_path.touch()
        
        # 设置模拟返回值
        mock_ak.return_value = self.mock_stock_info
        
        # 调用函数
        result = fetch_and_save_companies_info(
            tushare_token='',
            skip_confirm=False,
            output_path=self.output_path
        )
        
        # 验证结果
        self.assertEqual(result, self.output_path)
        self.assertTrue(self.output_path.exists())
        mock_input.assert_called_once()

    @patch('builtins.input', return_value='n')
    @patch('src.utils.getBasicInfo.ak.stock_info_a_code_name')
    @patch('src.utils.getBasicInfo.load_config')
    def test_file_overwrite_confirm_no(self, mock_load_config, mock_ak, mock_input):
        """测试文件覆盖确认（用户选择否）"""
        # 创建测试文件
        self.output_path.touch()
        
        # 设置模拟返回值
        mock_ak.return_value = self.mock_stock_info
        
        # 调用函数
        result = fetch_and_save_companies_info(
            tushare_token='',
            skip_confirm=False,
            output_path=self.output_path
        )
        
        # 验证结果
        self.assertIsNone(result)
        self.assertTrue(self.output_path.exists())
        mock_input.assert_called_once()

    @patch('src.utils.getBasicInfo.ak.stock_info_a_code_name')
    @patch('src.utils.getBasicInfo.load_config')
    @patch('src.utils.getBasicInfo.ts.pro_api')
    def test_tushare_error_fallback(self, mock_pro_api, mock_load_config, mock_ak):
        """测试 Tushare API 出错时回退到基本信息"""
        # 设置模拟返回值
        mock_ak.return_value = self.mock_stock_info
        mock_pro = MagicMock()
        mock_pro.stock_basic.side_effect = Exception("API error")
        mock_pro_api.return_value = mock_pro
        
        # 调用函数
        result = fetch_and_save_companies_info(
            tushare_token='valid_token',
            skip_confirm=True,
            output_path=self.output_path
        )
        
        # 验证结果
        self.assertEqual(result, self.output_path)
        self.assertTrue(self.output_path.exists())
        
        # 验证文件内容
        df = pd.read_csv(self.output_path, dtype={'股票代码': str})
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]['股票代码'], '000001')
        self.assertEqual(df.iloc[0]['公司简称'], '平安银行')
        self.assertTrue(pd.isna(df.iloc[0]['公司全称']) or df.iloc[0]['公司全称'] == '')

    @patch('src.utils.getBasicInfo.ak.stock_info_a_code_name')
    @patch('src.utils.getBasicInfo.load_config')
    def test_auto_create_directory(self, mock_load_config, mock_ak):
        """测试自动创建目录"""
        # 设置输出路径到不存在的目录
        new_output_path = self.test_output / 'new_dir' / 'companies_info.csv'
        
        # 设置模拟返回值
        mock_ak.return_value = self.mock_stock_info
        
        # 调用函数
        result = fetch_and_save_companies_info(
            tushare_token='',
            skip_confirm=True,
            output_path=new_output_path
        )
        
        # 验证结果
        self.assertEqual(result, new_output_path)
        self.assertTrue(new_output_path.exists())
        self.assertTrue(new_output_path.parent.exists())

    @patch('src.utils.getBasicInfo.ak.stock_info_a_code_name')
    @patch('src.utils.getBasicInfo.load_config')
    @patch('src.utils.getBasicInfo.ts.pro_api')
    def test_output_columns(self, mock_pro_api, mock_load_config, mock_ak):
        """测试输出文件包含所有必要列"""
        # 设置模拟返回值
        mock_ak.return_value = self.mock_stock_info
        mock_pro = MagicMock()
        mock_pro.stock_basic.return_value = self.mock_tushare_data
        mock_pro_api.return_value = mock_pro
        
        # 调用函数
        result = fetch_and_save_companies_info(
            tushare_token='valid_token',
            skip_confirm=True,
            output_path=self.output_path
        )
        
        # 验证文件列名
        df = pd.read_csv(self.output_path)
        expected_columns = [
            'TS代码', '股票代码', '公司简称', '公司全称',
            '所在地区', '所属行业', '市场类型', '上市日期', '上市状态'
        ]
        self.assertListEqual(list(df.columns), expected_columns)

if __name__ == '__main__':
    unittest.main()
