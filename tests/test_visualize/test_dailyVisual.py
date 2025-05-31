import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt
from src.visualize.dailyVisual import main

class TestDailyVisual(unittest.TestCase):
    
    @patch('argparse.ArgumentParser.parse_args')
    def test_invalid_ts_code(self, mock_parse_args):
        """测试无效股票代码的处理"""
        # 设置模拟参数
        mock_args = MagicMock()
        mock_args.ts_code = 'invalid_code'
        mock_parse_args.return_value = mock_args
        
        # 捕获 SystemExit 异常
        with self.assertRaises(SystemExit) as cm:
            main()
        
        # 验证退出码
        self.assertEqual(cm.exception.code, 1)
    
    @patch('argparse.ArgumentParser.parse_args')
    @patch('pathlib.Path.exists', side_effect=[False, True])
    @patch('subprocess.run')
    @patch('pandas.read_csv')
    @patch('matplotlib.pyplot.savefig')
    def test_data_file_creation(
        self, mock_savefig, mock_read_csv, mock_run, mock_exists, mock_parse_args
    ):
        """测试数据文件不存在时创建文件"""
        # 设置模拟参数
        mock_args = MagicMock()
        mock_args.ts_code = '000001.SZ'
        mock_args.start_date = '20230101'
        mock_args.end_date = '20231231'
        mock_parse_args.return_value = mock_args
        
        # 模拟子进程运行成功
        mock_run.return_value = MagicMock(returncode=0)
        
        # 创建模拟DataFrame
        mock_df = pd.DataFrame({
            'trade_date': ['20230101', '20230102'],
            'close': [10.0, 10.5]
        })
        mock_read_csv.return_value = mock_df
        
        # 调用主函数
        main()
        
        # 验证子进程被调用
        mock_run.assert_called_once()
        
        # 验证保存图表被调用
        mock_savefig.assert_called_once()
    
    @patch('argparse.ArgumentParser.parse_args')
    @patch('pathlib.Path.exists', return_value=True)
    @patch('pandas.read_csv')
    @patch('matplotlib.pyplot.savefig')
    def test_valid_data_processing(
        self, mock_savefig, mock_read_csv, mock_exists, mock_parse_args
    ):
        """测试有效数据处理和图表生成"""
        # 设置模拟参数
        mock_args = MagicMock()
        mock_args.ts_code = '600000.SH'
        mock_parse_args.return_value = mock_args
        
        # 创建模拟DataFrame
        mock_df = pd.DataFrame({
            'trade_date': ['20230101', '20230102', '20230103'],
            'close': [10.0, 10.5, 10.2]
        })
        mock_read_csv.return_value = mock_df
        
        # 调用主函数
        main()
        
        # 验证读取数据被调用
        mock_read_csv.assert_called_once()
        
        # 验证保存图表被调用
        mock_savefig.assert_called_once()
        
        # 验证日期转换
        processed_df = mock_read_csv.return_value
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(processed_df['trade_date']))
    
    @patch('argparse.ArgumentParser.parse_args')
    @patch('pathlib.Path.exists', return_value=True)
    @patch('pandas.read_csv')
    @patch('matplotlib.pyplot.savefig')
    def test_missing_columns(self, mock_savefig, mock_read_csv, mock_exists, mock_parse_args):
        """测试缺少必要列的处理"""
        # 设置模拟参数
        mock_args = MagicMock()
        mock_args.ts_code = '300001.SZ'
        mock_parse_args.return_value = mock_args
        
        # 创建缺少列的DataFrame
        mock_df = pd.DataFrame({
            'date': ['20230101', '20230102'],
            'price': [10.0, 10.5]
        })
        mock_read_csv.return_value = mock_df
        
        # 调用主函数
        main()
        
        # 验证保存图表未被调用
        mock_savefig.assert_not_called()
    
    @patch('argparse.ArgumentParser.parse_args')
    @patch('pathlib.Path.exists', return_value=True)
    @patch('pandas.read_csv')
    @patch('matplotlib.pyplot.savefig')
    def test_image_save_path(self, mock_savefig, mock_read_csv, mock_exists, mock_parse_args):
        """测试图表保存路径正确性"""
        # 设置模拟参数
        mock_args = MagicMock()
        mock_args.ts_code = '000001.SZ'
        mock_args.start_date = '20230101'
        mock_args.end_date = '20231231'
        mock_parse_args.return_value = mock_args
        
        # 创建模拟DataFrame
        mock_df = pd.DataFrame({
            'trade_date': ['20230101', '20230102'],
            'close': [10.0, 10.5]
        })
        mock_read_csv.return_value = mock_df
        
        # 调用主函数
        main()
        
        # 验证保存图表被调用
        self.assertTrue(mock_savefig.called)
        
        # 验证保存路径格式
        save_path = mock_savefig.call_args[0][0]
        self.assertIn('data/visualImage', str(save_path))
        self.assertIn('000001.SZ_20230101_20231231.png', str(save_path))

if __name__ == '__main__':
    unittest.main()
