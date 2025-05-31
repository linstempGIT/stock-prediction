import unittest
from src.utils.tsCodeUtils import normalize_ts_code, is_valid_ts_code

class TestTsCodeUtils(unittest.TestCase):
    
    def test_normalize_ts_code(self):
        """测试标准化股票代码格式"""
        # 测试已标准化的代码
        self.assertEqual(normalize_ts_code('000001.SZ'), '000001.SZ')
        
        # 测试小写后缀
        self.assertEqual(normalize_ts_code('000001.sz'), '000001.SZ')
        self.assertEqual(normalize_ts_code('600000.sh'), '600000.SH')
        
        # 测试混合大小写
        self.assertEqual(normalize_ts_code('300001.Sz'), '300001.SZ')
        
        # 测试没有点号的情况
        self.assertEqual(normalize_ts_code('000001SZ'), '000001SZ')
        self.assertEqual(normalize_ts_code('abc123'), 'ABC123')
        
        # 测试无效格式
        self.assertEqual(normalize_ts_code('12345.123'), '12345.123')
    
    def test_is_valid_ts_code(self):
        """测试验证股票代码格式"""
        # 测试有效格式
        self.assertTrue(is_valid_ts_code('000001.SZ'))
        self.assertTrue(is_valid_ts_code('600000.SH'))
        self.assertTrue(is_valid_ts_code('300001.SZ'))
        
        # 测试无效格式 - 缺少点号
        self.assertFalse(is_valid_ts_code('000001SZ'))
        self.assertFalse(is_valid_ts_code('600000SH'))
        
        # 测试无效格式 - 点号前长度不对
        self.assertFalse(is_valid_ts_code('00001.SZ'))   # 5位
        self.assertFalse(is_valid_ts_code('0000001.SZ')) # 7位
        
        # 测试无效格式 - 点号后长度不对
        self.assertFalse(is_valid_ts_code('000001.S'))   # 1位
        self.assertFalse(is_valid_ts_code('000001.SZZ')) # 3位
        
        # 测试无效格式 - 多个点号
        self.assertFalse(is_valid_ts_code('000.001.SZ'))
        
        # 测试非字符串输入
        self.assertFalse(is_valid_ts_code(None))
        self.assertFalse(is_valid_ts_code(123456))
        self.assertFalse(is_valid_ts_code(['000001', 'SZ']))

if __name__ == '__main__':
    unittest.main()
