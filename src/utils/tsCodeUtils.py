"""
股票代码工具函数
"""

def normalize_ts_code(ts_code: str) -> str:
    """
    标准化股票代码格式：将后缀统一为大写
    
    参数:
        ts_code: 股票代码字符串，格式为"XXXXXX.XX"
        
    返回:
        统一后缀大写后的股票代码
    """
    if '.' not in ts_code:
        return ts_code.upper()
    
    code_part, exchange_part = ts_code.split('.', 1)
    return f"{code_part}.{exchange_part.upper()}"

def is_valid_ts_code(ts_code: str) -> bool:
    """
    验证股票代码格式是否符合"XXXXXX.XX"格式
    
    参数:
        ts_code: 待验证的股票代码
        
    返回:
        bool: 是否有效
    """
    if not isinstance(ts_code, str):
        return False
    
    parts = ts_code.split('.')
    if len(parts) != 2:
        return False
        
    return len(parts[0]) == 6 and len(parts[1]) == 2
