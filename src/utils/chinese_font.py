import os
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

def setup_chinese_font():
    """
    设置中文字体支持，返回字体路径（如果成功）或 None
    """
    font_path = None
    
    try:
        # 添加系统字体路径
        system_font_dirs = ['/usr/share/fonts', '/usr/local/share/fonts']
        for font_dir in system_font_dirs:
            if os.path.exists(font_dir):
                font_files = fm.findSystemFonts(fontpaths=[font_dir])
                for font_file in font_files:
                    try:
                        fm.fontManager.addfont(font_file)
                    except:
                        continue
        
        # 尝试加载Noto Sans CJK SC字体
        noto_font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
        
        if os.path.exists(noto_font_path):
            try:
                # 使用绝对路径加载字体
                font_prop = fm.FontProperties(fname=noto_font_path)
                plt.rcParams['font.family'] = font_prop.get_name()
                font_path = noto_font_path
                print(f"使用字体: {font_path}")
            except Exception as e:
                print(f"字体加载警告: {e}")
        
        # 如果Noto字体不可用，尝试其他字体
        if not font_path:
            for font_name in ['Noto Sans CJK SC', 'SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']:
                try:
                    font_path = fm.findfont(font_name)
                    if font_path:
                        plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
                        print(f"使用字体: {font_path}")
                        break
                except:
                    continue
        
        if not font_path:
            print("警告: 未找到中文字体，图表标签将使用英文")
            
        # 解决负号显示问题
        plt.rcParams['axes.unicode_minus'] = False
        
    except Exception as e:
        print(f"字体设置警告: {e}")
    
    return font_path
