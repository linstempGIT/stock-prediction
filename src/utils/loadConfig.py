import os
import yaml
from pathlib import Path

def load_config():
    """
    加载应用配置到环境变量（支持多环境）
    """
    # 配置文件路径
    config_path = Path(__file__).parent.parent.parent / 'configs' / 'app-config.yml'
    
    # 检查配置文件是否存在
    if not config_path.exists():
        print(f"⚠️ 配置文件不存在: {config_path}")
        print("请创建配置文件并添加以下内容:")
        print("config_env: dev  # 或 prod")
        print("dev:")
        print("  tushare_token: '你的开发环境Tushare令牌'")
        print("prod:")
        print("  tushare_token: '你的生产环境Tushare令牌'")
        print("完成后重新运行程序")
        return None
    
    try:
        # 加载YAML配置
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # 获取当前环境（默认为dev）
        env = config.get('config_env', 'dev')
        if env not in ['dev', 'prod']:
            print(f"⚠️ 无效的环境配置: {env}，只支持 dev 或 prod")
            return None
        
        print(f"✅ 使用 {env} 环境配置")
        
        # 获取环境特定配置
        env_config = config.get(env, {})
        
        # 加载整个环境配置到环境变量
        loaded_vars = []
        for key, value in env_config.items():
            env_var = key.upper()  # 将配置项键名转换为大写作为环境变量名
            os.environ[env_var] = str(value)
            loaded_vars.append(env_var)
            print(f"✅ {env_var} 已从 {env} 环境加载")
        
        return loaded_vars
            
    except yaml.YAMLError as e:
        print(f"⚠️ 配置文件解析错误: {e}")
        return None
    except ModuleNotFoundError:
        print("⚠️ 缺少依赖: 请先安装 PyYAML 包")
        print("安装命令: pip install PyYAML")
        return None
    except Exception as e:
        print(f"⚠️ 加载配置时出错: {e}")
        return None

# 模块导入时自动加载配置
if __name__ == "__main__":
    loaded_vars = load_config()
    if loaded_vars:
        print(f"✅ 配置加载成功！加载了 {len(loaded_vars)} 个环境变量")
        print("加载的环境变量: " + ", ".join(loaded_vars))
    else:
        print("❌ 配置加载失败，请检查错误信息")
