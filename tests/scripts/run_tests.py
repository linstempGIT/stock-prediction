import argparse
import subprocess
from datetime import datetime

def run_tests(log_name=None):
    # 生成默认文件名（日期时间格式）
    if not log_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        log_name = f"{timestamp}-testResult.log"
    
    log_path = f"logs/{log_name}"
    
    # 执行 pytest 并重定向输出
    command = ["python", "-m", "pytest", "-v"]
    with open(log_path, "w") as log_file:
        process = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT  # 合并 stderr 到 stdout
        )
        process.communicate()  # 等待命令完成
    
    print(f"测试结果已保存至: {log_path}")
    return process.returncode

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行测试并保存结果到日志文件")
    parser.add_argument("-n", "--name", help="指定输出文件名（默认为日期时间-testResult.log）")
    
    args = parser.parse_args()
    exit_code = run_tests(args.name)
    
    if exit_code != 0:
        print(f"测试执行失败，退出码: {exit_code}")
