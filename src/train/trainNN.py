import pandas as pd
import numpy as np
import argparse
import joblib
import logging
import sys
import importlib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def setup_logger():
    """设置日志记录器"""
    logger = logging.getLogger('StockPredictionTrainer')
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

def load_data(file_path):
    """加载处理后的股票数据"""
    logger = logging.getLogger('StockPredictionTrainer')
    logger.info(f"加载数据: {file_path}")
    df = pd.read_csv(file_path)
    
    # 提取特征和标签（最后一列是标签）
    X = df.iloc[:, :-1].values  # 除最后一列外都是特征
    y = df.iloc[:, -1].values   # 最后一列是标签
    
    logger.info(f"数据集大小: {len(X)} 样本")
    logger.info(f"特征维度: {X.shape[1]}, 标签类别分布: {pd.Series(y).value_counts().to_dict()}")
    return X, y

def train_model(X_train, y_train, model_name):
    """训练预测模型"""
    logger = logging.getLogger('StockPredictionTrainer')
    logger.info("开始训练模型...")
    
    try:
        # 动态导入模型创建函数
        module_name = f"src.models.{model_name}"
        model_module = importlib.import_module(module_name)
        
        # 创建模型和标准化器
        model, scaler = model_module.create_model()
    except ImportError:
        logger.error(f"无法导入模型: {model_name}")
        raise
    except AttributeError:
        logger.error(f"模型 {model_name} 中没有 create_model 函数")
        raise
    
    # 数据标准化
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 训练模型
    model.fit(X_train_scaled, y_train)
    logger.info("模型训练完成")
    return model, scaler

def evaluate_model(model, scaler, X_test, y_test):
    """评估模型性能"""
    logger = logging.getLogger('StockPredictionTrainer')
    logger.info("评估模型性能...")
    
    # 数据标准化
    X_test_scaled = scaler.transform(X_test)
    
    # 预测
    y_pred = model.predict(X_test_scaled)
    
    # 计算指标
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    logger.info(f"测试集准确率: {accuracy:.4f}")
    logger.info("分类报告:\n" + report)
    logger.info("混淆矩阵:\n" + str(conf_matrix))
    
    return accuracy

def main():
    # 设置日志
    logger = setup_logger()
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='股票预测模型训练')
    parser.add_argument('-d', '--data_path', required=True, help='预处理后的数据文件路径')
    parser.add_argument('--model', required=True, help='模型文件名（如 predictModel30days）')
    parser.add_argument('--test_size', type=float, default=0.2, help='测试集比例（默认0.2）')
    parser.add_argument('--save_dir', default='data/models', help='模型保存目录（默认models）')
    
    args = parser.parse_args()
    
    try:
        # 加载数据
        X, y = load_data(args.data_path)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=42, stratify=y
        )
        logger.info(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
        
        # 训练模型
        model, scaler = train_model(X_train, y_train, args.model)
        
        # 评估模型
        accuracy = evaluate_model(model, scaler, X_test, y_test)
        
        # 保存模型和标准化器
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 根据模型名创建文件名
        model_filename = f"stock_prediction_model_{args.model}.pkl"
        scaler_filename = f"stock_prediction_scaler_{args.model}.pkl"
        report_filename = f"evaluation_report_{args.model}.txt"
        
        model_path = save_dir / model_filename
        scaler_path = save_dir / scaler_filename
        report_path = save_dir / report_filename
        
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        
        logger.info(f"模型已保存至: {model_path}")
        logger.info(f"标准化器已保存至: {scaler_path}")
        
        # 保存评估报告
        with open(report_path, 'w') as f:
            f.write(f"模型: {args.model}\n")
            f.write(f"测试集准确率: {accuracy:.4f}\n")
            f.write("模型参数:\n")
            f.write(f"输入层: {model.n_features_in_} 节点\n")
            f.write(f"隐藏层: {model.hidden_layer_sizes}\n")
            f.write(f"输出层: {model.n_outputs_} 节点\n")
        
        logger.info(f"评估报告已保存至: {report_path}")
        
    except Exception as e:
        logger.error(f"处理过程中发生错误: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
