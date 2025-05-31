# src/train/trainLSTMAttention.py
import argparse
import logging
import sys
import os
from pathlib import Path # For easier path manipulation
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import joblib
from torch.utils.data import DataLoader, TensorDataset

# It's good practice to ensure the model definition is available
# If LSTMAttentionModel is in a sibling directory's module:
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.LSTMAttentionModel import create_model # Assuming create_model is in LSTMAttentionModel.py
from utils.tsCodeUtils import normalize_ts_code, is_valid_ts_code

# --- CPU Thread Settings (moved to top for clarity) ---
try:
    cpu_num = os.cpu_count() # Use all available CPUs by default or a reasonable number
    if cpu_num is None or cpu_num < 1:
        cpu_num = 4 # Fallback if os.cpu_count() fails or returns an invalid number
    elif cpu_num > 8: # Cap at 8 for general use unless specific needs arise
        cpu_num = 8

    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)
    # print(f"PyTorch using {torch.get_num_threads()} threads.") # Optional: for verification
except Exception as e:
    print(f"Warning: Could not set CPU thread counts: {e}")
# --- End CPU Thread Settings ---

def setup_logger():
    """设置训练日志记录器"""
    logger = logging.getLogger('trainLSTMAttention')
    logger.setLevel(logging.INFO)
    if not logger.hasHandlers():
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

def load_processed_data(ts_code, start_date_str, end_date_str, seq_length, data_type, processed_data_dir="data/processed/lstm"):
    """
    加载处理后的数据文件.
    Filename format: <ts_code>_<start_date>_<end_date>_seq<XX>_<data_type>.<ext>
    """
    logger = logging.getLogger('trainLSTMAttention')
    base_dir = Path(processed_data_dir) # Use Path object
    
    # Construct filename based on CommonDataProcessor's saving convention
    filename_stem = f"{ts_code}_{start_date_str}_{end_date_str}_seq{seq_length}_{data_type}"
    
    if data_type == 'scaler':
        filename = f"{filename_stem}.joblib"
    else: # For 'X' and 'y' data
        filename = f"{filename_stem}.npy"
        
    filepath = base_dir / filename
    
    if not filepath.exists():
        logger.error(f"Processed data file not found: {filepath}")
        return None
    
    try:
        if data_type == 'scaler':
            return joblib.load(filepath)
        else:
            return np.load(filepath, allow_pickle=True)
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        return None


def get_artifact_save_path(save_dir: Path, model_identifier: str, ts_code: str, params_suffix: str, artifact_type: str, extension: str):
    """
    获取模型或Scaler等产物的保存路径.
    Filename format: <model_identifier>_<ts_code>_<params_suffix>_```html .<extension>
    Example: lstm_attention_000001.SZ_seq60_model.pth
             lstm_attention_000001.SZ_seq60_scaler.joblib
    """
    save_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists
    filename = f"{model_identifier}_{ts_code}_{params_suffix}_{artifact_type}.{extension}"
    return save_dir / filename

def main():
    logger = setup_logger()
    
    parser = argparse.ArgumentParser(description='LSTM+Attention模型训练')
    parser.add_argument('-c', '--ts_code', required=True, help='股票代码（如：000001.SZ）')
    parser.add_argument('--model', required=True, help='模型标识符 (例如: lstm_attention_v1)')
    parser.add_argument('--start_date', default='all', help='数据开始日期 (YYYYMMDD), 默认 "all"')
    parser.add_argument('--end_date', default='now', help='数据结束日期 (YYYYMMDD), 默认 "now"')
    parser.add_argument('--seq_length', type=int, default=60, help='时间序列长度（默认60）')
    parser.add_argument('--batch_size', type=int, default=32, help='批处理大小（默认32）')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数（默认100）')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率（默认0.001）')
    parser.add_argument('--save_dir', default='data/models', help='模型和Scaler保存目录（默认 data/models）')
    parser.add_argument('--processed_data_dir', default='data/processed/lstm', help='预处理数据加载目录 (默认 data/processed/lstm)')
    
    args = parser.parse_args()
    
    normalized_ts_code = normalize_ts_code(args.ts_code)
    if not is_valid_ts_code(normalized_ts_code):
        logger.error(f"股票代码格式错误: {args.ts_code}，应为'XXXXXX.XX'格式")
        sys.exit(1)
    args.ts_code = normalized_ts_code

    # Convert save_dir to Path object
    save_directory = Path(args.save_dir)

    try:
        logger.info("加载预处理数据...")
        # Use start_date and end_date from args for loading
        X = load_processed_data(args.ts_code, args.start_date, args.end_date, args.seq_length, 'X', args.processed_data_dir)
        y = load_processed_data(args.ts_code, args.start_date, args.end_date, args.seq_length, 'y', args.processed_data_dir)
        # The scaler loaded here is the one used for PREPROCESSING the data (fit on training data by LSTMDataProcess)
        # This scaler will be re-saved alongside the TRAINED model for prediction consistency.
        preprocessing_scaler = load_processed_data(args.ts_code, args.start_date, args.end_date, args.seq_length, 'scaler', args.processed_data_dir)
        
        if X is None:
            logger.error(f"X数据 (ts_code={args.ts_code}, start={args.start_date}, end={args.end_date}, seq={args.seq_length}, type=X) 加载失败，请确保已运行数据处理脚本。")
            sys.exit(1)
        if y is None:
            logger.error(f"y数据 (ts_code={args.ts_code}, start={args.start_date}, end={args.end_date}, seq={args.seq_length}, type=y) 加载失败，请确保已运行数据处理脚本。")
            sys.exit(1)
        if preprocessing_scaler is None:
            logger.error(f"Scaler数据 (ts_code={args.ts_code}, start={args.start_date}, end={args.end_date}, seq={args.seq_length}, type=scaler) 加载失败，请确保已运行数据处理脚本。")
            sys.exit(1)
            
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1) # Target needs to be [batch_size, 1] for MSELoss
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=min(cpu_num, 4), pin_memory=True if torch.cuda.is_available() else False) # Add num_workers and pin_memory
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create model (input_size comes from the data)
        # The LSTMAttentionModel doesn't have an associated scaler in its create_model like scikit-learn models
        # The scaler used is for data preprocessing before it hits the PyTorch model
        pytorch_model = create_model(
            input_size=X.shape[2], # Number of features per time step
            hidden_size=64,        # Example, can be parameterized
            num_layers=2           # Example, can be parameterized
        ).to(device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(pytorch_model.parameters(), lr=args.lr)
        
        logger.info(f"开始训练模型 '{args.model}'（设备: {device}）...")
        logger.info(f"数据来自: ts_code={args.ts_code}, start_date={args.start_date}, end_date={args.end_date}, seq_length={args.seq_length}")
        logger.info(f"训练参数: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")

        for epoch in range(args.epochs):
            pytorch_model.train()
            total_loss = 0
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                outputs = pytorch_model(batch_x)
                loss = criterion(outputs, batch_y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == args.epochs -1 : # Log every 10 epochs + first and last
                logger.info(f"轮次 [{epoch+1}/{args.epochs}], 损失: {avg_loss:.6f}")
        
        # Define suffix for saved artifact filenames
        params_suffix_for_saving = f'seq{args.seq_length}' # Could include other params like epochs if needed

        # Save PyTorch model state_dict
        model_save_path = get_artifact_save_path(
            save_directory, args.model, args.ts_code, params_suffix_for_saving, 'model', 'pth'
        )
        torch.save(pytorch_model.state_dict(), model_save_path)
        logger.info(f"PyTorch模型已保存至: {model_save_path}")
        
        # Save the PREPROCESSING scaler that was used for this data
        # This scaler is essential for making predictions on new raw data
        scaler_save_path = get_artifact_save_path(
            save_directory, args.model, args.ts_code, params_suffix_for_saving, 'scaler', 'joblib'
        )
        joblib.dump(preprocessing_scaler, scaler_save_path)
        logger.info(f"预处理标准化器已保存至: {scaler_save_path}")
        
    except Exception as e:
        logger.error(f"训练失败: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
