# src/predict/predict_stock.py
import argparse
import logging
import sys
import os
from pathlib import Path # For easier path manipulation
import torch
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.LSTMAttentionModel import LSTMAttentionModel, create_model # Assuming create_model is for PyTorch
from utils.tsCodeUtils import normalize_ts_code, is_valid_ts_code

def setup_logger():
    logger = logging.getLogger('predictStock')
    logger.setLevel(logging.INFO)
    if not logger.hasHandlers():
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

def get_trained_artifact_path(
    models_dir: Path, 
    model_identifier: str, 
    ts_code: str, 
    params_suffix: str, 
    artifact_type: str, # 'model' or 'scaler'
    extension: str
    ):
    """
    Constructs the path to a trained model or its associated scaler.
    Filename format: <model_identifier>_<ts_code>_<params_suffix>_```html .<extension>
    Example: lstm_attention_daily_v1_000001.SZ_seq60_model.pth
    """
    filename = f"{model_identifier}_{ts_code}_{params_suffix}_{artifact_type}.{extension}"
    return models_dir / filename

def load_model_and_scaler(
    models_dir: Path, 
    model_identifier: str, 
    ts_code: str, 
    seq_length: int, 
    device: torch.device,
    # These are model creation parameters, ideally they should be saved with the model
    # or inferred, but for now, we pass them.
    model_input_size: int = 5, 
    model_hidden_size: int = 64,
    model_num_layers: int = 2
    ):
    """Loads the trained PyTorch model and its corresponding scaler."""
    logger = logging.getLogger('predictStock')
    
    params_suffix_for_loading = f"seq{seq_length}" # Matches training save convention

    model_path = get_trained_artifact_path(
        models_dir, model_identifier, ts_code, params_suffix_for_loading, 'model', 'pth'
    )
    scaler_path = get_trained_artifact_path(
        models_dir, model_identifier, ts_code, params_suffix_for_loading, 'scaler', 'joblib'
    )

    if not model_path.exists():
        logger.error(f"模型文件未找到: {model_path}")
        return None, None
    if not scaler_path.exists():
        logger.error(f"Scaler文件未找到: {scaler_path}")
        return None, None

    logger.info(f"从 {model_path} 加载模型...")
    # Ensure create_model parameters match those used during training
    # Ideally, these would be saved in a config file alongside the model
    model = create_model(
        input_size=model_input_size, # This should match X.shape[2] from training
        hidden_size=model_hidden_size, 
        num_layers=model_num_layers
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    logger.info("模型加载成功.")

    logger.info(f"从 {scaler_path} 加载Scaler...")
    scaler = joblib.load(scaler_path)
    logger.info("Scaler加载成功.")
    
    return model, scaler

def prepare_input_data(df: pd.DataFrame, seq_length: int, scaler: MinMaxScaler, features: list):
    """Prepares the last `seq_length` data points for prediction."""
    logger = logging.getLogger('predictStock')
    if len(df) < seq_length:
        logger.error(f"数据不足以创建长度为 {seq_length} 的序列。需要 {seq_length} 条记录，但只有 {len(df)} 条。")
        return None
    
    # Ensure data is sorted chronologically (oldest first, newest last)
    # The input df should already be sorted by 'trade_date' ascending by main()
    input_df_for_sequence = df.tail(seq_length) # Get the LATEST seq_length rows
    
    if len(input_df_for_sequence) != seq_length:
        logger.error(f"未能获取 {seq_length} 条记录用于序列, 实际获取: {len(input_df_for_sequence)}。请检查输入数据和排序。")
        return None
    
    sequence_features = input_df_for_sequence[features].copy()
    logger.info(f"使用最新的 {len(sequence_features)} 条数据 (features: {features}) 进行预测。")
    
    scaled_data = scaler.transform(sequence_features)
    input_tensor = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0) # Add batch dimension
    return input_tensor

def predict_next_close(model: LSTMAttentionModel, input_tensor: torch.Tensor, scaler: MinMaxScaler, device: torch.device, features_list: list, target_feature: str = 'close'):
    """Makes a prediction and inverse transforms it."""
    logger = logging.getLogger('predictStock')
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        scaled_prediction = model(input_tensor) # Output shape should be [1, 1]
    
    scaled_prediction_value = scaled_prediction.item() # Get the single predicted value
    logger.info(f"模型输出 (归一化 '{target_feature}'): {scaled_prediction_value:.6f}")
    
    # Inverse transform the predicted 'close' price
    # Create a dummy array with the same shape as the scaler expects (num_samples, num_features)
    dummy_array = np.zeros((1, scaler.n_features_in_))
    
    try:
        target_feature_index = features_list.index(target_feature)
    except ValueError:
        logger.error(f"目标特征 '{target_feature}' 不在特征列表 {features_list} 中。")
        return None
        
    try:
        # Get feature names from the scaler itself
        # These are the names it learned during the .fit() call
        scaler_feature_names = list(scaler.feature_names_in_)

        num_features = len(scaler_feature_names)
        inverse_transform_data = np.zeros((1, num_features))
        
        # Ensure target_feature_index is valid for scaler_feature_names
        # This should be true if features_list and scaler_feature_names are consistent
        if target_feature not in scaler_feature_names:
            logger.error(f"Target feature '{target_feature}' not in scaler's known features: {scaler_feature_names}. Check consistency with 'features_list'.")
            # Attempt to use the original target_feature_index if lists are just out of sync but same length
            if target_feature_index >= num_features:
                logger.error("target_feature_index is out of bounds for scaler's features.")
                return None
        else:
             # If target_feature is in scaler_feature_names, use its index from there for robustness
             target_feature_index_for_scaler = scaler_feature_names.index(target_feature)
             inverse_transform_data[0, target_feature_index_for_scaler] = scaled_prediction_value


        # Create a DataFrame with the column names the scaler was fitted with
        inverse_df = pd.DataFrame(inverse_transform_data, columns=scaler_feature_names)
        
        # scaler.inverse_transform will return a NumPy array
        unscaled_prediction_numpy_array = scaler.inverse_transform(inverse_df)
        
        # Extract the value using the same index
        predicted_price = unscaled_prediction_numpy_array[0, target_feature_index_for_scaler]

    except AttributeError:
        # Fallback if scaler doesn't have feature_names_in_ (e.g., fitted on NumPy array)
        logger.warning("Scaler does not have 'feature_names_in_'. Proceeding with NumPy array for inverse_transform. This may cause UserWarnings if it was fitted on named features.")
        dummy_array = np.zeros((1, scaler.n_features_in_)) # Use n_features_in_ if feature_names_in_ is absent
        dummy_array[0, target_feature_index] = scaled_prediction_value # Use original target_feature_index
        unscaled_prediction_full = scaler.inverse_transform(dummy_array)
        predicted_price = unscaled_prediction_full[0, target_feature_index]
    
    return predicted_price

def main():
    logger = setup_logger()
    parser = argparse.ArgumentParser(description='LSTM+Attention模型股票价格预测')
    parser.add_argument('-c', '--ts_code', required=True, help='股票代码（如：000001.SZ）')
    parser.add_argument('--model_id', required=True, help='模型标识符 (例如: lstm_attention_daily_v1)')
    parser.add_argument('--models_dir', default='data/models', help='存储训练好的模型和scaler的目录')
    parser.add_argument('--seq_length', type=int, default=60, help='模型使用的时间序列长度（默认60）')
    parser.add_argument('--input_file', required=True, help='包含最新股价数据的CSV文件路径 (至少需要seq_length条记录)')
    # Optional: If model architecture varies and isn't saved with model
    parser.add_argument('--model_input_size', type=int, default=5, help='模型输入的特征数 (默认5 for open,high,low,close,vol)')
    parser.add_argument('--model_hidden_size', type=int, default=64, help='模型隐藏层大小')
    parser.add_argument('--model_num_layers', type=int, default=2, help='模型层数')

    args = parser.parse_args()

    normalized_ts_code = normalize_ts_code(args.ts_code)
    if not is_valid_ts_code(normalized_ts_code):
        logger.error(f"股票代码格式错误: {args.ts_code}，应为'XXXXXX.XX'格式")
        sys.exit(1)
    args.ts_code = normalized_ts_code

    # Define features used for training and expected by the scaler/model
    # This MUST match the features used during training
    features_to_use = ['open', 'high', 'low', 'close', 'vol']
    target_feature_to_predict = 'close' 
    num_model_input_features = len(features_to_use) # This should match args.model_input_size if passed

    if args.model_input_size != num_model_input_features:
        logger.warning(f"警告: --model_input_size ({args.model_input_size}) 与基于 features_to_use ({num_model_input_features}) 的数量不匹配。确保模型创建参数正确。")
        # Potentially override, or error out, depending on strictness
        # For now, we'll use args.model_input_size for model creation
        # and num_model_input_features for data slicing. This could lead to issues
        # if they are not aligned with how the model was actually trained.

    models_directory = Path(args.models_dir)

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {device}")

        model, scaler = load_model_and_scaler(
            models_directory, 
            args.model_id, 
            args.ts_code, 
            args.seq_length, 
            device,
            model_input_size=args.model_input_size, # Pass to model creation
            model_hidden_size=args.model_hidden_size,
            model_num_layers=args.model_num_layers
        )
        if model is None or scaler is None:
            logger.error("模型或Scaler加载失败。")
            sys.exit(1)
        
        # Validate scaler's expected features against our defined features
        if scaler.n_features_in_ != num_model_input_features:
            logger.error(f"Scaler期望 {scaler.n_features_in_} 个特征, 但当前定义的特征列表 {features_to_use} 有 {num_model_input_features} 个。这通常表示训练和预测时的特征集不匹配。")
            sys.exit(1)

        input_data_path = Path(args.input_file)
        if not input_data_path.exists():
            logger.error(f"输入数据文件未找到: {args.input_file}")
            sys.exit(1)
        
        logger.info(f"从 {args.input_file} 加载输入数据...")
        raw_input_df = pd.read_csv(input_data_path)
        
        # Validate required columns
        missing_cols = [col for col in features_to_use if col not in raw_input_df.columns]
        if missing_cols:
            logger.error(f"输入CSV文件缺少以下列: {', '.join(missing_cols)}. 需要: {features_to_use}")
            sys.exit(1)
        if 'trade_date' not in raw_input_df.columns:
            logger.error("输入CSV文件缺少 'trade_date' 列，无法正确排序数据。")
            sys.exit(1)
        
        # Prepare data: convert trade_date, sort, select features
        raw_input_df['trade_date'] = pd.to_datetime(raw_input_df['trade_date'].astype(str), format='%Y%m%d')
        raw_input_df = raw_input_df.sort_values(by='trade_date', ascending=True) # Ensure chronological order
        
        # The prepare_input_data function will take the tail(seq_length)
        input_tensor = prepare_input_data(raw_input_df, args.seq_length, scaler, features_to_use)
        if input_tensor is None:
            sys.exit(1)
        
        predicted_price = predict_next_close(model, input_tensor, scaler, device, features_to_use, target_feature_to_predict)
        if predicted_price is None:
            logger.error("价格预测失败。")
            sys.exit(1)
            
        last_input_data_point = raw_input_df.iloc[-1]
        last_actual_close_date = last_input_data_point['trade_date'].strftime('%Y-%m-%d')
        last_actual_close_price = last_input_data_point[target_feature_to_predict]

        logger.info("----------------------------------------------------")
        logger.info(f"股票代码: {args.ts_code}")
        logger.info(f"模型ID: {args.model_id}")
        logger.info(f"基于截至 {last_actual_close_date} (实际收盘价: {last_actual_close_price:.2f}) 的数据")
        logger.info(f"预测下一交易日 '{target_feature_to_predict}' 价格: {predicted_price:.2f}")
        logger.info("----------------------------------------------------")

    except Exception as e:
        logger.error(f"预测过程中发生错误: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
