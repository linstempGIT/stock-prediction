from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

def create_model():
    """
    创建股票预测模型
    
    返回:
        model: 初始化的MLP模型
        scaler: 初始化的标准化器
    """
    # 创建标准化器
    scaler = StandardScaler()
    
    # 创建MLP分类器
    model = MLPClassifier(
        hidden_layer_sizes=(2048, 512, 128, 32),
        activation='relu',
        solver='adam',
        max_iter=6000,
        random_state=42,
        early_stopping=True
    )
    
    return model, scaler
