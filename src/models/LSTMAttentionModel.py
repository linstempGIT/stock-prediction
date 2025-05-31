import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMAttentionModel(nn.Module):
    """
    LSTM+Attention模型用于股票价格预测
    
    参数:
        input_size: 输入特征维度 (默认5: open/high/low/close/vol)
        hidden_size: LSTM隐藏层维度 (默认64)
        num_layers: LSTM层数 (默认2)
        output_size: 输出维度 (默认1: 收盘价)
    """
    
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMAttentionModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # 全连接输出层
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量 (batch_size, seq_len=60, input_size=5)
        返回:
            output: 预测值 (batch_size, 1)
        """
        # LSTM输出: (batch_size, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)
        
        # 注意力权重: (batch_size, seq_len, 1)
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        
        # 上下文向量: (batch_size, hidden_size)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # 预测输出: (batch_size, 1)
        output = self.fc(context)
        return output

def create_model(input_size=5, hidden_size=64, num_layers=2, output_size=1):
    """创建LSTM+Attention模型实例"""
    return LSTMAttentionModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size
    )
