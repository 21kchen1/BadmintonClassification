import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd

# 超参数
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001
HIDDEN_SIZE = 64
INPUT_SIZE = 24  # 输入特征数（与特征列对应）
OUTPUT_SIZE = 4  # 输出类别数（actionType类别数）

# 加载数据集
def load_data(csv_path):
    """读取 CSV 文件，返回 DataFrame"""
    print(f"🚀 正在加载数据文件：{csv_path}")
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ 成功读取 CSV 文件，样本数量：{df.shape[0]}")
        return df
    except Exception as e:
        print(f"❌ 读取 CSV 文件 {csv_path} 出错：{e}")
        return None

# 数据准备
def prepare_data(df, feature_columns, label_column="actionType"):
    """
    根据指定的特征列和标签列，将 DataFrame 分离为特征矩阵 X 和标签向量 y。
    """
    print("🚀 正在准备数据（分离特征和标签）...")
    if label_column not in df.columns:
        print(f"❌ 标签列 {label_column} 不存在！")
        return None, None
    try:
        X = df[feature_columns].values
        y = df[label_column].values
        print(f"✅ 特征数据维度：{X.shape}，标签数量：{len(y)}")
        # 将标签转化为整数
        label_map = {label: idx for idx, label in enumerate(np.unique(y))}
        y = np.array([label_map[label] for label in y])
        y_tensor = torch.tensor(y, dtype=torch.long)  # 分类问题标签应为 long 类型
        X_tensor = torch.tensor(X, dtype=torch.float32)
        return X_tensor, y_tensor
    except Exception as e:
        print(f"❌ 数据分离出错：{e}")
        return None, None

# 定义神经网络模型
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # 输入层到隐藏层
        self.fc2 = nn.Linear(hidden_size, output_size)  # 隐藏层到输出层
        self.relu = nn.ReLU()  # 激活函数
        self.dropout = nn.Dropout(0.4)  # Dropout层，50%的丢弃率

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)  # 添加Dropout层
        x = self.fc2(x)
        return x

# 训练函数
def train_nn(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_preds += (predicted == labels).sum().item()
        total_preds += labels.size(0)
    
    accuracy = 100 * correct_preds / total_preds
    avg_loss = running_loss / len(train_loader)
    return avg_loss, accuracy

# 评估函数
def evaluate_model(model, val_loader):
    model.eval()
    correct_preds = 0
    total_preds = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)
    
    accuracy = 100 * correct_preds / total_preds
    return accuracy

# 早停机制
def early_stopping(verify_acc, best_val_acc, patience, no_improvement):
    if verify_acc > best_val_acc:
        best_val_acc = verify_acc
        no_improvement = 0
    else:
        no_improvement += 1
        if no_improvement >= patience:
            print("🚨 验证集准确率没有提升，提前停止训练！")
            return True, best_val_acc, no_improvement
    return False, best_val_acc, no_improvement

# 主函数
if __name__ == "__main__":
    # 设置路径和文件
    TRAIN_DATA_CSV = "D:/vscode_work/badminton_classification/data/processed/processed_train_fft_normalized.csv"
    VERIFY_DATA_CSV = "D:/vscode_work/badminton_classification/data/processed/processed_verify_fft_normalized.csv"
    TEST_DATA_CSV = "D:/vscode_work/badminton_classification/data/processed/processed_test_fft_normalized.csv"
    
    # 特征列
    feature_columns = [
        "Ax_fft_mean", "Ax_fft_std", "Ax_fft_max", "Ax_dom_bin",
        "Ay_fft_mean", "Ay_fft_std", "Ay_fft_max", "Ay_dom_bin",
        "Az_fft_mean", "Az_fft_std", "Az_fft_max", "Az_dom_bin",
        "angularSpeedX_fft_mean", "angularSpeedX_fft_std", "angularSpeedX_fft_max", "angularSpeedX_dom_bin",
        "angularSpeedY_fft_mean", "angularSpeedY_fft_std", "angularSpeedY_fft_max", "angularSpeedY_dom_bin",
        "angularSpeedZ_fft_mean", "angularSpeedZ_fft_std", "angularSpeedZ_fft_max", "angularSpeedZ_dom_bin"
    ]
    
    # 加载训练、验证和测试数据
    print("🚀 加载训练集数据...")
    train_df = load_data(TRAIN_DATA_CSV)
    print("🚀 加载验证集数据...")
    verify_df = load_data(VERIFY_DATA_CSV)
    print("🚀 加载测试集数据...")
    test_df = load_data(TEST_DATA_CSV)

    if train_df is None or verify_df is None or test_df is None:
        exit(1)

    # 准备训练、验证和测试数据
    X_train, y_train = prepare_data(train_df, feature_columns)
    X_verify, y_verify = prepare_data(verify_df, feature_columns)
    X_test, y_test = prepare_data(test_df, feature_columns)

    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    verify_dataset = TensorDataset(X_verify, y_verify)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    verify_loader = DataLoader(verify_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 定义模型、损失函数和优化器
    model = SimpleNN(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 训练过程
    best_val_acc = 0
    patience = 5  # 提前停止的耐心度
    no_improvement = 0
    
    for epoch in range(EPOCHS):
        print(f"🚀 训练第 {epoch + 1} 轮...")
        train_loss, train_acc = train_nn(model, train_loader, criterion, optimizer)
        verify_acc = evaluate_model(model, verify_loader)
        print(f"✅ 训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%")
        print(f"✅ 验证准确率: {verify_acc:.2f}%")
        
        # 提前停止判断
        stop, best_val_acc, no_improvement = early_stopping(verify_acc, best_val_acc, patience, no_improvement)
        if stop:
            break
    
    # 测试集评估
    test_acc = evaluate_model(model, test_loader)
    print(f"✅ 测试集准确率: {test_acc:.2f}%")
