import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from dtaidistance import dtw

# 示例数据
data = np.random.randn(100, 5)  # 100 个样本，每个样本 5 个特征
labels = np.random.randint(0, 4, 100)  # 4 个类别

# 数据预处理
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data) - seq_length - 1):
        _x = data[i:(i + seq_length)]
        _y = data[i + seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x), np.array(y)

# 特征提取
seq_length = 10  # 示例窗口长度
X, y = sliding_windows(data_scaled, seq_length)

# 划分训练集和测试集
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# 计算 DTW 距离矩阵
def compute_dtw_distances(X_train, X_test):
    distances = []
    for test_sample in X_test:
        sample_distances = []
        for train_sample in X_train:
            distance = dtw.distance(test_sample, train_sample)
            sample_distances.append(distance)
        distances.append(sample_distances)
    return np.array(distances)

# 使用 k-NN 分类
k = 3  # 示例 k 值
knn = KNeighborsClassifier(n_neighbors=k, metric='precomputed')
distances = compute_dtw_distances(X_train, X_test)
knn.fit(X_train, y_train)
y_pred = knn.predict(distances)

# 评估模型
print(classification_report(y_test, y_pred))