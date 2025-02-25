import json                           # 导入 json 模块，用于处理 JSON 格式的数据
import pandas as pd                   # 导入 pandas 模块，用于数据处理与分析
import numpy as np                    # 导入 numpy 模块，用于数值计算
import keras                          # 导入 keras 模块，用于深度学习模型构建与训练
from tensorflow.keras.models import load_model  # 从 tensorflow.keras.models 中导入 load_model 函数，用于加载保存的模型
import time                           # 导入 time 模块，用于计时

'''
    此步骤将300维的语义向量降维到15维数据。
    由于原始数据集过于庞大，这里选择了2000个样本进行展示。
'''

# 打开 JSON 文件，读取语义向量数据并进行降维处理
with open('./data/hdfs_semantic_vec.json') as f:       # 以只读方式打开包含语义向量的 JSON 文件
    gdp_list = json.load(f)                              # 加载 JSON 文件内容，将其转换为 Python 字典
    value = list(gdp_list.values())                      # 获取字典中所有的值（语义向量），并转换为列表
    from sklearn.decomposition import PCA              # 从 scikit-learn 库中导入 PCA 模块，用于主成分分析降维
    estimator = PCA(n_components=15)                     # 创建 PCA 实例，将目标降维后的维数设为15
    pca_result = estimator.fit_transform(value)          # 对语义向量进行 PCA 降维，并返回降维后的结果
    ppa_result = []                                      # 初始化一个空列表，用于存储后续处理后的向量结果
    result = pca_result - np.mean(pca_result)            # 对 PCA 结果进行均值中心化（每个向量减去整体均值）
    pca = PCA(n_components=15)                           # 再次创建一个 PCA 实例（此处可视为对中心化数据的再处理）
    pca_result = pca.fit_transform(result)               # 对中心化后的数据再次进行 PCA 降维
    U = pca.components_                                  # 获取 PCA 模型中的主成分（特征向量），存储在 U 中
    for i, x in enumerate(result):                       # 遍历中心化后的每一个向量（索引 i 和向量 x）
        for u in U[0:7]:                                 # 对前7个主成分进行处理（可能用来消除某些主成分的影响）
            x = x - np.dot(u.transpose(), x) * u         # 将向量 x 在主成分 u 上的投影分量移除
        ppa_result.append(list(x))                       # 将处理后的向量转换为列表后追加到 ppa_result 中
    ppa_result = np.array(ppa_result)                    # 将存储结果的列表转换为 numpy 数组，便于后续数值运算

# 定义用于读取测试数据的函数
def read_test(path):                                   # 定义函数 read_test，参数 path 为测试数据文件路径
    logs_series = pd.read_csv(path)                    # 使用 pandas 读取 CSV 文件，返回 DataFrame 对象
    logs_series = logs_series.values                  # 将 DataFrame 转换为 numpy 数组
    label = logs_series[:,1]                           # 提取数组中第二列作为标签数据
    logs_data = logs_series[:,0]                       # 提取数组中第一列作为日志数据（字符串形式的数字序列）
    logs = []                                          # 初始化空列表，用于存储处理后的日志数据
    for i in range(0, len(logs_data)):                 # 遍历每一条日志数据
        padding = np.zeros((300, 15))                  # 初始化一个 300x15 的全零矩阵，用作日志数据的填充（300个时间步，每个步长15维）
        data = logs_data[i]                            # 获取当前日志数据（字符串格式）
        data = [int(n) for n in data.split()]          # 将字符串按空格分割，并将每个分割出的数字转换为整数，形成列表
        for j in range(0, len(data)):                  # 遍历该条日志中的每个数字（索引 j 对应时间步）
            padding[j] = ppa_result[data[j]]           # 利用数字作为索引，从 ppa_result 中取出对应的向量，填充到 padding 的第 j 行
        padding = list(padding)                        # 将 numpy 数组转换为列表（可选步骤，主要为了格式一致）
        logs.append(padding)                           # 将处理好的 padding 数据追加到 logs 列表中
    logs = np.array(logs)                              # 将所有日志数据的列表转换为 numpy 数组
    train_x = logs                                   # 将处理后的日志数据赋值给变量 train_x 作为输入特征
    train_y = label                                  # 将提取的标签数据赋值给变量 train_y
    text_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 15))  # 将输入特征调整为 (样本数, 300, 15) 的形状
    text_y = keras.utils.to_categorical(np.array(train_y))  # 将标签转换为 one-hot 编码格式
    return text_x, text_y, label                     # 返回处理好的测试数据、one-hot 标签和原始标签

# 定义测试模型的函数
def test_model(test_x):                              # 定义函数 test_model，参数 test_x 为测试数据的特征
    model = load_model('./model/E_TCN_GAP_dim15.h5')       # 加载预先训练好的模型文件
    y_pred = model.predict(test_x, batch_size=512)   # 使用模型对测试数据进行预测，设置批处理大小为 512
    return y_pred                                    # 返回模型预测的结果

# 数据处理阶段
start_data = time.perf_counter()                           # 记录数据处理开始时间（注意：time.clock() 在新版 Python 中已被弃用，可使用 time.perf_counter()）
test_path = './data/log_test_2000.csv'               # 指定测试数据 CSV 文件的路径
test_x, test_y, label = read_test(test_path)         # 调用 read_test 函数读取并处理测试数据
end_data = time.perf_counter()                               # 记录数据处理结束时间
print('The data processing time is', end_data - start_data)  # 打印数据处理所耗费的时间

# 检测（预测）阶段
start_detect = time.perf_counter()                          # 记录检测（模型预测）开始时间
y_pred = test_model(test_x)                          # 调用 test_model 函数，对测试数据进行预测，获得预测结果
end_detect = time.perf_counter()                           # 记录检测结束时间
print('The detection time is', end_detect - start_detect)  # 打印检测过程所耗费的时间

# 结果处理与性能指标计算
y_pred = np.argmax(y_pred, axis=1)                   # 将预测结果转换为类别标签，即取每行最大概率对应的索引
tp = 0                                             # 初始化真正例（True Positive）的计数
fp = 0                                             # 初始化假正例（False Positive）的计数
tn = 0                                             # 初始化真负例（True Negative）的计数
fn = 0                                             # 初始化假负例（False Negative）的计数

for j in range(0, len(y_pred)):                    # 遍历所有预测结果
    if label[j] == y_pred[j] and label[j] == 0:    # 如果真实标签与预测标签相同，且标签为 0，则计为真正例
        tp = tp + 1
    elif label[j] != y_pred[j] and label[j] == 0:  # 如果真实标签为 0但预测错误，则计为假正例
        fp = fp + 1
    elif label[j] == y_pred[j] and label[j] == 1:  # 如果真实标签与预测标签相同，且标签为 1，则计为真负例
        tn = tn + 1
    elif label[j] != y_pred[j] and label[j] == 1:  # 如果真实标签为 1但预测错误，则计为假负例
        fn = fn + 1

print('TP,FP,TN,FN are: ', [tp, fp, tn, fn])      # 输出混淆矩阵中的各项计数：真正例、假正例、真负例、假负例

# 计算并输出 Precision（精确率）、Recall（召回率）以及 F1-measure（F1 值）
# 注意：这里的计算可能与传统的正负类定义不同，根据代码中条件，0 被当作“异常”类别
precision = tn / (tn + fn) if (tn + fn) != 0 else 0  # 计算精确率，防止除零错误
recall = tn / (tn + fp) if (tn + fp) != 0 else 0     # 计算召回率，防止除零错误
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0  # 计算 F1-measure
print('Precision, Recall, F1-measure are:', precision, recall, f1)

# 将混淆矩阵的结果保存到 CSV 文件中
datas = pd.DataFrame(data=[tp, fp, tn, fn])         # 将计数结果转换为 pandas DataFrame
datas.to_csv('./result/result_HDFS_15', index=False, header=False)  # 将结果保存为 CSV 文件，不包含索引和表头