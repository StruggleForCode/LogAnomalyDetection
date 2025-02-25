import json  # 导入用于处理JSON文件的模块
import pandas as pd  # 导入用于数据处理的Pandas库
import numpy as np  # 导入NumPy库，用于科学计算和数组操作
import keras  # 导入Keras深度学习库
from tensorflow.keras.models import load_model  # 导入从TensorFlow Keras加载模型的功能
import time  # 导入时间模块，用于记录处理时间
import sys  # 导入sys模块，用于重定向输出
import logging  # 导入日志模块

# 设置日志记录
log_file = './log/test_pca_dim20_epoch150_output.log'  # 日志文件路径
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')

# 继承sys.stdout来重定向打印内容到日志
class LogToFile(object):
    def write(self, message):
        logging.info(message.strip())  # 将消息写入日志文件
        sys.__stdout__.write(message)  # 同时在控制台打印

    def flush(self):
        # 对于日志文件来说，flush方法其实不需要做任何事，但必须定义它
        pass

sys.stdout = LogToFile()  # 重定向标准输出到日志文件

# Step 1: 数据预处理

# 打开一个包含JSON数据的文件，文件路径为'./data/bgl_semantic_vec.json'
with open('./data/bgl_semantic_vec.json') as f:
    # Step1-1 open file: 打开文件并加载数据
    gdp_list = json.load(f)  # 从JSON文件中读取数据并加载为字典
    value = list(gdp_list.values())  # 提取字典中的所有值并转换成列表

    # Step1-2 PCA: 使用主成分分析进行数据降维，降到20维
    from sklearn.decomposition import PCA  # 导入PCA模块
    estimator = PCA(n_components=20)  # 创建PCA模型，设置降维后的维度为20
    pca_result = estimator.fit_transform(value)  # 对数据进行PCA降维

    # Step1-3 PPA: 去均值处理
    ppa_result = []  # 初始化一个空列表用于存储PPA处理后的结果
    result = pca_result - np.mean(pca_result)  # 对PCA结果进行去均值处理
    pca = PCA(n_components=20)  # 创建PCA模型（再次用于去均值后的数据降维）
    pca_result = pca.fit_transform(result)  # 对去均值后的数据进行PCA降维
    U = pca.components_  # 获取PCA降维后的主成分

    # 对每个数据点进行去平均值的操作
    for i, x in enumerate(result):
        for u in U[0:7]:  # 对前7个主成分进行处理
            x = x - np.dot(u.transpose(), x) * u  # 去除与主成分的投影部分
        ppa_result.append(list(x))  # 将去均值后的数据点添加到结果列表
    ppa_result = np.array(ppa_result)  # 将结果转换为NumPy数组

# Step 2: 读取测试数据

def read_test(split = 0.7):
    logs_data = pd.read_csv('./data/bgl_data.csv')  # 读取日志数据文件
    logs_data = logs_data.values  # 将数据转化为NumPy数组
    label = pd.read_csv('./data/bgl_label.csv')  # 读取标签数据文件
    label = label.values  # 将标签数据转化为NumPy数组
    logs = []  # 用于存储处理后的日志数据

    # 对每一条日志数据进行处理
    for i in range(0, len(logs_data)):
        padding = np.zeros((300, 20))  # 初始化一个300x20的零矩阵用于填充
        data = logs_data[i]  # 获取当前日志数据
        for j in range(0, len(data)):
            padding[j] = pca_result[int(data[j] - 1)]  # 用PCA降维后的数据填充矩阵
        padding = list(padding)  # 转换为列表
        logs.append(padding)  # 将处理后的日志添加到列表中
    logs = np.array(logs)  # 将日志数据转换为NumPy数组

    # 划分数据集，split_boundary为训练集和验证集的划分点
    split_boundary = int(logs.shape[0] * split)
    valid_x = logs[split_boundary:]  # 获取验证集的特征数据
    test_y = label[split_boundary:]  # 获取验证集的标签数据

    # 重新调整验证集数据的形状，使其适应模型输入
    valid_x = np.reshape(valid_x, (valid_x.shape[0], valid_x.shape[1], 20))
    valid_y = keras.utils.to_categorical(np.array(test_y))  # 对标签进行独热编码
    return valid_x, valid_y, test_y  # 返回验证集数据和标签

# Step 3: 加载并测试模型

def test_model(test_x):
    model = load_model('./model/E-TCN-PCA-DIM20-EPOCH100.h5')  # 加载训练好的模型
    y_pred = model.predict(test_x, batch_size=512)  # 使用模型对测试数据进行预测
    return y_pred  # 返回预测结果

# Step 4: 执行测试和时间计算

test_x, valid_y, label = read_test()  # 读取测试数据和标签
start = time.perf_counter()  # 记录开始时间
y_pred = test_model(test_x)  # 对测试数据进行预测
end = time.perf_counter()  # 记录结束时间
print('The detection time is', end - start)  # 输出检测时间

# Step 5: 评估模型性能

y_pred = np.argmax(y_pred, axis=1)  # 将预测结果转为类别标签
tp = 0  # 真阳性
fp = 0  # 假阳性
tn = 0  # 真阴性
fn = 0  # 假阴性

# 计算TP, FP, TN, FN
for j in range(0, len(y_pred)):
    if label[j] == y_pred[j] and label[j] == 0:
        tp = tp + 1
    elif label[j] != y_pred[j] and label[j] == 0:
        fp = fp + 1
    elif label[j] == y_pred[j] and label[j] == 1:
        tn = tn + 1
    elif label[j] != y_pred[j] and label[j] == 1:
        fn = fn + 1

print('E-TCN-PCA-DIM20-EPOCH150 ')
# 输出TP, FP, TN, FN值
print('TP, FP, TN, FN are: ', [tp, fp, tn, fn])
# 输出精确率、召回率和F1分数
print('Precision, Recall, F1-measure are:', tn / (tn + fn), tn / (tn + fp), 2 * (tn / (tn + fn) * (tn / (tn + fp)) / (tn / (tn + fn) + tn / (tn + fp))))

# 保存结果到CSV文件
datas = pd.DataFrame(data=[tp, fp, tn, fn])
datas.to_csv('./result/E-TCN-PCA-DIM20-EPOCH150', index=False, header=False)