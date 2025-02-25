# 导入必要的库
import json  # 用于处理 JSON 文件
import pandas as pd  # 用于数据处理和 CSV 文件读取
import numpy as np  # 用于数值计算和数组操作
import keras  # 导入 Keras 框架（用于构建和训练神经网络）
from tensorflow.keras.models import Model  # 导入 Model 类，用于构建神经网络模型
from tensorflow.keras.layers import add, Input, Conv1D, Activation, Flatten, Dense, GlobalAveragePooling1D

# 导入各类网络层：add（张量相加）、Input（输入层）、Conv1D（一维卷积层）、Activation（激活层）、Flatten（展平层）、Dense（全连接层）、
# GlobalAveragePooling1D（全局平均池化层）

# -------------------- 语义向量降维及后处理 --------------------

# 打开并加载语义向量文件，文件存储了每个日志事件的高维语义向量
with open('./data/bgl_semantic_vec.json') as f:
    # Step1-1: 从 JSON 文件中加载数据
    gdp_list = json.load(f)  # 读取 JSON 数据，返回一个字典，其中键对应日志事件ID，值为对应的语义向量
    value = list(gdp_list.values())  # 提取所有语义向量，转换为列表

    # Step1-2: 使用主成分分析（PCA）将高维语义向量降至20维
    from sklearn.decomposition import PCA  # 导入 PCA 模块

    estimator = PCA(n_components=20)  # 创建 PCA 对象，设定目标维度为20
    pca_result = estimator.fit_transform(value)
    # 对原始语义向量进行拟合与转换，返回每个向量在20个主成分方向上的投影，保留了数据中大部分的方差信息

    # Step1-3: 投影后处理调整（PPA）：对降维结果进一步去均值和剔除全局主导分量
    ppa_result = []  # 初始化一个空列表，用于存放后处理后的向量
    result = pca_result - np.mean(pca_result)
    # 对 PCA 得到的结果进行去均值处理，确保数据中心化（每个特征减去其均值）
    pca = PCA(n_components=20)  # 再次创建 PCA 对象，目标维度仍然为20
    pca_result = pca.fit_transform(result)
    # 对中心化后的数据再次进行 PCA 分解
    U = pca.components_  # 获取 PCA 得到的主成分矩阵，每一行对应一个主成分
    for i, x in enumerate(result):  # 对于每个中心化后的样本向量（遍历所有样本）
        for u in U[0:7]:  # 对前7个主成分进行循环（认为这7个主成分反映了全局的共性特征）
            # 计算样本 x 在主成分 u 上的投影，并将该投影从 x 中减去
            x = x - np.dot(u.transpose(), x) * u
        ppa_result.append(list(x))  # 将剔除全局分量后的向量以列表形式加入结果列表中
    ppa_result = np.array(ppa_result)  # 将列表转换为 NumPy 数组，便于后续计算


# -------------------- 数据读取与构建 --------------------

def read_data(split=0.7):
    """
    读取日志数据及其对应的标签，并构造固定长度（300）的日志序列，
    每条日志事件由20维的降维语义向量表示。

    参数：
        split: 训练集与验证集的划分比例（默认为70%训练，30%验证）

    返回：
        train_x, train_y: 训练集数据及标签
        valid_x, valid_y: 验证集数据及标签
    """
    # 从 CSV 文件中读取日志数据，文件中每行包含一个日志序列，各值为日志事件的编号
    logs_data = pd.read_csv('./data/bgl_data.csv')
    logs_data = logs_data.values  # 将 DataFrame 转换为 NumPy 数组
    # 从 CSV 文件中读取日志标签，标签用于判断序列是否异常
    label = pd.read_csv('./data/bgl_label.csv')
    label = label.values  # 将 DataFrame 转换为 NumPy 数组

    logs = []  # 初始化列表，用于存储每个日志序列的处理结果
    for i in range(0, len(logs_data)):  # 遍历每一条日志序列
        padding = np.zeros((300, 20))  # 初始化一个大小为 (300, 20) 的零矩阵，300为日志序列的固定长度，20为语义向量的维度
        data = logs_data[i]  # 取出第 i 条日志序列，每个值代表一个日志事件编号
        for j in range(0, len(data)):  # 遍历该序列中的每个日志事件
            # 根据日志事件编号（编号从1开始，因此索引为 data[j]-1），
            # 从 pca_result 中提取对应的20维语义向量，赋值到矩阵中的第 j 行
            padding[j] = pca_result[int(data[j]-1)]
            # padding[j] = pca_result[int(data[j] - 1)]
        padding = list(padding)  # 将矩阵转换为列表形式（非必须步骤，但便于后续处理）
        logs.append(padding)  # 将处理后的日志序列添加到 logs 列表中
    logs = np.array(logs)  # 将所有日志序列转换为 NumPy 数组

    # 根据 split 参数计算训练集与验证集的分割边界（样本总数 * split比例）
    split_boundary = int(logs.shape[0] * split)
    train_x = logs[:split_boundary]  # 取前 split_boundary 个样本作为训练集
    valid_x = logs[split_boundary:]  # 取剩余样本作为验证集
    train_y = label[:split_boundary]  # 取对应的标签作为训练集标签
    valid_y = label[split_boundary:]  # 取对应的标签作为验证集标签

    # 重塑数据形状，确保每个样本均为 (300, 20)
    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 20))
    valid_x = np.reshape(valid_x, (valid_x.shape[0], valid_x.shape[1], 20))

    # 将标签转换为 one-hot 编码格式，适用于分类任务
    train_y = keras.utils.to_categorical(np.array(train_y))
    valid_y = keras.utils.to_categorical(np.array(valid_y))

    return train_x, train_y, valid_x, valid_y  # 返回训练集与验证集的数据和标签


# -------------------- 残差块构建 --------------------

def ResBlock(x, filters, kernel_size, dilation_rate):
    """
    构建残差块（Residual Block），包含两个一维卷积层，并利用快捷连接实现残差学习。
    同时通过扩张卷积（dilated convolution）捕捉长距离依赖关系。

    参数：
        x: 输入张量
        filters: 卷积输出通道数
        kernel_size: 卷积核大小
        dilation_rate: 扩张率，用于控制卷积的感受野
    返回：
        o: 残差块的输出张量
    """
    # 第一层卷积：一维卷积，输出通道数为 filters，卷积核大小为 kernel_size，
    # 采用 'same' 填充，扩张率为 dilation_rate，并使用 ReLU 激活函数
    r = Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate, activation='relu')(x)

    # 第二层卷积：一维卷积，输出通道数固定为 1，卷积核大小为 3，使用 'same' 填充和相同扩张率
    # 注：可在此处添加 Batch Normalization 或 Weight Normalization（此处已注释掉）
    r = Conv1D(1, 3, padding='same', dilation_rate=dilation_rate)(r)

    # 判断输入 x 的通道数是否与 filters 相等
    if x.shape[-1] == filters:
        shortcut = x  # 若相等，直接使用 x 作为快捷连接
    else:
        # 否则，使用一维卷积调整 x 的通道数，使其与 r 匹配
        shortcut = Conv1D(filters, kernel_size, padding='same')(x)

    # 将卷积分支 r 与快捷连接 shortcut 相加
    o = add([r, shortcut])
    # 对相加结果进行 ReLU 激活
    o = Activation('relu')(o)

    return o  # 返回残差块的输出


# -------------------- TCN 模型构建与训练 --------------------

'''
说明：
BGL 日志数据中每条日志序列并不包含日志事件所属的块（Block）信息，
因此我们采用滑动窗口的方法对日志序列进行划分。在本实验中，滑动窗口的单位是日志条数，
这使得我们能够灵活地控制实时异常检测的粒度。
'''


def TCN(train_x, train_y, valid_x, valid_y):
    """
    构建、训练并保存时序卷积网络（TCN）模型，用于日志异常检测。

    参数：
        train_x, train_y: 训练集数据和标签
        valid_x, valid_y: 验证集数据和标签
    """
    # 定义模型输入，输入形状为 (300, 20) —— 300为日志序列长度，20为每条日志的特征维度
    inputs = Input(shape=(300, 20))

    # 通过堆叠多个残差块构建模型，各残差块采用不同扩张率以捕捉不同尺度的时序依赖
    x = ResBlock(inputs, filters=3, kernel_size=3, dilation_rate=1)  # 第一层残差块，扩张率为1
    x = ResBlock(x, filters=3, kernel_size=3, dilation_rate=2)  # 第二层残差块，扩张率为2
    x = ResBlock(x, filters=3, kernel_size=3, dilation_rate=4)  # 第三层残差块，扩张率为4
    x = ResBlock(x, filters=3, kernel_size=3, dilation_rate=8)  # 第四层残差块，扩张率为8

    # 可选：使用 Flatten 层将卷积输出展平（此处注释，采用全局池化）
    # x = Flatten()(x)

    # 使用全局平均池化层，对时间维度（300）进行池化，得到固定长度的特征向量
    x = GlobalAveragePooling1D()(x)

    # 添加全连接层，将特征映射到2个输出神经元，用于二分类（例如：正常 vs 异常），使用 softmax 激活函数
    x = Dense(2, activation='softmax')(x)

    # 构建模型，指定输入和输出
    # 注：此处使用旧版本语法，新版本中可写为 Model(inputs=inputs, outputs=x)
    #model = Model(input=inputs, output=x)
    model = Model(inputs=inputs, outputs=x)

    # 打印模型结构摘要，便于调试和了解模型参数
    model.summary()

    # 编译模型：使用 Adam 优化器，损失函数为二分类交叉熵，并以准确率作为评估指标
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # 训练模型：设定 batch_size 为 64，训练 100 个 epoch，verbose=2 表示输出训练过程简略信息
    # 同时在验证集上监控模型性能
    model.fit(train_x, train_y, batch_size=64, epochs=100, verbose=2, validation_data=(valid_x, valid_y))

    # 训练完成后，将模型保存到指定文件中
    model.save('./model/E-TCN-PCA.h5')

# -------------------- 主程序入口 --------------------

# 调用 read_data() 函数读取数据并构造训练集与验证集
train_x, train_y, valid_x, valid_y = read_data()

# 调用 TCN() 函数构建、训练模型，并保存模型文件
y_pred = TCN(train_x, train_y, valid_x, valid_y)