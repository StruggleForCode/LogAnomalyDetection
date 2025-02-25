import json             # 导入json模块，用于处理json格式的数据
import pandas as pd     # 导入pandas模块，用于数据读取和数据处理
import numpy as np      # 导入numpy模块，用于数值计算和数组操作
import keras            # 导入keras，用于构建和训练神经网络模型
from tensorflow.keras.models import Model    # 从tensorflow.keras.models模块中导入Model类，用于构建模型
# 从tensorflow.keras.layers中导入常用层函数：add（相加层）、Input（输入层）、Conv1D（一维卷积层）、Activation（激活层）、Flatten（展平层）、Dense（全连接层）、GlobalAveragePooling1D（全局平均池化层）、BatchNormalization（批量归一化层）
from tensorflow.keras.layers import add, Input, Conv1D, Activation, Flatten, Dense, GlobalAveragePooling1D, BatchNormalization
# 导入time模块，用于时间处理（虽然本代码中未使用）
import time
# 导入tensorflow模块
import tensorflow as tf
# 导入tensorflow.keras.backend模块，并命名为K，用于访问后端的会话等功能
import tensorflow.keras.backend as K



# Tool 1: Calculate flops
# 定义函数用于计算模型的浮点运算次数（FLOPs）
def get_flops(model):
    # 创建RunMetadata对象，用于存储运行过程中的元数据信息
    run_meta = tf.compat.v1.RunMetadata()
    # 使用ProfileOptionBuilder构造选项，此处选项为统计浮点运算数
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    # 获取默认图（注意：在禁用 eager execution 后 tf.compat.v1.get_default_graph() 可以返回计算图）
    graph = tf.compat.v1.get_default_graph()
    # 使用tf.profiler.profile对当前会话中的计算图进行性能分析，统计所有操作的浮点运算数
    flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)
    # 返回计算得到的总浮点运算数
    return flops.total_float_ops



'''
    Step1: Dimensionality reduction of high-dimensional semantic vectors using PCA-PPA. 
    通过PCA-PPA对高维语义向量进行降维，此步骤将300维语义向量降到20维。
'''
# 打开存储高维语义向量的json文件
with open('./data/hdfs_semantic_vec.json') as f:
    # Step1-1: 读取json文件内容，将json数据加载为python字典
    gdp_list = json.load(f)
    # 提取字典中所有的value，构成一个列表，每个value代表一个300维向量
    value = list(gdp_list.values())

    # Step1-2: 使用PCA对数据进行初步降维到15维
    from sklearn.decomposition import PCA  # 导入sklearn库中的PCA模块
    estimator = PCA(n_components=15)  # 创建PCA对象，将目标维数设为15
    pca_result = estimator.fit_transform(value)  # 对300维数据进行PCA降维，得到15维表示

    # Step1-3: 进行PPA处理，即对降维后的数据去均值并进一步调整
    ppa_result = []  # 初始化列表，用于存储PPA处理后的结果
    # 对pca_result进行去均值操作，每个样本向量减去整个数据集的均值
    result = pca_result - np.mean(pca_result)
    # 再次进行PCA降维（此处依然设置n_components=15），用于获取主成分
    pca = PCA(n_components=15)
    pca_result = pca.fit_transform(result)
    # 获取PCA得到的主成分（每一行是一个主成分对应的向量）
    U = pca.components_
    # 对每个样本向量进行处理
    for i, x in enumerate(result):
        # 对前7个主成分进行循环处理，去除x在这些主成分方向上的成分
        for u in U[0:7]:
            # 计算x在u方向上的投影，并从x中减去该投影
            x = x - np.dot(u.transpose(), x) * u
        # 将处理后的向量转换为列表形式并添加到ppa_result中
        ppa_result.append(list(x))
    # 将ppa_result转换为numpy数组，便于后续数值计算
    ppa_result = np.array(ppa_result)


'''
    Step2: Read training data. In this process it is necessary to ensure a balance between abnormal and normal samples.
    读取训练数据，此过程中需要保证异常样本和正常样本的平衡。
'''
def read_data(path, split=0.7):
    # 读取csv文件中的日志序列数据（每行包含日志数据及其标签）
    logs_series = pd.read_csv(path)
    # 将pandas的DataFrame转换为numpy数组
    logs_series = logs_series.values
    # 取第二列作为标签（通常标记为异常或正常）
    label = logs_series[:, 1]
    # 取第一列作为日志数据，每条数据为一串数字（事件索引）
    logs_data = logs_series[:, 0]
    logs = []  # 用于存储每条日志数据的向量表示
    # 遍历所有日志数据
    for i in range(0, len(logs_data)):
        # 初始化一个形状为(300,15)的零数组，300表示日志中最多包含300个事件，15为每个事件的向量维度
        padding = np.zeros((300, 15))
        # 获取第i条日志数据（字符串形式）
        data = logs_data[i]
        # 将字符串按照空格分割，并将每个分割后的数字转换为整数，得到事件索引列表
        data = [int(n) for n in data.split()]
        # 遍历当前日志中的每个事件索引
        for j in range(0, len(data)):
            # 根据事件索引从预处理好的ppa_result中获取对应的15维向量，填入padding数组中对应的位置
            padding[j] = pca_result[data[j]]
        # 将padding数组转换为列表形式后添加到logs列表中
        padding = list(padding)
        logs.append(padding)
    # 将所有日志数据转换为numpy数组
    logs = np.array(logs)
    # 根据split比例计算训练集和验证集之间的分界索引
    split_boundary = int(logs.shape[0] * split)
    # 取前split_boundary个样本作为训练集数据
    train_x = logs[:split_boundary]
    # 剩余样本作为验证集数据
    valid_x = logs[split_boundary:]
    # 取相应的标签作为训练集标签
    train_y = label[:split_boundary]
    # 剩余标签作为验证集标签
    valid_y = label[split_boundary:]
    # 重塑训练集数据的形状为 (样本数, 300, 15)
    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 15))
    # 重塑验证集数据的形状为 (样本数, 300, 15)
    valid_x = np.reshape(valid_x, (valid_x.shape[0], valid_x.shape[1], 15))
    # 将标签转换为one-hot编码形式，便于分类任务训练
    train_y = keras.utils.to_categorical(np.array(train_y))
    valid_y = keras.utils.to_categorical(np.array(valid_y))
    # 返回训练数据、训练标签、验证数据和验证标签
    return train_x, train_y, valid_x, valid_y



# Residual block
# 定义残差块函数，用于构建Temporal Convolutional Network (TCN)中的基础模块
def ResBlock(x, filters, kernel_size, dilation_rate):
    # 使用1D卷积层进行卷积操作，设置过滤器个数、卷积核大小、扩张系数和激活函数ReLU
    r = Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate, activation='relu')(x)
    # 第二个卷积层，用于进行特征重参数化，卷积核大小固定为3，扩张系数同上
    # 注释中提到可以使用BatchNormalization或Weight Normalization进行归一化，但此处暂未启用
    r = Conv1D(1, 3, padding='same', dilation_rate=dilation_rate)(r)
    # 判断输入x的通道数是否与过滤器数量匹配
    if x.shape[-1] == filters:
        # 如果匹配，直接使用x作为shortcut分支
        shortcut = x
    else:
        # 如果不匹配，通过1D卷积调整x的通道数，使其与r相同
        shortcut = Conv1D(filters, kernel_size, padding='same')(x)  # shortcut分支
    # 将卷积分支r和shortcut分支相加，实现残差连接
    o = add([r, shortcut])
    # 对相加后的结果应用ReLU激活函数
    o = Activation('relu')(o)
    # 返回经过残差块处理后的输出
    return o

'''
    Step3: training model. Since our proposed method works better, we have not optimised the parameters too much.
    This method can be further improved by parameter optimisation.
    Step3: 训练模型。由于我们提出的方法效果较好，因此参数未做过多优化。后续可以通过参数调优进一步提升效果。
'''
def TCN(train_x, train_y, valid_x, valid_y):
    # 定义模型输入，形状为(300,15)，代表每条日志有300个事件，每个事件15维
    inputs = Input(shape=(300, 15))
    # 第一个残差块：设置过滤器数量为3，卷积核大小为3，扩张系数为1
    x = ResBlock(inputs, filters=3, kernel_size=3, dilation_rate=1)
    # 第二个残差块：扩张系数增大为2
    x = ResBlock(x, filters=3, kernel_size=3, dilation_rate=2)
    # 第三个残差块：扩张系数为4
    x = ResBlock(x, filters=3, kernel_size=3, dilation_rate=4)
    # 第四个残差块：扩张系数为8，进一步扩大感受野
    x = ResBlock(x, filters=3, kernel_size=3, dilation_rate=8)
    # 使用全局平均池化层对时间步维度进行池化，代替直接展平（Flatten），可以减少过拟合
    x = GlobalAveragePooling1D()(x)
    # 添加全连接层，输出节点数为2（对应二分类任务），激活函数使用softmax
    x = Dense(2, activation='softmax')(x)
    # 定义模型，指定输入和输出
    #model = Model(input=inputs, output=x)
    model = Model(inputs=inputs, outputs=x)
    # 输出模型的浮点运算数（FLOPs）
    print('flops is ', get_flops(model))
    # 打印模型结构，便于观察网络各层的参数和输出形状
    model.summary()
    # 编译模型，采用adam优化器，二元交叉熵损失函数，并以准确率作为评估指标
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # 开始训练模型：设置批次大小为64，训练100个epoch，verbose=2显示训练过程，并使用验证集数据进行验证
    model.fit(train_x, train_y, batch_size=64, epochs=100, verbose=2, validation_data=(valid_x, valid_y))
    # 指定训练后模型的保存路径
    model_path = './model/E_TCN_GAP_PCA_dim15.h5'
    print(model_path)
    # 将训练好的模型保存到指定文件中
    model.save(model_path)



# 指定训练数据文件的路径
train_path = './data/log_train.csv'
# 通过read_data函数读取训练数据和验证数据，并进行预处理
train_x, train_y, valid_x, valid_y = read_data(train_path)
# 调用TCN函数开始训练模型
TCN(train_x, train_y, valid_x, valid_y)