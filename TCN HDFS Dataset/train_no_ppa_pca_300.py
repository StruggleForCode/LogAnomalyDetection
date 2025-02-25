import json
import pandas as pd
import numpy as np
import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import add, Input, Conv1D, Activation, GlobalAveragePooling1D, Dense
import tensorflow as tf
import tensorflow.keras.backend as K


def get_flops(model):
    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    graph = tf.compat.v1.get_default_graph()
    flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)
    return flops.total_float_ops


# Step1: Load high-dimensional semantic vectors (without PCA/PPA)
with open('./data/hdfs_semantic_vec.json') as f:
    gdp_list = json.load(f)
    semantic_vectors = np.array(list(gdp_list.values()))  # 直接使用 300 维向量


# Step2: Read training data
def read_data(path, split=0.7):
    logs_series = pd.read_csv(path).values
    label = logs_series[:, 1]
    logs_data = logs_series[:, 0]
    logs = []

    for i in range(len(logs_data)):
        padding = np.zeros((300, 300))  # 300 维度
        data = [int(n) for n in logs_data[i].split()]
        for j in range(len(data)):
            padding[j] = semantic_vectors[data[j]]
        logs.append(padding)

    logs = np.array(logs)
    split_boundary = int(logs.shape[0] * split)
    train_x, valid_x = logs[:split_boundary], logs[split_boundary:]
    train_y, valid_y = label[:split_boundary], label[split_boundary:]
    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 300))
    valid_x = np.reshape(valid_x, (valid_x.shape[0], valid_x.shape[1], 300))
    train_y = keras.utils.to_categorical(np.array(train_y))
    valid_y = keras.utils.to_categorical(np.array(valid_y))
    return train_x, train_y, valid_x, valid_y


# Residual block
def ResBlock(x, filters, kernel_size, dilation_rate):
    r = Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate, activation='relu')(x)
    r = Conv1D(1, 3, padding='same', dilation_rate=dilation_rate)(r)
    shortcut = x if x.shape[-1] == filters else Conv1D(filters, kernel_size, padding='same')(x)
    o = add([r, shortcut])
    return Activation('relu')(o)


# Step3: Train TCN model
def TCN(train_x, train_y, valid_x, valid_y):
    inputs = Input(shape=(300, 300))
    x = ResBlock(inputs, filters=3, kernel_size=3, dilation_rate=1)
    x = ResBlock(x, filters=3, kernel_size=3, dilation_rate=2)
    x = ResBlock(x, filters=3, kernel_size=3, dilation_rate=4)
    x = ResBlock(x, filters=3, kernel_size=3, dilation_rate=8)
    x = GlobalAveragePooling1D()(x)
    x = Dense(2, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=x)

    print('flops is', get_flops(model))
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_x, train_y, batch_size=64, epochs=100, verbose=2, validation_data=(valid_x, valid_y))
    model.save('./model/E_TCN_GAP_dim300.h5')


# Load data and train model
train_path = './data/log_train.csv'
train_x, train_y, valid_x, valid_y = read_data(train_path)
TCN(train_x, train_y, valid_x, valid_y)
