import json
import pandas as pd
import numpy as np
import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import add, Input, Conv1D, Activation, GlobalAveragePooling1D, Dense
import sys
import logging

# 设置日志文件，所有控制台输出同时写入 output.log
log_file = "./log/train_dim300_epoch150_output.log"
logging.basicConfig(level=logging.INFO, format='%(message)s',
                    handlers=[logging.FileHandler(log_file, 'w'), logging.StreamHandler(sys.stdout)])


def log_print(*args, **kwargs):
    """封装 print，使其输出到日志文件和控制台"""
    print(*args, **kwargs)
    logging.info(' '.join(map(str, args)))


# Step1: Load high-dimensional semantic vectors (without PCA/PPA)
log_print("Loading semantic vector JSON file...")
with open('./data/bgl_semantic_vec.json') as f:
    gdp_list = json.load(f)
    semantic_vectors = np.array(list(gdp_list.values()), dtype=np.float32)  # 直接使用 300 维向量
    log_print(f"Loaded {len(semantic_vectors)} semantic vectors.")


# Step2: Read data
def read_data(split=0.7):
    log_print("Reading data from CSV files...")
    logs_data = pd.read_csv('./data/bgl_data.csv').values
    label = pd.read_csv('./data/bgl_label.csv').values
    log_print(f"Loaded {len(logs_data)} log sequences.")
    logs = []

    for i in range(len(logs_data)):
        padding = np.zeros((300, 300))  # 300 维度
        data = logs_data[i]
        for j in range(len(data)):
            padding[j] = semantic_vectors[int(data[j] - 1)]
        logs.append(padding)

    logs = np.array(logs)
    split_boundary = int(logs.shape[0] * split)
    train_x, valid_x = logs[:split_boundary], logs[split_boundary:]
    train_y, valid_y = label[:split_boundary], label[split_boundary:]
    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 300))
    valid_x = np.reshape(valid_x, (valid_x.shape[0], valid_x.shape[1], 300))
    train_y = keras.utils.to_categorical(np.array(train_y))
    valid_y = keras.utils.to_categorical(np.array(valid_y))
    log_print("Data successfully loaded and preprocessed.")
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
    log_print("Building and training the TCN model...")
    inputs = Input(shape=(300, 300))
    x = ResBlock(inputs, filters=3, kernel_size=3, dilation_rate=1)
    x = ResBlock(x, filters=3, kernel_size=3, dilation_rate=2)
    x = ResBlock(x, filters=3, kernel_size=3, dilation_rate=4)
    x = ResBlock(x, filters=3, kernel_size=3, dilation_rate=8)
    x = GlobalAveragePooling1D()(x)
    x = Dense(2, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=x)

    model.summary(print_fn=log_print)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_x, train_y, batch_size=64, epochs=150, verbose=2, validation_data=(valid_x, valid_y))
    model.save('./model/E-TCN-DIM300-EPOCH150.h5')
    log_print("Model training complete and saved.")


# Load data and train model
train_x, train_y, valid_x, valid_y = read_data()
y_pred = TCN(train_x, train_y, valid_x, valid_y)
