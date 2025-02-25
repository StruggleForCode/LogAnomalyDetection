import json
import pandas as pd
import numpy as np
import keras
from tensorflow.keras.models import load_model
import time

# Step1: Load high-dimensional semantic vectors (without PCA/PPA)
with open('./data/hdfs_semantic_vec.json') as f:
    gdp_list = json.load(f)
    semantic_vectors = np.array(list(gdp_list.values()))  # 直接使用 300 维向量


# Step2: Read test data
def read_test(path):
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
    train_x = logs
    train_y = label
    text_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 300))
    text_y = keras.utils.to_categorical(np.array(train_y))
    return text_x, text_y, label


# Step3: Test model
def test_model(test_x):
    model = load_model('./model/E_TCN_GAP_dim300.h5')
    y_pred = model.predict(test_x, batch_size=512)
    return y_pred


# Data processing
start_data = time.perf_counter()
test_path = './data/log_test_2000.csv'
test_x, test_y, label = read_test(test_path)
end_data = time.perf_counter()
print('The data processing time is', end_data - start_data)

# Detection (prediction)
start_detect = time.perf_counter()
y_pred = test_model(test_x)
end_detect = time.perf_counter()
print('The detection time is', end_detect - start_detect)

# Processing results
y_pred = np.argmax(y_pred, axis=1)
tp, fp, tn, fn = 0, 0, 0, 0

for j in range(len(y_pred)):
    if label[j] == y_pred[j] and label[j] == 0:
        tp += 1
    elif label[j] != y_pred[j] and label[j] == 0:
        fp += 1
    elif label[j] == y_pred[j] and label[j] == 1:
        tn += 1
    elif label[j] != y_pred[j] and label[j] == 1:
        fn += 1

print('TP, FP, TN, FN are:', [tp, fp, tn, fn])

# Compute metrics
precision = tn / (tn + fn) if (tn + fn) != 0 else 0
recall = tn / (tn + fp) if (tn + fp) != 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
print('Precision, Recall, F1-measure are:', precision, recall, f1)

# Save results
datas = pd.DataFrame(data=[tp, fp, tn, fn])
datas.to_csv('./result/result_HDFS_NO_PPA_PCA_300', index=False, header=False)
