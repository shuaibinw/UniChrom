import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Flatten, Activation, BatchNormalization
from tensorflow.keras import Model, initializers
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
import pyBigWig
import pandas as pd
from tensorflow.keras.regularizers import l2
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense,Dropout, BatchNormalization, Bidirectional, GRU, Input, Conv1D, MaxPooling1D, LSTM, Flatten,Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import roc_curve, auc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Reshape, RepeatVector
from tensorflow.keras.layers import Lambda
from keras.layers import Input, Conv1D, MaxPooling1D, Dropout, Flatten, Dense, BatchNormalization, LSTM, Activation, Bidirectional, Concatenate
from keras.optimizers import Adam
from keras.regularizers import l2
import pickle
import numpy as np
import os
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten
import os
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

#第一组数据

anchors1_data_test = np.load('data_test_1.npy')
anchors1_label_test = np.load('label_test_1.npy')

anchors2_data_test = np.load('data_test_2.npy')
anchors2_label_test = np.load('label_test_2.npy')



# Model parameters
merged_input1_length = 5000 # TODO: get this from input
merged_input2_length = 5000 # TODO: get this from input
n_kernels = 1024 # Number of kernels; used to be 1024
filter_length = 40 # Length of each kernel
LSTM_out_dim = 512 # Output direction of ONE DIRECTION of LSTM; used to be 512
dense_layer_size = 925

# Define enhancer branch
merged_input1 = Input(shape=(merged_input1_length, 4))
enhancer_conv = Conv1D(filters=n_kernels,
                       kernel_size=filter_length,
                       padding="valid",
                       kernel_regularizer=l2(1e-5))(merged_input1)
enhancer_relu = Activation("relu")(enhancer_conv)
enhancer_max_pool = MaxPooling1D(pool_size=filter_length // 2,
                                 strides=filter_length // 2)(enhancer_relu)

# Define promoter branch
merged_input2 = Input(shape=(merged_input2_length, 4))
promoter_conv = Conv1D(filters=n_kernels,
                       kernel_size=filter_length,
                       padding="valid",
                       kernel_regularizer=l2(1e-5))(merged_input2)
promoter_relu = Activation("relu")(promoter_conv)
promoter_max_pool = MaxPooling1D(pool_size=filter_length // 2,
                                 strides=filter_length // 2)(promoter_relu)

# Concatenate enhancer and promoter branches
merged = Concatenate(axis=1)([enhancer_max_pool, promoter_max_pool])

# Define the main model layers
biLSTM = Bidirectional(LSTM(LSTM_out_dim, return_sequences=True))(merged)
flat = Flatten()(biLSTM)
dense = Dense(dense_layer_size, kernel_initializer="glorot_uniform", kernel_regularizer=l2(1e-6))(flat)
dense_bn = BatchNormalization()(dense)
dense_relu = Activation("relu")(dense_bn)
dense_drop = Dropout(0.5)(dense_relu)
output = Dense(1)(dense_drop)
output_bn = BatchNormalization()(output)
output_sigmoid = Activation("sigmoid")(output_bn)

model1 = Model(inputs=[merged_input1,merged_input2 ], outputs=output_sigmoid)

model1.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
model1.summary()

# 评估模型
model1.load_weights('SPEID_k562_seq_SMC.h5')
score = model1.evaluate([anchors1_data_test, anchors2_data_test],
                        anchors2_label_test, verbose=0)

# ROC曲线
pred1 = model1.predict([anchors1_data_test, anchors2_data_test])
fpr1, tpr1, thresholds = roc_curve(anchors2_label_test, pred1)
roc_auc1 = auc(fpr1, tpr1)
print('AUC:', roc_auc1)
precision1, recall1, _ = precision_recall_curve(anchors2_label_test, pred1)
auprc1 = average_precision_score(anchors2_label_test, pred1)

# 定义输入层
sequence_input1 = Input(shape=(5000, 4))
sequence_input2 = Input(shape=(5000, 4))

# 第一组数据处理卷积池化模块
x1 = Conv1D(512, 20, padding='same', activation='relu', kernel_regularizer=l2(0.001))(sequence_input1)
x1 = MaxPooling1D(10)(x1)
x1 = Conv1D(512, 20, padding='same', activation='relu', kernel_regularizer=l2(0.001))(x1)
x1 = MaxPooling1D(10)(x1)
x1 = BatchNormalization()(x1)
x1 = Dropout(0.5)(x1)

x2 = Conv1D(512, 20, padding='same', activation='relu', kernel_regularizer=l2(0.001))(sequence_input2)
x2 = MaxPooling1D(10)(x2)
x2 = Conv1D(512, 20, padding='same', activation='relu', kernel_regularizer=l2(0.001))(x2)
x2 = MaxPooling1D(10)(x2)
x2 = BatchNormalization()(x2)
x2 = Dropout(0.5)(x2)

merge1 = Concatenate(axis=1)([x1, x2])
merge1 = Bidirectional(GRU(512, return_sequences=True))(merge1)
final_merged = BatchNormalization()(merge1)
final_merged = Dropout(0.5)(final_merged)

dense = Dense(128, activation='relu')(final_merged)
dense = BatchNormalization()(dense)
dense = Dropout(0.5)(dense)

output = Dense(1, activation='sigmoid')(dense)

model2 = Model(inputs=[sequence_input1, sequence_input2], outputs=output)

model2.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
model2.summary()

# 评估模型
model2.load_weights('new_k562_sq_SMC.h5')
score = model2.evaluate([anchors1_data_test, anchors2_data_test],
                       anchors2_label_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# 预测
pred2 = model2.predict([anchors1_data_test, anchors2_data_test])

if len(pred2.shape) == 3:
    pred_flat = pred2.reshape(-1)
    anchors2_label_test_flat = np.repeat(anchors2_label_test, pred2.shape[1])
else:
    pred_flat = pred2.flatten()
    anchors2_label_test_flat = anchors2_label_test.flatten()

# 计算 ROC 曲线
fpr2, tpr2, thresholds = roc_curve(anchors2_label_test_flat, pred_flat)
roc_auc2 = auc(fpr2, tpr2)
print('AUC:', roc_auc2)
precision2, recall2, _ = precision_recall_curve(anchors2_label_test_flat, pred_flat)
auprc2 = average_precision_score(anchors2_label_test_flat, pred_flat)

X_en_test = np.load('k562_anchor1_ts.npy')
X_pr_test = np.load('k562_anchor2_ts.npy')
y_test = np.load('k562_labels_test_1.npy')

enhancer_input = Input(shape=(64, 64, 4))
enhancer_first_conv1 = Conv2D(128, (3, 3), activation='relu', strides=1)(enhancer_input)  # 256
enhancer_first_MaxPool1 = MaxPooling2D((2, 2), strides=(1, 1))(enhancer_first_conv1)
enhancer_first_MaxPool1 = Flatten()(enhancer_first_MaxPool1)

promoter_input = Input(shape=(64, 64, 4))
promoter_first_conv1 = Conv2D(128, (3, 3), activation='relu', strides=1)(promoter_input)  # 128
promoter_first_MaxPool1 = MaxPooling2D((2, 2), strides=(1, 1))(promoter_first_conv1)
promoter_first_MaxPool1 = Flatten()(promoter_first_MaxPool1)

branch_output = layers.concatenate([enhancer_first_MaxPool1, promoter_first_MaxPool1], axis=1)
branch_output1 = layers.Dropout(0.5)(branch_output)
branch_output2 = Dense(128, activation='relu')(branch_output1)
output = Dense(1, activation='sigmoid')(branch_output2)
model3 = Model(inputs=[enhancer_input, promoter_input], outputs=[output])
# model.summary()



# model = Model(inputs=[enhancer_input,  promoter_input], outputs=output)

model3.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
model3.summary()





# 评估模型
model3.load_weights('k562_EPIHilbert.h5')
score = model3.evaluate([X_en_test , X_pr_test],
                        y_test, verbose=0)


# ROC曲线
pred3 = model3.predict([X_en_test , X_pr_test])
fpr3, tpr3, thresholds = roc_curve(y_test, pred3)
roc_auc3 = auc(fpr3, tpr3)
print('AUC:', roc_auc3)

precision3, recall3, _ = precision_recall_curve(y_test, pred3)
auprc3 = average_precision_score(y_test, pred3)




#第四个模型
# Reshape data
X_test_1 = anchors1_data_test.reshape(anchors1_data_test.shape[0], -1)
X_test_2 = anchors2_data_test.reshape(anchors2_data_test.shape[0], -1)

# Combine anchor data
X_test = np.hstack((X_test_1, X_test_2))
y_test = anchors1_label_test

# Load the trained model
model_path = 'saved_models/k562_gradient_boosting_model.pkl'
with open(model_path, 'rb') as f:
    estimator = pickle.load(f)
print(f"Model loaded from {model_path}")

# Make predictions
y_pred_proba = estimator.predict_proba(X_test)[:, 1]
y_pred = estimator.predict(X_test)



# 计算第四个模型的 ROC 曲线
fpr4, tpr4, _ = roc_curve(y_test, y_pred_proba)
roc_auc4 = auc(fpr4, tpr4)

# 计算第四个模型的 Precision-Recall 曲线
precision4, recall4, _ = precision_recall_curve(y_test, y_pred_proba)
auprc4 = average_precision_score(y_test, y_pred_proba)


# 加载测试数据
anchors1_data_test = np.load('k562_features_test_1.npy')
anchors2_data_test = np.load('k562_features_test_2.npy')
test_labels = np.load('k562_labels_test_1.npy')

# 合并测试特征
combined_features_test = np.concatenate((anchors1_data_test, anchors2_data_test), axis=1)

# 加载保存的模型
print('Loading models...')
knn_predictor = joblib.load("k562_knn_predictor.pkl")
xgb_predictor = joblib.load("k562_xgb_predictor.pkl")
lgb_predictor = joblib.load("k562_lgb_predictor.pkl")
lr_predictor = joblib.load("k562_lr_predictor.pkl")

# 进行预测
print('Testing models...')
print('Predicting KNN...')
knn_test_pred = knn_predictor.predict_proba(combined_features_test)[:, 1]

print('Predicting XGB...')
xgb_test_pred = xgb_predictor.predict(xgb.DMatrix(combined_features_test))

print('Predicting LGB...')
lgb_test_pred = lgb_predictor.predict(combined_features_test)



# 构建stacking特征并进行最终预测
stacking_X_test = np.vstack([knn_test_pred, xgb_test_pred, lgb_test_pred]).T

print('Predicting Stacking Model (Logistic Regression)...')
lr_test_pred = lr_predictor.predict_proba(stacking_X_test)[:, 1]

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# 计算堆叠模型（第五个模型）的 ROC 曲线
fpr5, tpr5, _ = roc_curve(test_labels, lr_test_pred)
roc_auc5 = auc(fpr5, tpr5)

# 计算堆叠模型的 Precision-Recall 曲线
precision5, recall5, _ = precision_recall_curve(test_labels, lr_test_pred)
auprc5 = average_precision_score(test_labels, lr_test_pred)



# 为每个模型分别保存ROC和PR曲线数据
models = ['SPEID', 'UniChrom', 'EPI_Hilbert', 'Targetfinder', 'Fusnet']
fprs = [fpr1, fpr2, fpr3, fpr4, fpr5]
tprs = [tpr1, tpr2, tpr3, tpr4, tpr5]
precisions = [precision1, precision2, precision3, precision4, precision5]
recalls = [recall1, recall2, recall3, recall4, recall5]
roc_aucs = [roc_auc1, roc_auc2, roc_auc3, roc_auc4, roc_auc5]
auprcs = [auprc1, auprc2, auprc3, auprc4, auprc5]

for i, model in enumerate(models):
    # 保存ROC数据
    roc_filename = f'{model}_ROC.txt'
    with open(roc_filename, 'w') as f:
        f.write(f'{model} ROC Curve Data\n')
        f.write('=' * (len(model) + 14) + '\n\n')
        f.write(f'ROC AUC Score: {roc_aucs[i]:.4f}\n\n')
        
        # 写入表头
        f.write('X-axis: False Positive Rate (FPR)\n')
        f.write('Y-axis: True Positive Rate (TPR)\n\n')
        
        # 写入数据
        f.write('FPR\tTPR\n')
        f.write('-' * 20 + '\n')
        for j in range(len(fprs[i])):
            f.write(f'{fprs[i][j]:.6f}\t{tprs[i][j]:.6f}\n')
    
    # 保存PR数据
    pr_filename = f'{model}_PR.txt'
    with open(pr_filename, 'w') as f:
        f.write(f'{model} PR Curve Data\n')
        f.write('=' * (len(model) + 13) + '\n\n')
        f.write(f'PR AUC Score: {auprcs[i]:.4f}\n\n')
        
        # 写入表头
        f.write('X-axis: Recall\n')
        f.write('Y-axis: Precision\n\n')
        
        # 写入数据
        f.write('Recall\tPrecision\n')
        f.write('-' * 20 + '\n')
        for j in range(len(recalls[i])):
            f.write(f'{recalls[i][j]:.6f}\t{precisions[i][j]:.6f}\n')

# 创建一个总体性能对比文件
with open('models_comparison.txt', 'w') as f:
    f.write('Models Performance Comparison\n')
    f.write('===========================\n\n')
    f.write('Model         ROC AUC    PR AUC\n')
    f.write('--------------------------------\n')
    for i, model in enumerate(models):
        f.write(f'{model:<12} {roc_aucs[i]:>.4f}    {auprcs[i]:.4f}\n')