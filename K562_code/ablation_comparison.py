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
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

#第一组数据

anchors1_data_test = np.load('data_test_1.npy')
anchors1_label_test = np.load('label_test_1.npy')

anchors2_data_test = np.load('data_test_2.npy')
anchors2_label_test = np.load('label_test_2.npy')

def load_histone_modification_data(bed_files, bigwig_files):
    all_features = []
    for bed_file in bed_files:
        bed_df = pd.read_csv(bed_file, sep='\t', header=None)
        bed_features = []
        for index, row in bed_df.iterrows():
            sample_features = []
            for bigwig_file in bigwig_files:
                bw = pyBigWig.open(bigwig_file)
                value = bw.stats(row[0], row[1], row[2], exact=True)[0]
                sample_features.append(value if value is not None else 0)
                bw.close()
            loop_distance = abs(row[2] - row[1])
            sample_features.append(loop_distance)
            bed_features.append(sample_features)
        all_features.extend(bed_features)
    return np.array(all_features)

# bigwig_files = ['GM12878_H3K4me1-log2r.bw','GM12878_H3K4me2-log2r.bw',
#                 'GM12878_H3K4me3-log2r.bw', 'GM12878_H3K9ac-log2r.bw',
#                 'GM12878_H3K9me3-log2r.bw', 'GM12878_H3K27ac-log2r.bw',
#                 'GM12878_H3K27me3-log2r.bw', 'GM12878_H3K36me3-log2r.bw',
#                 'GM12878_H3K79me2-log2r.bw', 'GM12878_H4K20me1-log2r.bw',
#                 'GM12878_Rad21.bigWig', 'GM12878_SMC3.bigWig', 'GM12878_CTCF.bigWig']


bigwig_files = ['K562-H3K4me1.fc.signal.bigwig','K562-H3K4me2.fc.signal.bigwig',
                'K562-H3K4me3.fc.signal.bigwig', 'K562-H3K9ac.fc.signal.bigwig',
                'K562-H3K9me3.fc.signal.bigwig', 'K562-H3K27ac.fc.signal.bigwig',
                'K562-H3K27me3.fc.signal.bigwig', 'K562-H3K36me3.fc.signal.bigwig',
                'K562-H3K79me2.fc.signal.bigwig', 'K562-H4K20me1.fc.signal.bigwig',
                'K562_Rad21.bigWig', 'K562_SMC3.bigWig', 'K562_CTCF.bigWig']


# bigwig_files = ['IMR90-H3K4me1.fc.signal.bigwig','IMR90-H3K4me2.fc.signal.bigwig',
#                 'IMR90-H3K4me3.fc.signal.bigwig', 'IMR90-H3K9ac.fc.signal.bigwig',
#                 'IMR90-H3K9me3.fc.signal.bigwig', 'IMR90-H3K27ac.fc.signal.bigwig',
#                 'IMR90-H3K27me3.fc.signal.bigwig', 'IMR90-H3K36me3.fc.signal.bigwig',
#                 'IMR90-H3K79me2.fc.signal.bigwig', 'IMR90-H4K20me1.fc.signal.bigwig',
#                 'IMR90_Rad21.bigWig', 'IMR90_SMC3.bigWig', 'IMR90_CTCF.bigWig']


                             
bed_files_test_anchors1 = ['test_pos_1.bed', 'test_neg_1.bed']                           
bed_files_test_anchors2 = ['test_pos_2.bed', 'test_neg_2.bed']

histone_test_anchors1 = load_histone_modification_data(bed_files_test_anchors1, bigwig_files)
histone_test_anchors2 = load_histone_modification_data(bed_files_test_anchors2, bigwig_files)

np.random.seed(666)



from keras.layers import Bidirectional, LSTM, Attention

sequence_input1 = Input(shape=(5000, 4))
sequence_input2 = Input(shape=(5000, 4))
histone_input1 = Input(shape=(14,))
histone_input2 = Input(shape=(14,))

x1 = Conv1D(512, 20, padding='same', activation='relu', kernel_regularizer=l2(0.001))(sequence_input1)
x1 = MaxPooling1D(10)(x1)
x1 = Conv1D(512, 20, padding='same', activation='relu', kernel_regularizer=l2(0.001))(sequence_input1)
x1 = MaxPooling1D(10)(x1)
x1 = BatchNormalization()(x1)
x1 = Dropout(0.5)(x1)


x2 = Conv1D(512, 20, padding='same', activation='relu', kernel_regularizer=l2(0.001))(sequence_input2)
x2 = MaxPooling1D(10)(x2)
x2 = Conv1D(512, 20, padding='same', activation='relu', kernel_regularizer=l2(0.001))(sequence_input2)
x2 = MaxPooling1D(10)(x2)
x2 = BatchNormalization()(x2)
x2 = Dropout(0.5)(x2)


merged_sequences = Concatenate()([x1, x2])
merged_sequences = Bidirectional(GRU(512, return_sequences=True))(merged_sequences)


x3 = Reshape((14, 1))(histone_input1)
x4 = Reshape((14, 1))(histone_input2)
merged_histones = Concatenate()([x3, x4])
merged_histones = Bidirectional(GRU(8, return_sequences=True))(merged_histones)


reshaped_histones = Flatten()(merged_histones)
reshaped_histones = Dense(np.prod(merged_sequences.shape[1:]), activation='relu')(reshaped_histones)
reshaped_histones = Reshape(merged_sequences.shape[1:])(reshaped_histones)

final_merged = Concatenate(axis=1)([merged_sequences, reshaped_histones])

final_merged = Bidirectional(LSTM(256, return_sequences=False))(final_merged)
final_merged = BatchNormalization()(final_merged)
final_merged = Dropout(0.5)(final_merged)

query, value = tf.split(final_merged, num_or_size_splits=2, axis=-1)
attention_layer = Attention(name='attention_layer')([query, value])

attention_layer = BatchNormalization()(attention_layer)
attention_layer = Dropout(0.5)(attention_layer)
dense = Dense(128, activation='relu')(attention_layer)
dense = BatchNormalization()(dense)
dense = Dropout(0.5)(dense)
output = Dense(1, activation='sigmoid')(dense)


anchors2_label_test= np.expand_dims(anchors2_label_test, axis=-1)

model1 = Model(inputs=[sequence_input1, sequence_input2, histone_input1, histone_input2], outputs=output)

model1.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
model1.summary()



model1.load_weights('new_k562_SMC.h5')
score = model1.evaluate([anchors1_data_test, anchors2_data_test, histone_test_anchors1, histone_test_anchors2],
                        anchors2_label_test, verbose=0)


pred1 = model1.predict([anchors1_data_test, anchors2_data_test, histone_test_anchors1, histone_test_anchors2])
fpr1, tpr1, thresholds = roc_curve(anchors2_label_test, pred1)
roc_auc1 = auc(fpr1, tpr1)
print('AUC:', roc_auc1)

precision1, recall1, _ = precision_recall_curve(anchors2_label_test, pred1)
auprc1 = average_precision_score(anchors2_label_test, pred1)


histone_input1 = Input(shape=(14,))
histone_input2 = Input(shape=(14,))


merged_histones = Concatenate()([histone_input1, histone_input2])

# merged_histones = Reshape((-1,))(merged_histones)
output = Dense(1, activation='sigmoid')(merged_histones)



model2 = Model(inputs=[histone_input1, histone_input2], outputs=output)


# 编译模型
model2.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
model2.summary()


model2.load_weights('histones_k562_SMC.h5')
# 评估模型
score = model2.evaluate([histone_test_anchors1, histone_test_anchors2], anchors2_label_test, verbose=0)


# ROC曲线
pred2 = model2.predict([histone_test_anchors1, histone_test_anchors2])

fpr2, tpr2, thresholds = roc_curve(anchors2_label_test, pred2)
roc_auc2 = auc(fpr2, tpr2)
print('AUC:', roc_auc2)

precision2, recall2, _ = precision_recall_curve(anchors2_label_test, pred2)
auprc2 = average_precision_score(anchors2_label_test, pred2)

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

model3 = Model(inputs=[sequence_input1, sequence_input2], outputs=output)

model3.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
model3.summary()

# 评估模型
model3.load_weights('new_k562_sq_SMC.h5')
score = model3.evaluate([anchors1_data_test, anchors2_data_test],
                       anchors2_label_test, verbose=0)


# 预测
pred3 = model3.predict([anchors1_data_test, anchors2_data_test])

if len(pred3.shape) == 3:
    pred_flat = pred3.reshape(-1)
    anchors2_label_test_flat = np.repeat(anchors2_label_test, pred3.shape[1])
else:
    pred_flat = pred3.flatten()
    anchors2_label_test_flat = anchors2_label_test.flatten()

# 计算 ROC 曲线
fpr3, tpr3, thresholds = roc_curve(anchors2_label_test_flat, pred_flat)
roc_auc3 = auc(fpr3, tpr3)
print('AUC:', roc_auc3)
precision3, recall3, _ = precision_recall_curve(anchors2_label_test_flat, pred_flat)
auprc3 = average_precision_score(anchors2_label_test_flat, pred_flat)




# 为每个模型分别保存ROC和PR曲线数据
models = ['UniChrom', 'UniChrom-histones', 'UniChrom-sequence']
fprs = [fpr1, fpr2, fpr3]
tprs = [tpr1, tpr2, tpr3]
precisions = [precision1, precision2, precision3]
recalls = [recall1, recall2, recall3]
roc_aucs = [roc_auc1, roc_auc2, roc_auc3]
auprcs = [auprc1, auprc2, auprc3]

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
