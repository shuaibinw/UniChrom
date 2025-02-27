import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import pyBigWig
import pandas as pd
from tensorflow.keras.regularizers import l2
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense,Dropout, BatchNormalization, Bidirectional,Reshape, GRU, Input, Conv1D, Add,MaxPooling1D, LSTM, Flatten,Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import roc_curve, auc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, f1_score
import time

import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

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



bigwig_files = ['IMR90-H3K4me1.fc.signal.bigwig','IMR90-H3K4me2.fc.signal.bigwig',
                'IMR90-H3K4me3.fc.signal.bigwig', 'IMR90-H3K9ac.fc.signal.bigwig',
                'IMR90-H3K9me3.fc.signal.bigwig', 'IMR90-H3K27ac.fc.signal.bigwig',
                'IMR90-H3K27me3.fc.signal.bigwig', 'IMR90-H3K36me3.fc.signal.bigwig',
                'IMR90-H3K79me2.fc.signal.bigwig', 'IMR90-H4K20me1.fc.signal.bigwig',
                'IMR90_Rad21.bigWig', 'IMR90_SMC3.bigWig', 'IMR90_CTCF.bigWig']



bed_files_train_anchors1 = ['train_pos_1.bed', 'train_neg_1.bed']
bed_files_val_anchors1 = ['val_pos_1.bed', 'val_neg_1.bed']                                    
bed_files_test_anchors1 = ['test_pos_1.bed', 'test_neg_1.bed']

bed_files_train_anchors2 = ['train_pos_2.bed', 'train_neg_2.bed']
bed_files_val_anchors2 = ['val_pos_2.bed', 'val_neg_2.bed']                                    
bed_files_test_anchors2 = ['test_pos_2.bed', 'test_neg_2.bed']

#第二组数据
histone_train_anchors1 = load_histone_modification_data(bed_files_train_anchors1, bigwig_files)
histone_val_anchors1 = load_histone_modification_data(bed_files_val_anchors1, bigwig_files)
histone_test_anchors1 = load_histone_modification_data(bed_files_test_anchors1, bigwig_files)

histone_train_anchors2= load_histone_modification_data(bed_files_train_anchors2, bigwig_files)
histone_val_anchors2 = load_histone_modification_data(bed_files_val_anchors2, bigwig_files)
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

model = Model(inputs=[sequence_input1, sequence_input2, histone_input1, histone_input2], outputs=output)

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()



# 评估模型
model.load_weights('new_gm12878_SMC.h5')
score = model.evaluate([anchors1_data_test, anchors2_data_test, histone_test_anchors1, histone_test_anchors2],
                       anchors2_label_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# ROC曲线
pred = model.predict([anchors1_data_test, anchors2_data_test, histone_test_anchors1, histone_test_anchors2])
fpr, tpr, thresholds = roc_curve(anchors2_label_test, pred)
roc_auc = auc(fpr, tpr)
print('AUC:', roc_auc)





# # 计算精确度, 召回率, 准确率, F1分数
# pred_labels = (pred > 0.5).astype(int)
# precision = precision_score(anchors2_label_test, pred_labels)
# recall = recall_score(anchors2_label_test, pred_labels)
# accuracy = accuracy_score(anchors2_label_test, pred_labels)
# f1 = f1_score(anchors2_label_test, pred_labels)


# print('Precision:', precision)
# print('Recall:', recall)
# print('Accuracy:', accuracy)
# print('F1 Score:', f1)


from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, f1_score, accuracy_score, recall_score, precision_score
# 计算其他指标
def compute_metrics(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    auprc = average_precision_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, np.round(y_pred))
    f1 = f1_score(y_true, np.round(y_pred))
    recall_score_ = recall_score(y_true, np.round(y_pred))
    precision_score_ = precision_score(y_true, np.round(y_pred))
    return roc_auc, auprc, accuracy, f1, recall_score_, precision_score_

roc_auc1, auprc1, acc1, f11, recall1, precision1 = compute_metrics(anchors2_label_test, pred)
print(f"GM12878 - AUROC: {roc_auc1:.4f}, AUPRC: {auprc1:.4f}, ACC: {acc1:.4f}, F1: {f11:.4f}, Recall: {recall1:.4f}, Precision: {precision1:.4f}")



# import tensorflow as tf
# from tensorflow.keras.optimizers import Adam
# import pyBigWig
# import pandas as pd
# from tensorflow.keras.regularizers import l2
# from tensorflow.keras import layers
# from tensorflow.keras.layers import Dense,Dropout, BatchNormalization, Bidirectional,Reshape, GRU, Input, Conv1D, Add,MaxPooling1D, LSTM, Flatten,Concatenate
# from tensorflow.keras.models import Model
# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
# from sklearn.metrics import roc_curve, auc
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

# from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, f1_score
# import time

# import numpy as np
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# #第一组数据

# anchors1_data_test = np.load('data_test_1.npy')
# anchors1_label_test = np.load('label_test_1.npy')


# anchors2_data_test = np.load('data_test_2.npy')
# anchors2_label_test = np.load('label_test_2.npy')

# def load_histone_modification_data(bed_files, bigwig_files):
#     all_features = []
#     for bed_file in bed_files:
#         bed_df = pd.read_csv(bed_file, sep='\t', header=None)
#         bed_features = []
#         for index, row in bed_df.iterrows():
#             sample_features = []
#             for bigwig_file in bigwig_files:
#                 bw = pyBigWig.open(bigwig_file)
#                 value = bw.stats(row[0], row[1], row[2], exact=True)[0]
#                 sample_features.append(value if value is not None else 0)
#                 bw.close()
#             loop_distance = abs(row[2] - row[1])
#             sample_features.append(loop_distance)
#             bed_features.append(sample_features)
#         all_features.extend(bed_features)
#     return np.array(all_features)




# bigwig_files = ['GM12878_H3K4me1-log2r.bw','GM12878_H3K4me2-log2r.bw',
#                 'GM12878_H3K4me3-log2r.bw', 'GM12878_H3K9ac-log2r.bw',
#                 'GM12878_H3K9me3-log2r.bw', 'GM12878_H3K27ac-log2r.bw',
#                 'GM12878_H3K27me3-log2r.bw', 'GM12878_H3K36me3-log2r.bw',
#                 'GM12878_H3K79me2-log2r.bw', 'GM12878_H4K20me1-log2r.bw',
#                 'GM12878_Rad21.bigWig', 'GM12878_SMC3.bigWig', 'GM12878_CTCF.bigWig']


# bed_files_train_anchors1 = ['train_pos_1.bed', 'train_neg_1.bed']
# bed_files_val_anchors1 = ['val_pos_1.bed', 'val_neg_1.bed']                                    
# bed_files_test_anchors1 = ['test_pos_1.bed', 'test_neg_1.bed']

# bed_files_train_anchors2 = ['train_pos_2.bed', 'train_neg_2.bed']
# bed_files_val_anchors2 = ['val_pos_2.bed', 'val_neg_2.bed']                                    
# bed_files_test_anchors2 = ['test_pos_2.bed', 'test_neg_2.bed']

# #第二组数据
# histone_train_anchors1 = load_histone_modification_data(bed_files_train_anchors1, bigwig_files)
# histone_val_anchors1 = load_histone_modification_data(bed_files_val_anchors1, bigwig_files)
# histone_test_anchors1 = load_histone_modification_data(bed_files_test_anchors1, bigwig_files)

# histone_train_anchors2= load_histone_modification_data(bed_files_train_anchors2, bigwig_files)
# histone_val_anchors2 = load_histone_modification_data(bed_files_val_anchors2, bigwig_files)
# histone_test_anchors2 = load_histone_modification_data(bed_files_test_anchors2, bigwig_files)

# np.random.seed(666)


# from keras.layers import Bidirectional, LSTM, Attention

# def residual_block(x, filters, kernel_size, padding='same', activation='relu', kernel_regularizer=l2(0.001)):
#     shortcut = x
#     x = Conv1D(filters, kernel_size, padding=padding, activation=activation, kernel_regularizer=kernel_regularizer)(x)
#     x = Conv1D(filters, kernel_size, padding=padding, activation=None, kernel_regularizer=kernel_regularizer)(x)
#     x = Add()([x, shortcut])
#     x = layers.Activation(activation)(x)
#     return x

# sequence_input1 = Input(shape=(5000, 4))
# sequence_input2 = Input(shape=(5000, 4))
# histone_input1 = Input(shape=(14,))
# histone_input2 = Input(shape=(14,))

# x1 = Conv1D(512, 20, padding='same', activation='relu', kernel_regularizer=l2(0.001))(sequence_input1)
# x1 = MaxPooling1D(10)(x1)
# x1 = residual_block(x1, 512, 20)
# x1 = MaxPooling1D(10)(x1)
# x1 = BatchNormalization()(x1)
# x1 = Dropout(0.5)(x1)
# x1 = Bidirectional(GRU(512, return_sequences=True))(x1)

# x2 = Conv1D(512, 20, padding='same', activation='relu', kernel_regularizer=l2(0.001))(sequence_input2)
# x2 = MaxPooling1D(10)(x2)
# x2 = residual_block(x2, 512, 20)
# x2 = MaxPooling1D(10)(x2)
# x2 = BatchNormalization()(x2)
# x2 = Dropout(0.5)(x2)
# x2 = Bidirectional(GRU(512, return_sequences=True))(x2)

# merged_sequences = Concatenate()([x1, x2])

# x3 = Reshape((14, 1))(histone_input1)
# x4 = Reshape((14, 1))(histone_input2)
# merged_histones = Concatenate()([x3, x4])
# merged_histones = Bidirectional(GRU(8, return_sequences=True))(merged_histones)


# reshaped_histones = Flatten()(merged_histones)
# reshaped_histones = Dense(np.prod(merged_sequences.shape[1:]), activation='relu')(reshaped_histones)
# reshaped_histones = Reshape(merged_sequences.shape[1:])(reshaped_histones)

# final_merged = Concatenate(axis=1)([merged_sequences, reshaped_histones])

# final_merged = Bidirectional(LSTM(256, return_sequences=False))(final_merged)
# final_merged = BatchNormalization()(final_merged)
# final_merged = Dropout(0.5)(final_merged)

# query, value = tf.split(final_merged, num_or_size_splits=2, axis=-1)
# attention_layer = Attention(name='attention_layer')([query, value])

# attention_layer = BatchNormalization()(attention_layer)
# attention_layer = Dropout(0.5)(attention_layer)
# dense = Dense(128, activation='relu')(attention_layer)
# dense = BatchNormalization()(dense)
# dense = Dropout(0.5)(dense)
# output = Dense(1, activation='sigmoid')(dense)


# anchors2_label_test= np.expand_dims(anchors2_label_test, axis=-1)

# model = Model(inputs=[sequence_input1, sequence_input2, histone_input1, histone_input2], outputs=output)

# model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
# model.summary()





# # 评估模型
# model.load_weights('k562_SMC.h5')
# score = model.evaluate([anchors1_data_test, anchors2_data_test, histone_test_anchors1, histone_test_anchors2],
#                        anchors2_label_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])


# # ROC曲线
# pred = model.predict([anchors1_data_test, anchors2_data_test, histone_test_anchors1, histone_test_anchors2])
# fpr, tpr, thresholds = roc_curve(anchors2_label_test, pred)
# roc_auc = auc(fpr, tpr)
# print('AUC:', roc_auc)



# # 计算精确度, 召回率, 准确率, F1分数
# pred_labels = (pred > 0.5).astype(int)
# precision = precision_score(anchors2_label_test, pred_labels)
# recall = recall_score(anchors2_label_test, pred_labels)
# accuracy = accuracy_score(anchors2_label_test, pred_labels)
# f1 = f1_score(anchors2_label_test, pred_labels)


# print('Precision:', precision)
# print('Recall:', recall)
# print('Accuracy:', accuracy)
# print('F1 Score:', f1)




