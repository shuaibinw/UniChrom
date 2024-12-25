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
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import roc_curve, auc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Reshape

import time
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Reshape

from tensorflow.keras.layers import Reshape, RepeatVector
from tensorflow.keras.layers import Lambda

from keras.layers import Input, Conv1D, MaxPooling1D, Dropout, Flatten, Dense, BatchNormalization, LSTM, Activation, Bidirectional, Concatenate
from keras.optimizers import Adam
from keras.regularizers import l2
import xgboost as xgb
import numpy as np
import os
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from joblib import load
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from transfomer import Transformer_Merged
from sklearn.preprocessing import StandardScaler
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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

#test
if anchors1_data_test.dtype == tf.int64:
    anchors1_data_test = tf.cast(anchors1_data_test, tf.float32)
histone_test_anchors1 = tf.repeat(histone_test_anchors1[:, np.newaxis, :], repeats=anchors1_data_test.shape[1], axis=1)
merged_test_input1 = Concatenate(axis=-1)([anchors1_data_test, histone_test_anchors1])
if anchors2_data_test.dtype == tf.int64:
    anchors2_data_test = tf.cast(anchors2_data_test, tf.float32)
histone_test_anchors2 = tf.repeat(histone_test_anchors2[:, np.newaxis, :], repeats=anchors2_data_test.shape[1], axis=1)
merged_test_input2 = Concatenate(axis=-1)([anchors2_data_test, histone_test_anchors2])
print(merged_test_input1.shape)



from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, Concatenate, MaxPooling1D, Add,LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomNormal

# 定义模型输入
input1 = Input(shape=(5000, 18))
input2 = Input(shape=(5000, 18))


x1= Conv1D(128, 8, padding='same', strides=1, activation='LeakyReLU')(input1)
x1 = MaxPooling1D(4)(x1)

x1= Conv1D(256, 8, padding='same', strides=1, activation='LeakyReLU')(x1)
x1 = MaxPooling1D(4)(x1)

x1= Conv1D(128, 8, padding='same', strides=1)(x1)
x1 = MaxPooling1D(4)(x1)

# print(input1.shape)

x2= Conv1D(128, 8, padding='same', strides=1, activation='LeakyReLU')(input2)
x2 = MaxPooling1D(4)(x2)

x2= Conv1D(256, 8, padding='same', strides=1, activation='LeakyReLU')(x2)
x2 = MaxPooling1D(4)(x2)


x2= Conv1D(128, 8, padding='same', strides=1,)(x2)
x2 = MaxPooling1D(4)(x2)
#定义一个weighted_sum
import tensorflow as tf
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Reshape

def weighted_sum(input1, input2):
    # 处理 input1
    half1 = input1.shape[1] // 2
    conv1 = Conv1D(filters=128, kernel_size=1, padding='valid')

    fw1 = conv1(input1[:, :half1, :])
    rc1 = conv1(tf.reverse(input1[:, half1:, :], [1]))
    x3 = tf.concat([fw1, rc1], axis=1)
    
    # 处理 input2
    half2 = input2.shape[1] // 2
    conv2 = Conv1D(filters=128, kernel_size=1, padding='valid')

    fw2 = conv2(input2[:, :half2, :])
    rc2 = conv2(tf.reverse(input2[:, half2:, :], [1]))
    x4 = tf.concat([fw2, rc2], axis=1)
    
    # 合并处理后的 input1 和 input2
    x = tf.concat([x3, x4], axis=-1)
    return x
#拼接经过weighted_sum处理后的特征
merged=weighted_sum(input1, input2)

dense = Dense(1024, activation='LeakyReLU')(merged)

flatten = Flatten()(dense)
output = Dense(1, activation='sigmoid')(flatten)


model2 = Model(inputs=[input1,input2], outputs=output)

model2.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
model2.summary()


# 评估模型
model2.load_weights('chinn_k562_SMC.h5')
score = model2.evaluate([merged_test_input1, merged_test_input2],
                        anchors2_label_test, verbose=0)




# ROC曲线
pred2 = model2.predict([merged_test_input1, merged_test_input2])
fpr2, tpr2, thresholds = roc_curve(anchors2_label_test, pred2)
roc_auc2 = auc(fpr2, tpr2)
print('AUC:', roc_auc2)

precision2, recall2, _ = precision_recall_curve(anchors2_label_test, pred2)
auprc2 = average_precision_score(anchors2_label_test, pred2)

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


# bigwig_files = ['IMR90-H3K4me1.fc.signal.bigwig','IMR90-H3K4me2.fc.signal.bigwig',
#                 'IMR90-H3K4me3.fc.signal.bigwig', 'IMR90-H3K9ac.fc.signal.bigwig',
#                 'IMR90-H3K9me3.fc.signal.bigwig', 'IMR90-H3K27ac.fc.signal.bigwig',
#                 'IMR90-H3K27me3.fc.signal.bigwig', 'IMR90-H3K36me3.fc.signal.bigwig',
#                 'IMR90-H3K79me2.fc.signal.bigwig', 'IMR90-H4K20me1.fc.signal.bigwig',
#                 'IMR90_Rad21.bigWig', 'IMR90_SMC3.bigWig', 'IMR90_CTCF.bigWig']

bigwig_files = ['K562-H3K4me1.fc.signal.bigwig','K562-H3K4me2.fc.signal.bigwig',
                'K562-H3K4me3.fc.signal.bigwig', 'K562-H3K9ac.fc.signal.bigwig',
                'K562-H3K9me3.fc.signal.bigwig', 'K562-H3K27ac.fc.signal.bigwig',
                'K562-H3K27me3.fc.signal.bigwig', 'K562-H3K36me3.fc.signal.bigwig',
                'K562-H3K79me2.fc.signal.bigwig', 'K562-H4K20me1.fc.signal.bigwig',
                'K562_Rad21.bigWig', 'K562_SMC3.bigWig', 'K562_CTCF.bigWig']
                             
bed_files_test_anchors1 = ['test_pos_1.bed', 'test_neg_1.bed']                           
bed_files_test_anchors2 = ['test_pos_2.bed', 'test_neg_2.bed']

histone_test_anchors1 = load_histone_modification_data(bed_files_test_anchors1, bigwig_files)
histone_test_anchors2 = load_histone_modification_data(bed_files_test_anchors2, bigwig_files)

merged_n_heads=9
merged_feed_forward_size=256
merged_encoder_stack=1

en_pool_size=15
pr_pool_size=10
en_strides=en_pool_size
pr_strides=pr_pool_size

en_kernal_size = 80
pr_kernal_size = 61

num_filters =72
model_dim = 100

def get_model():

    sequence_input1 = Input(shape=(5000, 4))
    sequence_input2 = Input(shape=(5000, 4))

   
    enhancer_conv_layer = Conv1D(filters=num_filters,
                                 kernel_size=en_kernal_size,
                                 padding="valid",
                                 activation='relu')(sequence_input1)

    enhancer_max_pool_layer = MaxPooling1D(pool_size=en_pool_size, strides=en_strides)(enhancer_conv_layer)

    promoter_conv_layer = Conv1D(filters=num_filters,
                                 kernel_size=pr_kernal_size,
                                 padding="valid",
                                 activation='relu')(sequence_input2)

    promoter_max_pool_layer = MaxPooling1D(pool_size=pr_pool_size, strides=pr_strides)(promoter_conv_layer)

    # merge
    merge1=Concatenate(axis=1)([enhancer_max_pool_layer, promoter_max_pool_layer])
    
    histone_input1 = Input(shape=(14,))
    histone_input2 = Input(shape=(14,))
    
    
    merge2=Concatenate(axis=1)([histone_input1, histone_input2])
    
    
    merge2 = Dense(822 * 28)(merge2)
    
    merge2 = Reshape((822, 28))(merge2)
    
    
    merge3 = Concatenate(axis=2)([merge1, merge2])
    
    

    bn=BatchNormalization()(merge3)

    dt=Dropout(0.5)(bn)
    model_dim = 100

    transformer1 = Transformer_Merged(encoder_stack=merged_encoder_stack,
                                feed_forward_size=merged_feed_forward_size,
                                n_heads=merged_n_heads,
                                model_dim=model_dim)

    trf = transformer1(dt)

    Gmaxpool = GlobalMaxPooling1D()(trf)

    merge4 = Dense(50)(Gmaxpool)

    bn2=BatchNormalization()(merge4)

    acti = Activation('relu')(bn2)

    preds = Dense(1, activation='sigmoid')(acti)

    model = Model([ sequence_input1,  sequence_input2,histone_input1,histone_input2], preds)

    # opt = tf.keras.optimizers.Nadam(learning_rate=0.01)

    # opt = tf.keras.optimizers.Nadam()

    # model.compile(loss='binary_crossentropy', optimizer=opt)

    return model


model3 = get_model()


opt = tf.keras.optimizers.Nadam(learning_rate=0.01)
model3.compile(loss='binary_crossentropy', optimizer=opt,
              metrics=['acc'])
model3.summary()

model3.load_weights('Trans_k562_SMC.h5')
score = model3.evaluate([anchors1_data_test, anchors2_data_test, histone_test_anchors1, histone_test_anchors2],
                        anchors2_label_test, verbose=0)

# ROC曲线
pred3 = model3.predict([anchors1_data_test, anchors2_data_test, histone_test_anchors1, histone_test_anchors2])
fpr3, tpr3, thresholds = roc_curve(anchors2_label_test, pred3)
roc_auc3 = auc(fpr3, tpr3)
print('AUC:', roc_auc3)

precision3, recall3, _ = precision_recall_curve(anchors2_label_test, pred3)
auprc3 = average_precision_score(anchors2_label_test, pred3)


def Enhancer_MDLF():
    sequence_input1 = Input(shape=(5000, 4))
    sequence_input2 = Input(shape=(5000, 4))

    # input_data_dna2vec = Input(shape=data_shape_dna2vec)
    x_dna2vec1 = Conv1D(16, kernel_size=9, strides=2, activation='relu', padding='same')(
        sequence_input1)
    x_dna2vec1 = MaxPooling1D(pool_size=2)(x_dna2vec1)
    x_dna2vec1 = Conv1D(16, kernel_size=9, strides=2, activation='relu', padding='same')(
        x_dna2vec1)
    x_dna2vec1 = MaxPooling1D(pool_size=2)(x_dna2vec1)
    x_dna2vec1 = Conv1D(16, kernel_size=9, strides=2, activation='relu', padding='same')(
        x_dna2vec1)
    x_dna2vec1 = MaxPooling1D(pool_size=2)(x_dna2vec1)
    x_dna2vec1 = Dropout(0.3)(x_dna2vec1)
    x_dna2vec1 = Flatten()(x_dna2vec1)
    x_dna2vec1 = Dense(500, activation='relu')(x_dna2vec1)
    
    x_dna2vec2 = Conv1D(16, kernel_size=9, strides=2, activation='relu', padding='same')(
        sequence_input2)
    x_dna2vec2 = MaxPooling1D(pool_size=2)(x_dna2vec2)
    x_dna2vec2 = Conv1D(16, kernel_size=9, strides=2, activation='relu', padding='same')(
        x_dna2vec2)
    x_dna2vec2 = MaxPooling1D(pool_size=2)(x_dna2vec2)
    x_dna2vec2 = Conv1D(16, kernel_size=9, strides=2, activation='relu', padding='same')(
        x_dna2vec2)
    x_dna2vec2 = MaxPooling1D(pool_size=2)(x_dna2vec2)
    x_dna2vec2 = Dropout(0.3)(x_dna2vec2)
    x_dna2vec2 = Flatten()(x_dna2vec2)
    x_dna2vec2 = Dense(500, activation='relu')(x_dna2vec2)
    merge1 = Concatenate(axis=1)([x_dna2vec1, x_dna2vec2])
    
    
    
    histone_input1 = Input(shape=(14,))
    histone_input2 = Input(shape=(14,))
    # input_data_motif = Input(shape=data_shape_motif)
    x_motif1 = Dense(128, activation='relu')( histone_input1)
    x_motif1 = Dense(64, activation='relu')(x_motif1)
    x_motif1 = Dense(16, activation='relu')(x_motif1)
    x_motif1 = Dropout(0.3)(x_motif1)
    
    x_motif2 = Dense(128, activation='relu')( histone_input2)
    x_motif2 = Dense(64, activation='relu')(x_motif2)
    x_motif2 = Dense(16, activation='relu')(x_motif2)
    x_motif2 = Dropout(0.3)(x_motif2)
    
    merge2 = Concatenate(axis=1)([x_motif1, x_motif2])
    
    
    

    merge3 = Concatenate(axis=1)([merge1, merge2])
    merge3 = Dropout(0.3)(merge3)
    output = Dense(1, activation='sigmoid')(merge3)
    model = Model(
        [sequence_input1, sequence_input2,histone_input1,histone_input2],
        output)
    print(model.summary())
    return model


model4 = Enhancer_MDLF()
model4.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
model4.summary()


model4.load_weights('MDLF_k562_SMC.h5')
score = model4.evaluate([anchors1_data_test, anchors2_data_test, histone_test_anchors1, histone_test_anchors2],
                        anchors2_label_test, verbose=0)

# ROC曲线
pred4 = model4.predict([anchors1_data_test, anchors2_data_test, histone_test_anchors1, histone_test_anchors2])
fpr4, tpr4, thresholds = roc_curve(anchors2_label_test, pred4)
roc_auc4 = auc(fpr4, tpr4)
print('AUC:', roc_auc4)

precision4, recall4, _ = precision_recall_curve(anchors2_label_test, pred4)
auprc4 = average_precision_score(anchors2_label_test, pred4)



KERNEL_NUMBER = 32
KERNEL_SIZE = 5
LSTM_UNITS = 32

def three_CNN_LSTM1():
    inputs1 = Input(shape=(5000, 4))
    inputs2 = Input(shape=(5000, 4))
    histone_input1 = Input(shape=(14,))
    histone_input2 = Input(shape=(14,))
    
    x1 = Conv1D(32, kernel_size=5, activation='relu')(inputs1)
    x1 = MaxPooling1D()(x1)
    x1 = Dropout(0.3)(x1)
    
    x1 = Conv1D(32, kernel_size=5, activation='relu')(x1)
    x1 = MaxPooling1D()(x1)
    x1 = Dropout(0.3)(x1)
    
    x1 = Conv1D(32, kernel_size=5, activation='relu')(x1)
    x1 = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2, recurrent_dropout=0.5))(x1)
    
    x2 = Conv1D(32, kernel_size=5, activation='relu')(inputs2)
    x2 = MaxPooling1D()(x2)
    x2 = Dropout(0.3)(x2)
    
    x2 = Conv1D(32, kernel_size=5, activation='relu')(x2)
    x2 = MaxPooling1D()(x2)
    x2 = Dropout(0.3)(x2)
    
    x2 = Conv1D(32, kernel_size=5, activation='relu')(x2)
    x2 = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2, recurrent_dropout=0.5))(x2)
    
    merge1=Concatenate(axis=1)([x1, x2])
    
    merge2=Concatenate(axis=1)([histone_input1, histone_input2])
    
    
    merge2 = Dense(2486 * 28)(merge2)
    
    merge2 = Reshape((2486, 28))(merge2)
    print(merge1.shape)
    print(merge2.shape)
    merge3 = Concatenate(axis=2)([merge1, merge2])
    
    
    x = Flatten()(merge3)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model([inputs1, inputs2,histone_input1,histone_input2], outputs)
    
    return model

model5 = three_CNN_LSTM1()

model5.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
model5.summary()


model5.load_weights('CLNNloop_k562_SMC.h5')
score5 = model5.evaluate([anchors1_data_test, anchors2_data_test, histone_test_anchors1, histone_test_anchors2],
                        anchors2_label_test, verbose=0)

# ROC曲线
pred5 = model5.predict([anchors1_data_test, anchors2_data_test, histone_test_anchors1, histone_test_anchors2])
fpr5, tpr5, thresholds = roc_curve(anchors2_label_test, pred5)
roc_auc5 = auc(fpr5, tpr5)
print('AUC:', roc_auc5)

precision5, recall5, _ = precision_recall_curve(anchors2_label_test, pred5)
auprc5 = average_precision_score(anchors2_label_test, pred5)


# 为每个模型分别保存ROC和PR曲线数据
models = ['UniChrom', 'CHINN', 'EPI_Trans', 'Enhancer_MDLF', 'CLNN_loop']
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