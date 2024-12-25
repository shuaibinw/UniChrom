# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 10:57:55 2024

@author: 123
"""

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import pyBigWig
import pandas as pd
from tensorflow.keras.regularizers import l2
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization,Add,Reshape, Bidirectional, GRU, Input, Conv1D, MaxPooling1D, LSTM, Flatten, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import roc_curve, auc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import re
import time
from keras.layers import Bidirectional, LSTM, Attention
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

BIGWIG_FILES = {
    'GM12878': ['GM12878_H3K4me1-log2r.bw', 'GM12878_H3K4me2-log2r.bw',
               'GM12878_H3K4me3-log2r.bw', 'GM12878_H3K9ac-log2r.bw',
               'GM12878_H3K9me3-log2r.bw', 'GM12878_H3K27ac-log2r.bw',
               'GM12878_H3K27me3-log2r.bw', 'GM12878_H3K36me3-log2r.bw',
               'GM12878_H3K79me2-log2r.bw', 'GM12878_H4K20me1-log2r.bw',
               'GM12878_Rad21.bigWig', 'GM12878_SMC3.bigWig', 'GM12878_CTCF.bigWig'],
    'IMR90': ['IMR90-H3K4me1.fc.signal.bigwig', 'IMR90-H3K4me2.fc.signal.bigwig',
              'IMR90-H3K4me3.fc.signal.bigwig', 'IMR90-H3K9ac.fc.signal.bigwig',
              'IMR90-H3K9me3.fc.signal.bigwig', 'IMR90-H3K27ac.fc.signal.bigwig',
              'IMR90-H3K27me3.fc.signal.bigwig', 'IMR90-H3K36me3.fc.signal.bigwig',
              'IMR90-H3K79me2.fc.signal.bigwig', 'IMR90-H4K20me1.fc.signal.bigwig',
              'IMR90_Rad21.bigWig', 'IMR90_SMC3.bigWig', 'IMR90_CTCF.bigWig'],
    'K562': ['K562-H3K4me1.fc.signal.bigwig', 'K562-H3K4me2.fc.signal.bigwig',
             'K562-H3K4me3.fc.signal.bigwig', 'K562-H3K9ac.fc.signal.bigwig',
             'K562-H3K9me3.fc.signal.bigwig', 'K562-H3K27ac.fc.signal.bigwig',
             'K562-H3K27me3.fc.signal.bigwig', 'K562-H3K36me3.fc.signal.bigwig',
             'K562-H3K79me2.fc.signal.bigwig', 'K562-H4K20me1.fc.signal.bigwig',
             'K562_Rad21.bigWig', 'K562_SMC3.bigWig', 'K562_CTCF.bigWig']
}

def DNA_to_matrix(DNA, sequence_length=5000):
    data = np.zeros((1, sequence_length, 4), dtype='float32')
    for j in range(min(len(DNA), sequence_length)):
        if DNA[j] in ['A', 'a']:
            data[0, j, :] = np.array([1.0, 0.0, 0.0, 0.0], dtype='float32')
        elif DNA[j] in ['C', 'c']:
            data[0, j, :] = np.array([0.0, 1.0, 0.0, 0.0], dtype='float32')
        elif DNA[j] in ['G', 'g']:
            data[0, j, :] = np.array([0.0, 0.0, 1.0, 0.0], dtype='float32')
        elif DNA[j] in ['T', 't']:
            data[0, j, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype='float32')
    return data

def extract_sequence_from_chrom_range(fasta_file, chrom_range):
    chrom, start_end = chrom_range.split(':')
    start, end = [int(x) for x in start_end.split('-')]
    target_length = end - start + 1

    with open(fasta_file, "r") as file:
        sequence = ""
        found_chrom = False
        for line in file:
            if line.startswith('>'):
                if found_chrom:
                    break
                if line.strip('>\n') == chrom:
                    found_chrom = True
            elif found_chrom:
                sequence += line.strip()
                if len(sequence) >= end:
                    sequence = sequence[start-1:end]
                    break

    if len(sequence) < target_length:
        raise ValueError(f"Sequence not found within the range {chrom_range} in the FASTA file.")

    return sequence

def load_histone_modification_data(bw_files, chrom_range):
    chrom, start_end = chrom_range.split(':')
    start, end = [int(x) for x in start_end.split('-')]

    histone_data = []
    for bw_file in bw_files:
        bw = pyBigWig.open(bw_file)
        value = bw.stats(chrom, start, end, exact=True)[0]
        histone_data.append(value if value is not None else 0)
        bw.close()

    loop_distance = abs(end - start)
    histone_data.append(loop_distance)

    return np.array(histone_data)



def create_model():
    
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

    model = Model(inputs=[sequence_input1, sequence_input2, histone_input1, histone_input2], outputs=output)

    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    return model




# def main(chrom_range1, chrom_range2, cell_line='IMR90'):
#     bigwig_files = BIGWIG_FILES.get(cell_line)
#     if bigwig_files is None:
#         raise ValueError(f"Unknown cell line '{cell_line}'. Please use one of the following: {', '.join(BIGWIG_FILES.keys())}")

#     sequence1 = extract_sequence_from_chrom_range("hg19.fa", chrom_range1)
#     sequence2 = extract_sequence_from_chrom_range("hg19.fa", chrom_range2)

#     histone_data1 = load_histone_modification_data(bigwig_files, chrom_range1)
#     histone_data2 = load_histone_modification_data(bigwig_files, chrom_range2)

#     DNA_data1 = DNA_to_matrix(sequence1)
#     DNA_data2 = DNA_to_matrix(sequence2)

#     model = create_model()

#     model.load_weights('new_gm12878_SMC.h5')
    
#     probability = model.predict([DNA_data1, DNA_data2, histone_data1.reshape(1, -1), histone_data2.reshape(1, -1)])[0][0]

#     print(f"Predicted probability of interaction between the sequences: {probability:.4f}")

# if __name__ == "__main__":
#     if len(sys.argv) != 4:
#         print("Usage: python predict.py 'chrom_range1' 'chrom_range2' 'cell_line'")
#         sys.exit(1)

#     chrom_range1 = sys.argv[1]
#     chrom_range2 = sys.argv[2]
#     cell_line = sys.argv[3]
#     main(chrom_range1, chrom_range2, cell_line)
def main(chrom_range1, chrom_range2, cell_line='IMR90'):
    bigwig_files = BIGWIG_FILES.get(cell_line)
    if bigwig_files is None:
        raise ValueError(f"Unknown cell line '{cell_line}'. Please use one of the following: {', '.join(BIGWIG_FILES.keys())}")

    sequence1 = extract_sequence_from_chrom_range("hg19.fa", chrom_range1)
    sequence2 = extract_sequence_from_chrom_range("hg19.fa", chrom_range2)

    histone_data1 = load_histone_modification_data(bigwig_files, chrom_range1)
    histone_data2 = load_histone_modification_data(bigwig_files, chrom_range2)

    DNA_data1 = DNA_to_matrix(sequence1)
    DNA_data2 = DNA_to_matrix(sequence2)

    model = create_model()
    
    model.load_weights(f'new_{cell_line}_SMC.h5')
    
    probability = model.predict([DNA_data1, DNA_data2, histone_data1.reshape(1, -1), histone_data2.reshape(1, -1)])[0][0]

    print(f"Predicted probability of interaction between the sequences: {probability:.4f}")
    
    
    
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python predict.py 'chrom_range1' 'chrom_range2' 'cell_line'")
        sys.exit(1)

    chrom_range1 = sys.argv[1]
    chrom_range2 = sys.argv[2]
    cell_line = sys.argv[3]
    main(chrom_range1, chrom_range2, cell_line)