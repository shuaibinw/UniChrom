import keras
from tensorflow.python.ops.gen_math_ops import select
import deeplift
from deeplift.conversion import kerasapi_conversion as kc
from deeplift.layers import NonlinearMxtsMode
from deeplift.util import get_shuffle_seq_ref_function
from deeplift.dinuc_shuffle import dinuc_shuffle 
from tensorflow.keras.models import load_model
import random 
import seaborn as sns
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from keras.models import model_from_json
import h5py
from scipy.interpolate import make_interp_spline
from matplotlib.colors import LinearSegmentedColormap

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


import pandas as pd
def plot_a(ax, base, left_edge, height, color):
    a_polygon_coords = [
        np.array([
        [0.0, 0.0],
        [0.5, 1.0],
        [0.5, 0.8],
        [0.2, 0.0],
        ]),
        np.array([
        [1.0, 0.0],
        [0.5, 1.0],
        [0.5, 0.8],
        [0.8, 0.0],
        ]),
        np.array([
        [0.225, 0.45],
        [0.775, 0.45],
        [0.85, 0.3],
        [0.15, 0.3],
        ])
    ]
    for polygon_coords in a_polygon_coords:
        ax.add_patch(matplotlib.patches.Polygon((np.array([1,height])[None,:]*polygon_coords
                                                + np.array([left_edge,base])[None,:]),
                                                facecolor=color, edgecolor=color))

def plot_c(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=1.3, height=height,
                                            facecolor=color, edgecolor=color))
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=0.7*1.3, height=0.7*height,
                                            facecolor='white', edgecolor='white'))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+1, base], width=1.0, height=height,
                                            facecolor='white', edgecolor='white', fill=True))

def plot_g(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=1.3, height=height,
                                            facecolor=color, edgecolor=color))
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=0.7*1.3, height=0.7*height,
                                            facecolor='white', edgecolor='white'))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+1, base], width=1.0, height=height,
                                            facecolor='white', edgecolor='white', fill=True))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+0.825, base+0.085*height], width=0.174, height=0.415*height,
                                            facecolor=color, edgecolor=color, fill=True))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+0.625, base+0.35*height], width=0.374, height=0.15*height,
                                            facecolor=color, edgecolor=color, fill=True))

def plot_t(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+0.4, base],
                width=0.2, height=height, facecolor=color, edgecolor=color, fill=True))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge, base+0.8*height],
                width=1.0, height=0.2*height, facecolor=color, edgecolor=color, fill=True))

default_colors = {0:'green', 1:'blue', 2:'orange', 3:'red'}
default_plot_funcs = {0:plot_a, 1:plot_c, 2:plot_g, 3:plot_t}

def plot_weights_given_ax(ax, array,
                height_padding_factor,
                length_padding,
                subticks_frequency,
                highlight,
                colors=default_colors,
                plot_funcs=default_plot_funcs):
    if len(array.shape)==3:
        array = np.squeeze(array)
    assert len(array.shape)==2, array.shape
    if (array.shape[0]==4 and array.shape[1] != 4):
        array = array.transpose(1,0)
    assert array.shape[1]==4
    max_pos_height = 0.0
    min_neg_height = 0.0
    heights_at_positions = []
    depths_at_positions = []
    for i in range(array.shape[0]):
        acgt_vals = sorted(enumerate(array[i,:]), key=lambda x: abs(x[1]))
        positive_height_so_far = 0.0
        negative_height_so_far = 0.0
        for letter in acgt_vals:
            plot_func = plot_funcs[letter[0]]
            color=colors[letter[0]]
            if (letter[1] > 0):
                height_so_far = positive_height_so_far
                positive_height_so_far += letter[1]                
            else:
                height_so_far = negative_height_so_far
                negative_height_so_far += letter[1]
            plot_func(ax=ax, base=height_so_far, left_edge=i, height=letter[1], color=color)
        max_pos_height = max(max_pos_height, positive_height_so_far)
        min_neg_height = min(min_neg_height, negative_height_so_far)
        heights_at_positions.append(positive_height_so_far)
        depths_at_positions.append(negative_height_so_far)

    for color in highlight:
        for start_pos, end_pos in highlight[color]:
            assert start_pos >= 0.0 and end_pos <= array.shape[0]
            min_depth = np.min(depths_at_positions[start_pos:end_pos])
            max_height = np.max(heights_at_positions[start_pos:end_pos])
            ax.add_patch(
                matplotlib.patches.Rectangle(xy=[start_pos,min_depth],
                    width=end_pos-start_pos,
                    height=max_height-min_depth,
                    edgecolor=color, fill=False))
        
    ax.set_xlim(-length_padding, array.shape[0]+length_padding)
    ax.xaxis.set_ticks(np.arange(0.0, array.shape[0]+1, subticks_frequency))
    height_padding = max(abs(min_neg_height)*(height_padding_factor),
                        abs(max_pos_height)*(height_padding_factor))
    ax.set_ylim(min_neg_height-height_padding, max_pos_height+height_padding)
    





def plot_weights(array, idx,
                  figsize=(15, 2),
                  height_padding_factor=0.2,
                  length_padding=1.0,
                  subticks_frequency=1.0,
                  colors=default_colors,
                  plot_funcs=default_plot_funcs,
                  highlight={}):
    # 确保每个元素都是 NumPy 数组
    array = [np.array(arr) if isinstance(arr, list) else arr for arr in array]

    # 定义每个片段的长度
    segment_length = 156

    # 创建输出文件夹
    output_dir = './silico_output'
    os.makedirs(output_dir, exist_ok=True)
    
   

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    ax.set_xlim(0, segment_length)
    ax.set_ylabel('In-silico\ntiling deletion')
    
    if len(array[1][0:segment_length]) == segment_length:
        # 创建x坐标和y坐标
        x = np.arange(segment_length)
        y = array[1][0:segment_length]
        
        # 创建平滑曲线
        x_smooth = np.linspace(0, segment_length-1, segment_length * 5)
        spl = make_interp_spline(x, y, k=3)
        y_smooth = spl(x_smooth)
        
        # 创建颜色映射
        colors_positive = ['#FFFFFF', '#C8D9F9', '#A0C7F2', '#1A61D5']  # 白色到蓝色
        colors_negative = ['#FFFFFF', '#F8C7F9', '#F1A7F6', '#FF9CF5']  # 白色到粉色
        
        cmap_positive = LinearSegmentedColormap.from_list("custom_positive", colors_positive, N=200)
        cmap_negative = LinearSegmentedColormap.from_list("custom_negative", colors_negative, N=200)
        
        # 获取最大值用于归一化
        max_positive = max(max(y_smooth), 0)
        max_negative = abs(min(min(y_smooth), 0))
        
        # 为每个点创建渐变填充
        for i in range(len(x_smooth)-1):
            if y_smooth[i] >= 0:
                color_intensity = y_smooth[i] / max_positive if max_positive != 0 else 0
                color = cmap_positive(color_intensity)
            else:
                color_intensity = abs(y_smooth[i]) / max_negative if max_negative != 0 else 0
                color = cmap_negative(color_intensity)
                
            ax.fill_between(x_smooth[i:i+2], y_smooth[i:i+2], 0,
                          color=color,
                          alpha=0.8)
            
        # 绘制平滑的上边界线
        ax.plot(x_smooth, y_smooth, 
                color='k',
                linewidth=0.5,
                alpha=0.8)
        
    else:
        ax.text(0.5, 0.5, 'Data length mismatch', 
                ha='center', 
                va='center', 
                transform=ax.transAxes)
    
    # 设置x轴刻度
    xtick_positions = np.arange(0, segment_length, 10)
    xtick_labels = ["{:.0f}".format(pos) for pos in xtick_positions]
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels)
    
    # 移除上边框和右边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 添加水平参考线
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # 减少周围空白
    fig.tight_layout()
    

    
    fig.savefig(f'{output_dir}/sequence_{idx}_tiling_deletion_ridge_gradient.pdf', 
                format='pdf', 
                bbox_inches='tight', 
                dpi=300)
    plt.close(fig)

   

    
    
    
    # 2计算需要显示的刻度值
    xtick_positions = np.arange(0, segment_length, 10)  # 包含最后一个刻度的位置
    xtick_labels = ["{:.0f}".format(pos) for pos in xtick_positions]  # 根据刻度位置生成标签

    cmap_custom = LinearSegmentedColormap.from_list(
    'custom_cmap', 
    ['#1A61D5', '#A0C7F2', '#C8D9F9', '#FFFFFF', '#FFFFFF', '#F8C7F9', '#F1A7F6', '#FF9CF5']
    )

    fig3 = plt.figure(figsize=figsize)
    ax3 = fig3.add_subplot(111)
    sns.heatmap(array[2][:,:segment_length],
        cmap=cmap_custom,  
        cbar_kws={'location': 'top', 'fraction': 0.1},
        ax=ax3)
    ax3.set_ylabel('In-silico\nmutagenesis')
    ax3.set_xticks(xtick_positions)  
    ax3.set_xticklabels(xtick_labels)

    # 修改y轴标签
    ax3.set_yticks([0.5, 1.5, 2.5, 3.5])  # 设置刻度位置在每个方格中间
    ax3.set_yticklabels(['A', 'C', 'G', 'T'])  # 设置新的标签

    # 减少周围空白
    fig3.tight_layout()
    fig3.savefig(f'{output_dir}/sequence_{idx}_mutagenesis_segment.pdf', 
             format='pdf', 
             bbox_inches='tight')
    plt.close(fig3)


    
    
    
    

        

def one_hot_encode_along_channel_axis(sequence):
    to_return = np.zeros((len(sequence),4), dtype=np.int8)
    seq_to_one_hot_fill_in_array(zeros_array=to_return,
                                sequence=sequence, one_hot_axis=1)
    return to_return


def seq_to_one_hot_fill_in_array(zeros_array, sequence, one_hot_axis):
    assert one_hot_axis == 0 or one_hot_axis == 1
    if one_hot_axis == 0:
        assert zeros_array.shape[1] == len(sequence)
    elif one_hot_axis == 1:
        assert zeros_array.shape[0] == len(sequence)  
    for (i, char) in enumerate(sequence):
        if char in "Aa":
            char_idx = 0
        elif char in "Cc":
            char_idx = 1
        elif char in "Gg":
            char_idx = 2
        elif char in "Tt":
            char_idx = 3
        elif char in "Nn":
            continue
        else:
            raise RuntimeError("Unsupported character: " + str(char))
        if one_hot_axis == 0:
            zeros_array[char_idx, i] = 1
        elif one_hot_axis == 1:
            zeros_array[i, char_idx] = 1


def population_mutator(population_current, sequence_length):
    population_next = []  
    for i in range(len(population_current)):         
        for j in range(sequence_length):
            population_next.append(list(population_current[i]))
            population_next.append(list(population_current[i]))
            population_next.append(list(population_current[i]))
            population_next.append(list(population_current[i]))

            if (population_current[i][j] == 'A'):
                population_next[4*(sequence_length*i + j)][j] = 'A'
                population_next[4*(sequence_length*i + j) + 1][j] = 'C'
                population_next[4*(sequence_length*i + j) + 2][j] = 'G'
                population_next[4*(sequence_length*i + j) + 3][j] = 'T'
            elif (population_current[i][j] == 'C'):
                population_next[4*(sequence_length*i + j)][j] = 'A'
                population_next[4*(sequence_length*i + j) + 1][j] = 'C'
                population_next[4*(sequence_length*i + j) + 2][j] = 'G'
                population_next[4*(sequence_length*i + j) + 3][j] = 'T'
            elif (population_current[i][j] == 'G'):
                population_next[4*(sequence_length*i + j)][j] = 'A'
                population_next[4*(sequence_length*i + j) + 1][j] = 'C'
                population_next[4*(sequence_length*i + j) + 2][j] = 'G'
                population_next[4*(sequence_length*i + j) + 3][j] = 'T'
            elif (population_current[i][j] == 'T'):
                population_next[4*(sequence_length*i + j)][j] = 'A'
                population_next[4*(sequence_length*i + j) + 1][j] = 'C'
                population_next[4*(sequence_length*i + j) + 2][j] = 'G'
                population_next[4*(sequence_length*i + j) + 3][j] = 'T'
    return list(population_next)



def old_seq2feature(data):
    # 定义一热编码向量
    A_onehot = np.array([1, 0, 0, 0], dtype='float32')
    C_onehot = np.array([0, 1, 0, 0], dtype='float32')
    G_onehot = np.array([0, 0, 1, 0], dtype='float32')
    T_onehot = np.array([0, 0, 0, 1], dtype='float32')
    N_onehot = np.array([0, 0, 0, 0], dtype='float32')

    # 创建映射字典
    mapper = {
     'A': A_onehot, 'a': A_onehot,
     'C': C_onehot, 'c': C_onehot,
     'G': G_onehot, 'g': G_onehot,
     'T': T_onehot, 't': T_onehot,
     'N': N_onehot, 'n': N_onehot
 }
    
    transformed = np.asarray(([[mapper[k] for k in (data[i])] for i in (range(len(data)))]))
    return transformed




def calculate(sample_path):
    # 读取fasta文件
    with open(sample_path) as f:
        data = []
        while True:
            line = f.readline()
            if not line:
                break
            if not line[0] == '>':
                # 确保序列长度为5000，不足的用N填充
                sequence = line.strip()
                if len(sequence) < 5000:
                    sequence = sequence + 'N' * (5000 - len(sequence))
                elif len(sequence) > 5000:
                    sequence = sequence[:5000]
                data.append(sequence)


    # 模型文件路径
    keras_model_weights = "gm_SQ.h5"
    keras_model_json = "gm_SQ.json"

    def load_model_workaround(json_path, weights_path):
        with open(json_path, 'r') as f:
            model = model_from_json(f.read())
        weights_file = h5py.File(weights_path, 'r')
        for layer in model.layers:
            if layer.name in weights_file:
                layer_weights = []
                weight_names = [n.decode('utf8') if isinstance(n, bytes) else n 
                              for n in weights_file[layer.name].attrs['weight_names']]
                for weight_name in weight_names:
                    weight_value = weights_file[layer.name][weight_name][...]
                    layer_weights.append(weight_value)
                layer.set_weights(layer_weights)
        weights_file.close()
        return model

    # 加载模型
    try:
        print("Attempting to load model...")
        keras_model = load_model_workaround(keras_model_json, keras_model_weights)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
        
    # 使用固定的模型路径
    selected_path = 'silico.hdf5'  
    deeplift_model = kc.convert_model_from_saved_files(
        selected_path,
        nonlinear_mxts_mode=deeplift.layers.NonlinearMxtsMode.DeepLIFT_GenomicsDefault)
    l = list(deeplift_model.get_name_to_layer().keys())
    
    
 

    deeplift_contribs_func = deeplift_model.get_target_contribs_func(
        find_scores_layer_name=l[0],
        pre_activation_target_layer_name=l[-2])

    rescale_conv_revealcancel_fc_many_refs_func = get_shuffle_seq_ref_function(
        score_computation_function=deeplift_contribs_func,
        shuffle_func=dinuc_shuffle,
        one_hot_func=lambda x: np.array([one_hot_encode_along_channel_axis(seq) for seq in x]))


    

    num_refs_per_seq = 10

    scores_without_sum_applied = rescale_conv_revealcancel_fc_many_refs_func(
        task_idx=0,
        input_data_sequences=data,
        num_refs_per_seq=num_refs_per_seq,
        batch_size=200,
        progress_update=None)
    
    scores = np.sum(scores_without_sum_applied,axis=2)
    
    # 创建输出目录
    output_dir = './deeplift_score_new'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    onehot_data = np.array([one_hot_encode_along_channel_axis(seq) for seq in data])
    # model = load_model(selected_path)
    model=load_model_workaround(keras_model_json, keras_model_weights)
    
   
    onehot_data = np.array([one_hot_encode_along_channel_axis(seq) for seq in data])
    
    
    
    
    
    for idx in range(len(data)):
        print("sequence", idx)
        scores_for_idx = scores[idx]
        original_onehot = onehot_data[idx]
        scores_for_idx = original_onehot * scores_for_idx[:, None]
    
        predict_probability = model.predict(onehot_data)[0][0]
        preds = []
        segment_length=155
        # 修改滑动窗口的范围以适应5000bp长度
        for i in range(0, 5000-10+1, 1):
            temp = np.vstack((onehot_data[idx][:i,:], onehot_data[idx][i+10:,:]))
            seq = np.pad(temp, ((0,10), (0,0)), 'constant', constant_values=(0,0))[np.newaxis,:,:]
            predict = model.predict(seq)[0]
            preds.append(predict[0])
            
            # 计算每个删除窗口的预测概率与原始预测概率的差异
            diff_deletion = [i - predict_probability for i in preds]
            
            # 修改突变计算以适应5000bp长度
            population_1bp_all_sequences = population_mutator([data[idx]], 5000)
            population_1bp_all_feature = old_seq2feature(population_1bp_all_sequences)
            population_1bp_fitness = model.predict(population_1bp_all_feature)
            diff_mutation_o = population_1bp_fitness - predict_probability
            diff_mutation = np.reshape(diff_mutation_o, [5000, 4]).T
            
            # 假设要保存的长度是segment_length
            # 保存diff_deletion到CSV
            diff_deletion_segment = diff_deletion[:segment_length]  # 只取指定长度
            df_deletion = pd.DataFrame({
                'position': range(len(diff_deletion_segment)),
                'deletion_effect': diff_deletion_segment
                })
            df_deletion.to_csv(f'{output_dir}/new_sequence_{idx}_deletion_effects.csv', index=False)

            # # 保存diff_mutation到CSV
            # diff_mutation_segment = diff_mutation[:, :segment_length]  # 只取指定长度
            # df_mutation = pd.DataFrame(
            #     diff_mutation_segment,
            #     index=['A', 'C', 'G', 'T'],
            #     columns=[f'pos_{i}' for i in range(segment_length)]
            #     )
            # df_mutation.to_csv(f'{output_dir}/sequence_{idx}_mutation_effects.csv')
            
            # # 绘图部分保持不变
            # plot_weights([scores_for_idx, diff_deletion, diff_mutation], 
            #               idx=idx,
            #               subticks_frequency=10,  
            #               figsize=(15,2))
            
            
            # # 计算每个删除窗口的预测概率与原始预测概率的差异
            # diff_deletion = [i - predict_probability for i in preds]

            # # # 修改突变计算以适应5000bp长度
            # # population_1bp_all_sequences = population_mutator([data[idx]], 5000)
            # # population_1bp_all_feature = old_seq2feature(population_1bp_all_sequences)
            # # population_1bp_fitness = model.predict(population_1bp_all_feature)
            # # diff_mutation_o = population_1bp_fitness - predict_probability
            # # diff_mutation = np.reshape(diff_mutation_o, [5000, 4]).T
    
            # # 假设要保存的长度是segment_length
            # # 保存diff_deletion到CSV，仅保存索引范围 119 到 156
            # start_index = 119
            # end_index = 155
            # diff_deletion_segment = diff_deletion[start_index:end_index]  # 截取指定范围
            # df_deletion = pd.DataFrame({
            #     'position': range(start_index, end_index),
            #     'deletion_effect': diff_deletion_segment
            #     })
            # df_deletion.to_csv(f'{output_dir}/sequence_{idx}_deletion_effects.csv', index=False)

    
 

        
        
    
    

def main():
    sample_path = sys.argv[1] 
    calculate(sample_path)
    
if __name__ == "__main__":
    main()
    





