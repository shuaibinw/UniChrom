# UniChrom: A Universal Deep Learning Architecture for Cross-Scale Chromatin Interaction Prediction
## 1. Clone the UniChrom repository:
git clone https://github.com/shuaibinw/UniChrom.git
<br>cd UniChrom
## 2. Install the required dependencies:
tensorflow>=2.4.0
<br>keras>=2.4.0
<br>pandas>=1.2.0
<br>numpy>=1.19.0
<br>scipy>=1.7.3
<br>pyBigWig >=0.3.22
<br>matplotlib>=3.3.0
<br>seaborn>=0.12.2
<br>plotly>=4.14.0
<br>scikit-learn>=1.0.2
<br>shap>=1.0.0
<br>tqdm>=4.66.4
<br>pyfaidx>=0.8.10.3
<br>seaborn>=0.12.2
<br>deeplift>=0.6.13.0
<br>h5py>=2.10.0




## 3. Run UniChrom:
python GM12878_code/GM12878_main.py
<br>python IMR90_code/IMR90_main.py
<br>python K562_code/K562_main.py
## predict:
eg:python  predict_Chrom_interaction.py chr1:15420-100420  chr1:12420-14020 GM12878
