# UniChrom: A Universal Deep Learning Architecture for Cross-Scale Chromatin Interaction Prediction
## 1. Clone the TransSE repository:
git clone https://github.com/shuaibinw/UniChrom.git
<br>cd TransSE
## 2. Install the required dependencies:
tensorflow>=2.4.0
<br>keras>=2.4.0
<br>pandas>=1.2.0
<br>numpy>=1.19.0
<br>scipy>=1.5.0
<br>biopython>=1.78
<br>matplotlib>=3.3.0
<br>seaborn>=0.11.0
<br>plotly>=4.14.0
<br>scikit-learn>=0.24.0
<br>jupyter>=1.0.0
<br>tqdm>=4.50.0
## 3. Run UniChrom:
python GM12878_code/GM12878_main.py
<br>python IMR90_code/IMR90_main.py
<br>python K562_code/K562_main.py
## predict:
eg:python  predict_Chrom_interaction.py chr1:15420-100420  chr1:12420-14020 GM12878
