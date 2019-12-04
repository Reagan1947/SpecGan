# _SpecGan 基于Spec的Gan网络实现

_SpecGan 是一个基于频谱图的GAN网络结构

# 1.如何使用

## 1.1 配置config.py

通过config.py可配置：

- sample_rate
- n_windows: Windows size for FFT
- n_overlap: Overlap of window

## 1.2准备数据

**1). 创建CSV文件**

`python preapare_data.py create_csv --workspace ./workspace/ --speech_dir ./speech_wav/`

**2). 计算功率图谱**

`python preapare_data.py calculate_features --workspace ./workspace/ --speech_dir ./speech_wav/ --snr=0`

**3). 打包特征数据**

`python prepare_data.py pack_features --workspace=./workspace/ --snr=0 --n_concat=7 --n_hop=3`

## 1.3 GAN网络训练

## 1.4 Genrator与Dicriminator的提取

## 1.5 参考代码

_SpecGan基于以下项目开发: 

- Sednn: https://github.com/yongxuUSTC/sednn
- SpecGan: https://github.com/naotokui/SpecGAN
