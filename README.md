# _SpecGan 基于Spec的Gan网络实现

_SpecGan 是一个基于频谱图的GAN网络结构

# 1.如何使用

## 1.1 配置config.py

通过`config.py`可配置：

- sample_rate
- n_windows: Windows size for FFT
- n_overlap: Overlap of window

## 1.2准备数据

**1). 创建CSV文件**

`python preapare_data.py create_csv --workspace ./workspace/ --speech_dir ./speech_wav/`

**2). 计算功率图谱**

`python preapare_data.py calculate_features --workspace ./workspace/ --speech_dir ./speech_wav/ --snr=0`

**3). 打包特征数据**

`python prepare_data.py pack_features --workspace=./workspace/ --n_concat=7 --n_hop=3`

其中`n_concat`为语音切段数目，`n_hop`为跳帧 Ref:https://www.npmjs.com/package/frame-hop

**4). 计算scaler**

`python prepare_data.py compute_scaler --workspace=./workspace/`

## 1.3 GAN网络训练

`python train_specgan.py train --workpace=./workspace/`

## 1.4 Genrator与Dicriminator的提取

# 2 参考代码

_SpecGan基于以下项目开发: 

- Sednn: https://github.com/yongxuUSTC/sednn
- The implementation of Improved Wasserstein GAN proposed in https://arxiv.org/abs/1704.00028 was based on a keras implementation.

# 3 数据分析

感谢开源软件Sonicvisualiser提供了音频数据分析工具
- Sonicvisualiser: https://www.sonicvisualiser.org/
