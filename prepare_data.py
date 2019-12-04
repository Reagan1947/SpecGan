# coding=utf-8
import config as cfg
import numpy as np
from scipy import signal
import time
import soundfile
import librosa
import os
import csv
import pickle
import h5py
import argparse


def load_hdf5(hdf5_path):
    """Load hdf5 data.
    """
    with h5py.File(hdf5_path, 'r') as hf:
        x = hf.get('x')
        y = hf.get('y')
        x = np.array(x)     # (n_segs, n_concat, n_freq)
        y = np.array(y)     # (n_segs, n_freq)
    return x, y


def mat_2d_to_3d(x, agg_num, hop):
    # Pad to at least one block.
    len_x, n_in = x.shape
    if len_x < agg_num:
        x = np.concatenate((x, np.zeros((agg_num - len_x, n_in))))

    # Segment 2d to 3d.
    len_x = len(x)
    i1 = 0
    x3d = []
    while i1 + agg_num <= len_x:
        x3d.append(x[i1: i1 + agg_num])
        i1 += hop
    return np.array(x3d)


def log_sp(x):
    return np.log(x + 1e-08)


def pad_with_border(x, n_pad):
    n_pad = int(n_pad)
    x_pad_list = [x[0:1]] * n_pad + [x] + [x[-1:]] * n_pad
    return np.concatenate(x_pad_list, axis=0)


def pack_features(args):
    workspace = args.workspace
    snr = args.snr
    n_concat = args.n_concat
    n_hop = args.n_hop

    y_all = []  # (n_segs, n_freq)

    cnt = 0
    t1 = time.time()

    # Load all features.
    feat_dir = os.path.join(workspace, "features", "spectrogram")
    names = os.listdir(feat_dir)
    for na in names:
        # Load feature.
        feat_path = os.path.join(feat_dir, na)
        data = pickle.load(open(feat_path, 'rb'))
        [speech_x, na] = data

        # Pad start and finish of the spectrogram with boarder values.
        n_pad = (n_concat - 1) / 2
        speech_x = pad_with_border(speech_x, n_pad)

        # Cut target spectrogram and take the center frame of each 3D segment.
        speech_x_3d = mat_2d_to_3d(speech_x, agg_num=n_concat, hop=n_hop)
        y = speech_x_3d[:, int((n_concat - 1) / 2), :]
        y_all.append(y)

        # Print.
        # if cnt % 100 == 0:
        #     print(cnt)

        # if cnt == 3: break
        cnt += 1

    y_all = np.concatenate(y_all, axis=0)  # (n_segs, n_freq)

    y_all = log_sp(y_all).astype(np.float32)

    # Write out data to .h5 file.
    out_path = os.path.join(workspace, "packed_features", "spectrogram", "data.h5")
    create_folder(os.path.dirname(out_path))
    with h5py.File(out_path, 'w') as hf:
        hf.create_dataset('y', data=y_all)

    print("Write out to %s" % out_path)
    print("Pack features finished! %s s" % (time.time() - t1,))


# 创文件夹方法
def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)


# fs为采样率
def create_csv(args):
    workspace = args.workspace
    speech_dir = args.speech_dir

    # speech_names
    speech_names = [na for na in os.listdir(speech_dir) if na.lower().endswith(".wav")]

    # csv文件输出地址
    out_csv_path = os.path.join(workspace, "speech_csvs", "speech_data.csv")
    create_folder(os.path.dirname(out_csv_path))

    cnt = 0
    f = open(out_csv_path, 'w')
    f.write("%s\n" % "speech_name")
    for speech_na in speech_names:
        # Read speech.
        speech_path = os.path.join(speech_dir, speech_na)
        (speech_audio, _) = read_audio(speech_path)

        cnt += 1
        f.write("%s\n" % speech_na)
    f.close()
    print('specified directory is {}'.format(out_csv_path))

    # 输出在指定路径中读取了..文件
    print('{} were read in in specified directory'.format(cnt))


# 对路径给出文件计算其每个文件的功率图谱
def calculate_features(args):
    workspace = args.workspace
    speech_dir = args.speech_dir
    snr = args.snr
    fs = cfg.sample_rate

    # 打开语音的csv列表
    csv_path = os.path.join(workspace, "speech_csvs", "speech_data.csv")
    with open(csv_path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        lis = list(reader)

    t1 = time.time()
    cnt = 0
    for i1 in range(1, len(lis)):
        [speech_na] = lis[i1]

        # 读取audio
        speech_path = os.path.join(speech_dir, speech_na)
        (speech_audio, _) = read_audio(speech_path, target_fs=fs)

        # 计算功率图谱
        speech_x = calc_sp(speech_audio)

        # 导出feature
        out_speech_name = os.path.splitext(speech_na)[0]
        out_feat_path = os.path.join(workspace, "features", "spectrogram", "%s.p" % out_speech_name)

        create_folder(os.path.dirname(out_feat_path))
        data = [speech_x, out_speech_name]
        pickle.dump(data, open(out_feat_path, 'wb'))

        # Print.
        cnt += 1

    print("{} speech data were calculated feature".format(cnt))
    print("Extracting feature time: %s" % (time.time() - t1))


# 读取音频文件
def read_audio(path, target_fs=None):
    (audio, fs) = soundfile.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
    return audio, fs


# 计算功率图谱
def calc_sp(audio):
    """Calculate spectrogram.
    Args:
      audio: 1 Dynamic Array
    Returns:
      spectrogram: 2 Dynamic Array, (n_time, n_freq).
    """
    n_window = cfg.n_window
    n_overlap = cfg.n_overlap
    ham_win = np.hamming(n_window)
    [_, _, x] = signal.spectral.spectrogram(
                    audio,
                    window=ham_win,
                    nperseg=n_window,
                    noverlap=n_overlap,
                    detrend=False,
                    return_onesided=True,
    )
    x = x.T
    x = x.astype(np.float32)
    return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_create_mixture_csv = subparsers.add_parser('create_csv')
    parser_create_mixture_csv.add_argument('--workspace', type=str, required=True)
    parser_create_mixture_csv.add_argument('--speech_dir', type=str, required=True)

    parser_calculate_mixture_features = subparsers.add_parser('calculate_features')
    parser_calculate_mixture_features.add_argument('--workspace', type=str, required=True)
    parser_calculate_mixture_features.add_argument('--speech_dir', type=str, required=True)
    parser_calculate_mixture_features.add_argument('--snr', type=float, required=True)

    parser_pack_features = subparsers.add_parser('pack_features')
    parser_pack_features.add_argument('--workspace', type=str, required=True)
    parser_pack_features.add_argument('--snr', type=float, required=True)
    parser_pack_features.add_argument('--n_concat', type=int, required=True)
    parser_pack_features.add_argument('--n_hop', type=int, required=True)

    args = parser.parse_args()
    if args.mode == 'create_csv':
        create_csv(args)
    elif args.mode == 'calculate_features':
        calculate_features(args)
    elif args.mode == 'pack_features':
        pack_features(args)
    else:
        raise Exception("Error! Please check your argument.")


# n_concat * 7 是一组
# n_concat * n_freq 是一组
# 目标就只有  n_seq, n_freq
# 所以数据集合就是 1123 个 257的矩阵
