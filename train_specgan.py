import numpy as np
import os
import argparse
import time
import prepare_data as pp_data
import pickle


def train(args):
    workspace = args.workspace

    # Load data.
    t1 = time.time()
    tr_hdf5_path = os.path.join(workspace, "packed_features", "spectrogram", "data.h5")
    tr_y = pp_data.load_hdf5(tr_hdf5_path)
    tr_y = np.array(tr_y)[1]
    print("data shape is {}".format(tr_y.shape))
    print("Load data time: %s s" % (time.time() - t1,))

    # scaler data
    t1 = time.time()
    scaler_path = os.path.join(workspace, "packed_features", "spectrogram", "scaler.p")
    scaler = pickle.load(open(scaler_path, 'rb'))
    tr_y = pp_data.scale_on_2d(tr_y, scaler)
    print("Scale data time: %s s" % (time.time() - t1,))

    # batch
    batch_size = 500
    print("%d iterations / epoch" % (tr_y.shape[0] / batch_size))

    # data shape and input shape
    (n_segs, n_freq) = tr_y.shape


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--workspace', type=str, required=True)

    parser_inference = subparsers.add_parser('inference')

    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    else:
        raise Exception("Error!")
