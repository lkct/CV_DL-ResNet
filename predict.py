import argparse
import logging

import mxnet as mx
import numpy as np
import pandas as pd

from symbol_resnet import resnet

logger = logging.getLogger()
logger.setLevel(logging.INFO)

data_type = 'cifar10'


def main():
    kv = mx.kvstore.create(args.kv_store)
    devs = mx.cpu() if args.gpus is None else [
        mx.gpu(int(i)) for i in args.gpus.split(',')]
    model_prefix = "model/resnet-{}-{}-{}".format(
        data_type, args.depth, kv.rank)
    symbol, arg_params, aux_params = mx.model.load_checkpoint(
        model_prefix, args.model_load_epoch)
    if args.memonger:
        import memonger
        symbol = memonger.search_plan(
            symbol, data=(args.batch_size, 3, 32, 32))
    val = mx.io.ImageRecordIter(
        path_imgrec         = args.rec_file,
        label_width         = 1,
        data_shape          = (3, 32, 32),
        num_parts           = kv.num_workers,
        part_index          = kv.rank,
        batch_size          = args.batch_size,
    )
    model = mx.mod.Module(
        symbol              = symbol,
        data_names          = ('data', ),
        label_names         = ('softmax_label', ),
        context             = devs,
    )
    model.bind(data_shapes=val.provide_data, label_shapes=val.provide_label,
               for_training=False, grad_req='null')
    model.set_params(arg_params, aux_params)
    pred = model.predict(val).asnumpy()
    truth = pd.read_csv(args.truth, index_col=0)
    pred = pd.DataFrame(data={'level': pred}, index=truth.index, dtype=np.int32)
    if args.submit!='':
        pred.to_csv(args.submit)
    else:
        print pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="command for training resnet-v2")
    parser.add_argument('--gpus', type=str, default=None,
                        help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--rec-file', type=str, default='./data/cifar10/cifar10_val.rec',
                        help='the input data for submission')
    parser.add_argument('--bn-mom', type=float, default=0.9,
                        help='momentum for batch normlization')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='the batch size')
    parser.add_argument('--workspace', type=int, default=512,
                        help='memory space size(MB) used in convolution, if xpu '
                        ' memory is oom, then you can try smaller vale, such as --workspace 256')
    parser.add_argument('--depth', type=int, default=164,
                        help='the depth of resnet')
    parser.add_argument('--kv-store', type=str, default='device',
                        help='the kvstore type')
    parser.add_argument('--model-load-epoch', type=int, default=0,
                        help='load the model on an epoch using the model-load-prefix')
    parser.add_argument('--memonger', action='store_true', default=False,
                        help='true means using memonger to save momory, https://github.com/dmlc/mxnet-memonger')
    parser.add_argument('--truth', type=str, default='./data/val1.csv',
                        help='path of ground truth label csv')
    parser.add_argument('--submit', type=str, default='',
                        help='output csv file for submission')
    args = parser.parse_args()
    logging.info(args)
    main()
