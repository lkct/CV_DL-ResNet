import argparse
import logging
import os
import time

import mxnet as mx

from symbol_net3 import net3

fname = time.strftime("%Y%m%d%H%M%S", time.localtime())
logging.basicConfig(level=logging.INFO,
                    filename='log/'+fname+'.log', filemode='w')
logger = logging.getLogger()

data_type = 'cifar10'


def multi_factor_scheduler(begin_epoch, epoch_size, step=[60, 75, 90], factor=0.1):
    step_ = [epoch_size * (x-begin_epoch) for x in step if x-begin_epoch > 0]
    return mx.lr_scheduler.MultiFactorScheduler(step=step_, factor=factor) if len(step_) else None


def main():
    if (args.depth-2) % 9 == 0:  # and args.depth >= 164:
        per_unit = [(args.depth-2) / 9]
        filter_list = [16, 64, 128, 256]
        bottle_neck = True
    # elif (args.depth-2) % 6 == 0 and args.depth < 164:
    #     per_unit = [(args.depth-2) / 6]
    #     filter_list = [16, 16, 32, 64]
    #     bottle_neck = False
    else:
        raise ValueError(
            "no experiments done on detph {}, you can do it youself".format(args.depth))
    units = per_unit*3
    symbol = net3(units=units, num_stage=3, filter_list=filter_list, num_class=args.num_classes,
                    bottle_neck=bottle_neck, bn_mom=args.bn_mom, workspace=args.workspace)
    kv = mx.kvstore.create(args.kv_store)
    devs = mx.cpu() if args.gpus is None else [
        mx.gpu(int(i)) for i in args.gpus.split(',')]
    epoch_size = max(
        int(args.num_examples / args.batch_size / kv.num_workers), 1)
    begin_epoch = args.model_load_epoch if args.model_load_epoch else 0
    if not os.path.exists("./model"):
        os.mkdir("./model")
    model_prefix = "model/net3-{}-{}-{}".format(
        data_type, args.depth, kv.rank)
    checkpoint = mx.callback.do_checkpoint(model_prefix)
    arg_params = None
    aux_params = None
    if args.retrain:
        _, arg_params, aux_params = mx.model.load_checkpoint(
            model_prefix, args.model_load_epoch)
    train = mx.io.ImageRecordIter(
        path_imgrec         = os.path.join(args.data_dir, "cifar10_train.rec"),
        label_width         = 1,
        data_shape          = (3, 32, 32),
        num_parts           = kv.num_workers,
        part_index          = kv.rank,
        shuffle             = True,
        batch_size          = args.batch_size,
        rand_crop           = True,
        fill_value          = 127,  # only used when pad is valid
        pad                 = 4,
        rand_mirror         = True,
    )
    val = mx.io.ImageRecordIter(
        path_imgrec         = os.path.join(args.data_dir, "cifar10_val.rec"),
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
    model.fit(
        train_data          = train,
        eval_data           = val,
        eval_metric         = ['acc'],
        epoch_end_callback  = checkpoint,
        batch_end_callback  = mx.callback.Speedometer(args.batch_size, args.frequent),
        kvstore             = kv,
        optimizer           = 'nag',
        optimizer_params    = (('learning_rate', args.lr), ('momentum', args.mom), ('wd', args.wd), (
            'lr_scheduler', multi_factor_scheduler(begin_epoch, epoch_size, step=[80], factor=0.1))),
        initializer         = mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2),
        arg_params          = arg_params,
        aux_params          = aux_params,
        allow_missing       = True,
        begin_epoch         = begin_epoch,
        num_epoch           = args.end_epoch,
    )
    # logging.info("top-1 and top-5 acc is {}".format(model.score(X = val,
    #               eval_metric = ['acc', mx.metric.create('top_k_accuracy', top_k = 5)])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="command for training net3")
    parser.add_argument('--gpus', type=str, default=None,
                        help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--data-dir', type=str, default='./data/cifar10/',
                        help='the input data directory')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='initialization learning reate')
    parser.add_argument('--mom', type=float, default=0.9,
                        help='momentum for sgd')
    parser.add_argument('--bn-mom', type=float, default=0.9,
                        help='momentum for batch normlization')
    parser.add_argument('--wd', type=float, default=0.0001,
                        help='weight decay for sgd')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='the batch size')
    parser.add_argument('--workspace', type=int, default=512,
                        help='memory space size(MB) used in convolution, if xpu '
                        ' memory is oom, then you can try smaller vale, such as --workspace 256')
    parser.add_argument('--depth', type=int, default=164,
                        help='the depth of net3')
    parser.add_argument('--num-classes', type=int, default=10,
                        help='the class number of your task')
    parser.add_argument('--num-examples', type=int, default=50000,
                        help='the number of training examples')
    parser.add_argument('--kv-store', type=str, default='device',
                        help='the kvstore type')
    parser.add_argument('--model-load-epoch', type=int, default=0,
                        help='load the model on an epoch using the model-load-prefix')
    parser.add_argument('--end-epoch', type=int, default=120,
                        help='training ends at this num of epoch')
    parser.add_argument('--frequent', type=int, default=50,
                        help='frequency of logging')
    parser.add_argument('--retrain', action='store_true', default=False,
                        help='true means continue training')
    args = parser.parse_args()
    logging.info(args)
    main()
