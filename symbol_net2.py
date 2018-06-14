'''
Reproducing paper:
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Identity Mappings in Deep Residual Networks"
'''
import mxnet as mx


def residual_unit(data, data_prev, num_filter, stride, dim_match, name, bottle_neck=True, bn_mom=0.9, workspace=512):
    """Return ResNet Unit symbol for building ResNet
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    bnf : int
        Bottle neck channels factor with regard to num_filter
    stride : tupe
        Stride used in convolution
    dim_match : Boolen
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
    """
    if bottle_neck:
        # the same as https://github.com/facebook/fb.resnet.torch#notes, a bit difference with origin paper
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False,
                               eps=2e-5, momentum=bn_mom, name=name + '_bn1')
        act1 = mx.sym.Activation(
            data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter*0.25),
                                   kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                   no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False,
                               eps=2e-5, momentum=bn_mom, name=name + '_bn2')
        act2 = mx.sym.Activation(
            data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter*0.25),
                                   kernel=(3, 3), stride=stride, pad=(1, 1),
                                   no_bias=True, workspace=workspace, name=name + '_conv2')
        bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False,
                               eps=2e-5, momentum=bn_mom, name=name + '_bn3')
        act3 = mx.sym.Activation(
            data=bn3, act_type='relu', name=name + '_relu3')
        conv3 = mx.sym.Convolution(data=act3, num_filter=int(num_filter*0.5),
                                   kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                   no_bias=True, workspace=workspace, name=name + '_conv3')
        if dim_match:
            shortcut = data_prev
        else:
            shortcut = data
        return mx.sym.Concat(conv3, shortcut), conv3
    else:
        raise ValueError("must have bottleneck structure")


def transition_block(num_stage, data, num_filter, stride, name, bn_mom=0.9, workspace=512):
    """Return transition_block unit sym for building DenseNet
    Parameters
    ----------
    num_stage : int
        Number of stage
    data : str
        Input data
    num : int
        Number of output channels
    stride : tuple
        Stride used in convolution
    name : str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
    """
    bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False,
                           eps=2e-5, momentum=bn_mom, name=name + '_bn1')
    act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
    conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter*0.5),
                               kernel=(1, 1), stride=stride, pad=(0, 0),
                               no_bias=True, workspace=workspace, name=name + '_conv1')
    return mx.sym.Pooling(conv1, global_pool=False,
                          kernel=(2, 2), stride=(2, 2),
                          pool_type='avg', name=name + '_pool%d' % (num_stage + 1))


def resnet(units, num_stage, filter_list, num_class, bottle_neck=True, bn_mom=0.9, workspace=512):
    """Return ResNet symbol of cifar10 and imagenet
    Parameters
    ----------
    units : list
        Number of units in each stage
    num_stage : int
        Number of stage
    filter_list : list
        Channel size of each stage
    num_class : int
        Ouput size of symbol
    workspace : int
        Workspace used in convolution operator
    """
    num_unit = len(units)
    assert(num_unit == num_stage)
    lab = mx.sym.Variable(name='softmax_label')
    data = mx.sym.Variable(name='data')
    data = mx.sym.BatchNorm(data=data, fix_gamma=True,
                            eps=2e-5, momentum=bn_mom, name='bn_data')
    body = mx.sym.Convolution(data=data, num_filter=filter_list[0],
                              kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                              no_bias=True, name="conv0", workspace=workspace)
    for i in range(num_stage):
        if i != 0:
            body = transition_block(i, body, filter_list[i + 1], stride=(
                1, 1), name='stage%d_trans' % (i + 1), bn_mom=bn_mom, workspace=workspace)
        body, body_prev = residual_unit(body, None, filter_list[i + 1], (1, 1), False,
                                        name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, workspace=workspace)
        for j in range(units[i] - 1):
            body, body_prev = residual_unit(body, body_prev, filter_list[i + 1], (1, 1), True, name='stage%d_unit%d' % (i + 1, j + 2),
                                            bottle_neck=bottle_neck, workspace=workspace)
    bn1 = mx.sym.BatchNorm(data=body, fix_gamma=False,
                           eps=2e-5, momentum=bn_mom, name='bn1')
    relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')
    # Although kernel is not used here when global_pool=True, we should put one
    pool1 = mx.sym.Pooling(data=relu1, global_pool=True, kernel=(7, 7),
                           pool_type='avg', name='pool1')
    flat = mx.sym.Flatten(data=pool1)
    fc1 = mx.sym.FullyConnected(data=flat, num_hidden=num_class, name='fc1')
    return mx.sym.SoftmaxOutput(data=fc1, label=lab, name='softmax')
    # soft = mx.sym.softmax(data=fc1, axis=1)
    # return mx.sym.Custom(data=soft, label=lab, name='loss', op_type='loss')
