"""
Adapted from https://github.com/tornadomeet/ResNet/blob/master/sym_resnet.py 
Original author Wei Wu
Referenced https://github.com/bamos/densenet.pytorch/blob/master/densenet.py
Original author bamos
Referenced https://github.com/andreasveit/densenet-pytorch/blob/master/densenet.py
Original author andreasveit
Referenced https://github.com/Nicatio/Densenet/blob/master/mxnet/sym_densenet.py
Original author Nicatio

Implemented the following paper:     DenseNet-BC
Gao Huang, Zhuang Liu, Kilian Q. Weinberger, Laurens van der Maaten. "Densely Connected Convolutional Networks"

Coded by Lin Xiong Mar-1, 2017
"""
import mxnet as mx
import math


def basic_block(data, growth_rate, stride, name, bottle_neck=True, drop_out=0.0, bn_mom=0.9, workspace=512):
    """Return BaiscBlock unit sym for building dense_block
    Parameters
    ----------
    data : str
        Input data
    growth_rate : int
        Number of output channels
    stride : tupe
        Stride used in convolution
    drop_out : float
        Probability of an element to be zeroed. Default = 0.0
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
        conv1 = mx.sym.Convolution(data=act1, num_filter=int(growth_rate * 4),
                                   kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                   no_bias=True, workspace=workspace, name=name + '_conv1')
        if drop_out > 0:
            conv1 = mx.sym.Dropout(data=conv1, p=drop_out, name=name + '_dp1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False,
                               eps=2e-5, momentum=bn_mom, name=name + '_bn2')
        act2 = mx.sym.Activation(
            data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=int(growth_rate),
                                   kernel=(3, 3), stride=stride, pad=(1, 1),
                                   no_bias=True, workspace=workspace, name=name + '_conv2')
        if drop_out > 0:
            conv2 = mx.sym.Dropout(data=conv2, p=drop_out, name=name + '_dp2')
        # return mx.sym.Concat(data, conv2, name=name + '_concat0')
        return conv2
    else:
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False,
                               eps=2e-5, momentum=bn_mom, name=name + '_bn1')
        act1 = mx.sym.Activation(
            data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=int(growth_rate),
                                   kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                                   no_bias=True, workspace=workspace, name=name + '_conv1')
        if drop_out > 0:
            conv1 = mx.sym.Dropout(data=conv1, p=drop_out, name=name + '_dp1')
        # return mx.sym.Concat(data, conv1, name=name + '_concat0')
        return conv1


def dense_block(units_num, data, growth_rate, name, bottle_neck=True, drop_out=0.0, bn_mom=0.9, workspace=512):
    """Return dense_block unit sym for building DenseNet
    Parameters
    ----------
    units_num : int
        the number of basic_block in each dense_block
    data : str	
        Input data
    growth_rate : int
        Number of output channels
    drop_out : float
        Probability of an element to be zeroed. Default = 0.0
    workspace : int
        Workspace used in convolution operator
    """
    for i in range(units_num):
        Block = basic_block(data, growth_rate=growth_rate, stride=(1, 1), name=name + '_unit%d' % (i + 1),
                           bottle_neck=bottle_neck, drop_out=drop_out,
                           bn_mom=bn_mom, workspace=workspace)
        data = mx.sym.Concat(data, Block, name=name + '_concat%d' % (i + 1))
    return data


def transition_block(num_stage, data, num_filter, stride, name, drop_out=0.0, bn_mom=0.9, workspace=512):
    """Return transition_block unit sym for building DenseNet
    Parameters
    ----------
    num_stage : int
        Number of stage
    data : str
        Input data
    num : int
        Number of output channels
    stride : tupe
        Stride used in convolution
    name : str
        Base name of the operators
    drop_out : float
        Probability of an element to be zeroed. Default = 0.0
    workspace : int
        Workspace used in convolution operator
    """
    bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False,
                           eps=2e-5, momentum=bn_mom, name=name + '_bn1')
    act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
    conv1 = mx.sym.Convolution(data=act1, num_filter=num_filter,
                               kernel=(1, 1), stride=stride, pad=(0, 0),
                               no_bias=True, workspace=workspace, name=name + '_conv1')
    if drop_out > 0:
        conv1 = mx.sym.Dropout(data=conv1, p=drop_out, name=name + '_dp1')
    return mx.sym.Pooling(conv1, global_pool=False,
                          kernel=(2, 2), stride=(2, 2),
                          pool_type='avg', name=name + '_pool%d' % (num_stage + 1))


def densenet(units, num_stage, growth_rate, num_class, reduction=0.5, drop_out=0., bottle_neck=True, bn_mom=0.9, workspace=512):
    """Return DenseNet sym of imagenet
    Parameters
    ----------
    units : list
        Number of units in each stage
    num_stage : int
        Number of stage
    growth_rate : int
        Number of output channels
    num_class : int
        Ouput size of sym
    reduction : float
        Compression ratio. Default = 0.5
    drop_out : float
        Probability of an element to be zeroed. Default = 0.0
    workspace : int
        Workspace used in convolution operator
    """
    num_unit = len(units)
    assert(num_unit == num_stage)
    init_channels = 2 * growth_rate
    n_channels = init_channels
    data = mx.sym.Variable(name='data')
    data = mx.sym.BatchNorm(data=data, fix_gamma=True,
                            eps=2e-5, momentum=bn_mom, name='bn_data')
    body = mx.sym.Convolution(data=data, num_filter=growth_rate * 2,
                              kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                              no_bias=True, name="conv0", workspace=workspace)
    for i in range(num_stage - 1):
        body = dense_block(units[i], body, growth_rate=growth_rate, name='DBstage%d' % (
            i + 1), bottle_neck=bottle_neck, drop_out=drop_out, bn_mom=bn_mom, workspace=workspace)
        n_channels += units[i] * growth_rate
        n_channels = int(math.floor(n_channels * reduction))
        body = transition_block(i, body, n_channels, stride=(1, 1), name='TBstage%d' % (
            i + 1), drop_out=drop_out, bn_mom=bn_mom, workspace=workspace)
    body = dense_block(units[num_stage - 1], body, growth_rate=growth_rate, name='DBstage%d' % (
        num_stage), bottle_neck=bottle_neck, drop_out=drop_out, bn_mom=bn_mom, workspace=workspace)

    bn1 = mx.sym.BatchNorm(data=body, fix_gamma=False,
                           eps=2e-5, momentum=bn_mom, name='bn1')
    relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')
    pool1 = mx.sym.Pooling(data=relu1, global_pool=True,
                           kernel=(7, 7), pool_type='avg', name='pool1')
    flat = mx.sym.Flatten(data=pool1)
    fc1 = mx.sym.FullyConnected(data=flat, num_hidden=num_class, name='fc1')
    return mx.sym.SoftmaxOutput(data=fc1, name='softmax')
