'''
Adapted from https://github.com/tornadomeet/ResNet/blob/master/symbol_resnet.py
Original author Wei Wu

Implemented the following paper:
Saining Xie, Ross Girshick, Piotr Dollar, Zhuowen Tu, Kaiming He. "Aggregated Residual Transformations for Deep Neural Network"
'''
import mxnet as mx
import numpy as np


def residual_unit(data, num_filter, stride, dim_match, name, bottle_neck=True, num_group=32, bn_mom=0.9, workspace=256, memonger=False):
    """Return ResNet Unit symbol for building ResNet
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    bnf : int
        Bottle neck channels factor with regard to num_filter
    stride : tuple
        Stride used in convolution
    dim_match : Boolean
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
        conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter*0.5),
                                   kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                   no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False,
                               eps=2e-5, momentum=bn_mom, name=name + '_bn2')
        act2 = mx.sym.Activation(
            data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter*0.5), num_group=num_group,
                                   kernel=(3, 3), stride=stride, pad=(1, 1),
                                   no_bias=True, workspace=workspace, name=name + '_conv2')
        bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False,
                               eps=2e-5, momentum=bn_mom, name=name + '_bn3')
        act3 = mx.sym.Activation(
            data=bn3, act_type='relu', name=name + '_relu3')
        conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter,
                                   kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                   no_bias=True, workspace=workspace, name=name + '_conv3')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=data, num_filter=num_filter,
                                          kernel=(1, 1), stride=stride, no_bias=True,
                                          workspace=workspace, name=name+'_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return conv3 + shortcut
    else:
        raise ValueError("must have bottleneck structure to differ from resnet")
        # conv1 = mx.sym.Convolution(data=data, num_filter=num_filter,
        #                            kernel=(3, 3), stride=stride, pad=(1, 1),
        #                            no_bias=True, workspace=workspace, name=name + '_conv1')
        # bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False,
        #                        momentum=bn_mom, eps=2e-5, name=name + '_bn1')
        # act1 = mx.sym.Activation(
        #     data=bn1, act_type='relu', name=name + '_relu1')
        # conv2 = mx.sym.Convolution(data=act1, num_filter=num_filter,
        #                            kernel=(3, 3), stride=(1, 1), pad=(1, 1),
        #                            no_bias=True, workspace=workspace, name=name + '_conv2')
        # bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False,
        #                        momentum=bn_mom, eps=2e-5, name=name + '_bn2')
        # if dim_match:
        #     shortcut = data
        # else:
        #     shortcut_conv = mx.sym.Convolution(data=data, num_filter=num_filter,
        #                                        kernel=(1, 1), stride=stride, no_bias=True,
        #                                        workspace=workspace, name=name+'_sc')
        #     shortcut = mx.sym.BatchNorm(data=shortcut_conv, fix_gamma=False,
        #                                 eps=2e-5, momentum=bn_mom, name=name + '_sc_bn')
        # if memonger:
        #     shortcut._set_attr(mirror_stage='True')
        # eltwise = bn2 + shortcut
        # return mx.sym.Activation(data=eltwise, act_type='relu', name=name + '_relu')


def resnext(units, num_stage, filter_list, num_class, num_group, bottle_neck=True, bn_mom=0.9, workspace=256, memonger=False):
    """Return ResNeXt symbol of
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
    num_groupes : int
        Number of conv groups
    workspace : int
        Workspace used in convolution operator
    """
    num_unit = len(units)
    assert(num_unit == num_stage)
    data = mx.sym.Variable(name='data')
    data = mx.sym.BatchNorm(data=data, fix_gamma=True,
                            eps=2e-5, momentum=bn_mom, name='bn_data')
    body = mx.sym.Convolution(data=data, num_filter=filter_list[0],
                              kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                              no_bias=True, name="conv0", workspace=workspace)
    for i in range(num_stage):
        body = residual_unit(body, filter_list[i+1], (1 if i == 0 else 2, 1 if i == 0 else 2), False,
                             name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, num_group=num_group,
                             bn_mom=bn_mom, workspace=workspace, memonger=memonger)
        for j in range(units[i]-1):
            body = residual_unit(body, filter_list[i+1], (1, 1), True, name='stage%d_unit%d' % (i + 1, j + 2),
                                 bottle_neck=bottle_neck, num_group=num_group, bn_mom=bn_mom, workspace=workspace, memonger=memonger)
    bn1 = mx.sym.BatchNorm(data=body, fix_gamma=False,
                           eps=2e-5, momentum=bn_mom, name='bn1')
    relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')
    pool1 = mx.sym.Pooling(data=relu1, global_pool=True, kernel=(7, 7),
                           pool_type='avg', name='pool1')
    flat = mx.sym.Flatten(data=pool1)
    fc1 = mx.sym.FullyConnected(data=flat, num_hidden=num_class, name='fc1')
    return mx.sym.SoftmaxOutput(data=fc1, name='softmax')
