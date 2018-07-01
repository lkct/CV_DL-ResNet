import mxnet as mx


def bn_ac_conv(data, num_filter, name, kernel, stride=(1, 1), pad=(0, 0), bn_mom=0.9, workspace=512):
    bn = mx.sym.BatchNorm(data=data, fix_gamma=False,
                          eps=2e-5, momentum=bn_mom, name=name+'_bn')
    ac = mx.sym.Activation(
        data=bn, act_type='relu', name=name+'_relu')
    conv = mx.sym.Convolution(data=ac, num_filter=num_filter,
                              kernel=kernel, stride=stride, pad=pad,
                              no_bias=True, workspace=workspace, name=name+'_conv')
    return conv


def dual_path_block(data, num_filter, name, inc=12, type_='norm', workspace=512):
    if type_ is 'proj':
        stride = (1, 1)
        has_proj = True
    elif type_ is 'down':
        stride = (2, 2)
        has_proj = True
    elif type_ is 'norm':
        stride = (1, 1)
        has_proj = False

    if has_proj:
        if type(data) is list:
            data = mx.sym.Concat(data[0], data[1], name=name+'_input')
        shortcut = mx.sym.Convolution(data=data, num_filter=(num_filter+2*inc),
                                      kernel=(1, 1), stride=stride, pad=(0, 0),
                                      no_bias=True, workspace=workspace, name=name+'_sc')
        sc_plus = mx.sym.slice_axis(
            data=shortcut, axis=1, begin=0, end=num_filter, name=name+'_sc_plus')
        sc_conc = mx.sym.slice_axis(data=shortcut, axis=1, begin=num_filter, end=(
            num_filter+2*inc), name=name+'_sc_concat')
    else:
        sc_plus = data[0]
        sc_conc = data[1]
        data = mx.sym.Concat(data[0], data[1], name=name+'_input')

    conv1 = bn_ac_conv(data=data, num_filter=int(num_filter*0.25),
                       kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                       workspace=workspace, name=name+'_conv1')
    conv2 = bn_ac_conv(data=conv1, num_filter=int(num_filter*0.25),
                       kernel=(3, 3), stride=stride, pad=(1, 1),
                       workspace=workspace, name=name+'_conv2')
    conv3 = bn_ac_conv(data=conv2, num_filter=(num_filter+inc),
                       kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                       workspace=workspace, name=name+'_conv3')

    conv_plus = mx.sym.slice_axis(data=conv3, axis=1, begin=0, end=num_filter,
                                  name=name+'_conv_plus')
    conv_conc = mx.sym.slice_axis(data=conv3, axis=1, begin=num_filter, end=(num_filter+inc),
                                  name=name+'_conv_concat')

    plus = sc_plus + conv_plus
    conc = mx.sym.Concat(sc_conc, conv_conc, name=name+'_concat')

    return [plus, conc]


def dpnet(units, num_stage, filter_list, num_class, bn_mom=0.9, workspace=512):
    num_unit = len(units)
    assert(num_unit == num_stage)
    data = mx.sym.Variable(name='data')
    data = mx.sym.BatchNorm(data=data, fix_gamma=True,
                            eps=2e-5, momentum=bn_mom, name='bn_data')
    body = mx.sym.Convolution(data=data, num_filter=filter_list[0],
                              kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                              no_bias=True, name="conv0", workspace=workspace)
    for i in range(num_stage):
        body = dual_path_block(body, filter_list[i+1], name='stage%d_unit%d' % (i + 1, 1), workspace=workspace,
                               type_ ='proj' if i == 0 else 'down')
        for j in range(units[i]-1):
            body = dual_path_block(body, filter_list[i+1], name='stage%d_unit%d' % (i + 1, j + 2), workspace=workspace,
                                   type_='norm')
    body = mx.sym.Concat(body[0], body[1], name='final_concat')
    bn1 = mx.sym.BatchNorm(data=body, fix_gamma=False,
                           eps=2e-5, momentum=bn_mom, name='bn1')
    relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')
    # Although kernel is not used here when global_pool=True, we should put one
    pool1 = mx.sym.Pooling(data=relu1, global_pool=True, kernel=(7, 7),
                           pool_type='avg', name='pool1')
    flat = mx.sym.Flatten(data=pool1)
    fc1 = mx.sym.FullyConnected(data=flat, num_hidden=num_class, name='fc1')
    return mx.sym.SoftmaxOutput(data=fc1, name='softmax')
