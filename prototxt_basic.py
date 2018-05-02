import sys
# prototxt_basic

def data(txt_file, info):
  txt_file.write('name: "mxnet-mdoel"\n')
  txt_file.write('layer {\n')
  txt_file.write('  name: "data"\n')
  txt_file.write('  type: "Input"\n')
  txt_file.write('  top: "data"\n')
  txt_file.write('  input_param {\n')
  txt_file.write('    shape: { dim: 1 dim: 1 dim: 128 dim: 128 }\n') # TODO
  txt_file.write('  }\n')
  txt_file.write('}\n')
  txt_file.write('\n')

def Convolution(txt_file, info):
  if info['attrs']['no_bias'] == 'True':
    bias_term = 'false'
  else:
    bias_term = 'true'  
  txt_file.write('layer {\n')
  txt_file.write('	bottom: "%s"\n'       % info['bottom'][0])
  txt_file.write('	top: "%s"\n'          % info['top'])
  txt_file.write('	name: "%s"\n'         % info['top'])
  txt_file.write('	type: "Convolution"\n')
  txt_file.write('	convolution_param {\n')
  txt_file.write('		num_output: %s\n'   % info['attrs']['num_filter'])
  txt_file.write('		kernel_size: %s\n'  % info['attrs']['kernel'].split('(')[1].split(',')[0]) # TODO

  if info['attrs'].has_key('pad'):
    txt_file.write('		pad: %s\n'           % info['attrs']['pad'].split('(')[1].split(',')[0]) # TODO

  # txt_file.write('		group: %s\n'        % info['attrs']['num_group'])
  txt_file.write('		stride: %s\n'       % info['attrs']['stride'].split('(')[1].split(',')[0])
  txt_file.write('		bias_term: %s\n'    % bias_term)
  txt_file.write('	}\n')
  # if 'share' in info.keys() and info['share']:
  #   txt_file.write('	param {\n')
  #   txt_file.write('	name: "%s"\n'     % info['params'][0])
  #   txt_file.write('	}\n')
  txt_file.write('}\n')
  txt_file.write('\n')

def ChannelwiseConvolution(txt_file, info):
  Convolution(txt_file, info)
  
def BatchNorm(txt_file, info):
  txt_file.write('layer {\n')
  txt_file.write('  bottom: "%s"\n'       % info['bottom'][0])
  txt_file.write('  top: "%s"\n'          % info['top'])
  txt_file.write('  name: "%s"\n'         % info['top'])
  txt_file.write('  type: "BatchNorm"\n')
  txt_file.write('  batch_norm_param {\n')
  txt_file.write('    use_global_stats: true\n')        # TODO
  txt_file.write('    moving_average_fraction: 0.9\n')  # TODO
  txt_file.write('    eps: 0.00002\n')                    # TODO
  txt_file.write('  }\n')
  txt_file.write('}\n')

  # if info['fix_gamma'] is "True":
  #   a = 0# TODO

  txt_file.write('layer {\n')
  txt_file.write('  bottom: "%s"\n'  % info['top'])
  txt_file.write('  top: "%s"\n'     % info['top'])
  txt_file.write('  name: "%s_scale"\n'    % info['top'])
  txt_file.write('  type: "Scale"\n')
  txt_file.write('  scale_param { bias_term: true }\n')
  txt_file.write('}\n')
  txt_file.write('\n')
  pass

def Scale(txt_file, info):
  txt_file.write('layer {\n')
  txt_file.write('  bottom: "%s"\n' % info['bottom'][0])
  txt_file.write('  top: "%s"\n'    % info['top'])
  txt_file.write('  name: "%s"\n'   % info['top'])
  txt_file.write('  type: "Scale"\n')
  txt_file.write('  scale_param { bias_term: true }\n')
  txt_file.write('}\n')
  txt_file.write('\n')
  pass

def Activation(txt_file, info):
  txt_file.write('layer {\n')
  txt_file.write('  bottom: "%s"\n'       % info['bottom'][0])
  txt_file.write('  top: "%s"\n'          % info['top'])
  txt_file.write('  name: "%s"\n'         % info['name'])
  txt_file.write('  type: "ReLU"\n')      # TODO
  txt_file.write('}\n')
  txt_file.write('\n')
  pass

def Concat(txt_file, info):
  txt_file.write('layer {\n')
  txt_file.write('  name: "%s"\n'         % info['top'])
  txt_file.write('  type: "Concat"\n')
  for bottom_i in info['bottom']:
    txt_file.write('  bottom: "%s"\n'     % bottom_i)
  txt_file.write('  top: "%s"\n'          % info['top'])
  txt_file.write('}\n')
  txt_file.write('\n')
  pass
  
def ElementWiseSum(txt_file, info):
  txt_file.write('layer {\n')
  txt_file.write('  name: "%s"\n'         % info['top'])
  txt_file.write('  type: "Eltwise"\n')
  for bottom_i in info['bottom']:
    txt_file.write('  bottom: "%s"\n'     % bottom_i)
  txt_file.write('  top: "%s"\n'          % info['top'])
  txt_file.write('}\n')
  txt_file.write('\n')
  pass

def Pooling(txt_file, info):
  pool_type = 'AVE' if info['attrs']['pool_type'] == 'avg' else 'MAX'
  txt_file.write('layer {\n')
  txt_file.write('  bottom: "%s"\n'       % info['bottom'][0])
  txt_file.write('  top: "%s"\n'          % info['top'])
  txt_file.write('  name: "%s"\n'         % info['top'])
  txt_file.write('  type: "Pooling"\n')
  txt_file.write('  pooling_param {\n')
  txt_file.write('    pool: %s\n'         % pool_type)       # TODO
  txt_file.write('    kernel_size: %s\n'  % info['attrs']['kernel'].split('(')[1].split(',')[0])
  txt_file.write('    stride: %s\n'       % info['attrs']['stride'].split('(')[1].split(',')[0])
  txt_file.write('    pad: %s\n'          % info['attrs']['pad'].split('(')[1].split(',')[0])
  txt_file.write('  }\n')
  txt_file.write('}\n')
  txt_file.write('\n')
  pass


def FullyConnected(txt_file, info):
  txt_file.write('layer {\n')
  txt_file.write('  bottom: "%s"\n'     % info['bottom'][0])
  txt_file.write('  top: "%s"\n'        % info['top'])
  txt_file.write('  name: "%s"\n'       % info['top'])
  txt_file.write('  type: "InnerProduct"\n')
  txt_file.write('  inner_product_param {\n')
  txt_file.write('    num_output: %s\n' % info['attrs']['num_hidden'])
  txt_file.write('  }\n')
  txt_file.write('}\n')
  txt_file.write('\n')
  pass

def Flatten(txt_file, info):
  pass
  
def SoftmaxOutput(txt_file, info):
  pass


# ----------------------------------------------------------------
def write_node(txt_file, info):
    if 'label' in info['name']:
        return        
    if info['op'] == 'null' and info['name'] == 'data':
        data(txt_file, info)
    elif info['op'] == 'Convolution':
        Convolution(txt_file, info)
    elif info['op'] == 'ChannelwiseConvolution':
        ChannelwiseConvolution(txt_file, info)
    elif info['op'] == 'BatchNorm':
        BatchNorm(txt_file, info)
    elif info['op'] == 'Scale':
        Scale(txt_file, info)
    elif info['op'] == 'Activation':
        Activation(txt_file, info)
    elif info['op'] == 'elemwise_add':
        ElementWiseSum(txt_file, info)
    elif info['op'] == '_Plus':
        ElementWiseSum(txt_file, info)
    elif info['op'] == 'Concat':
        Concat(txt_file, info)
    elif info['op'] == 'Pooling':
        Pooling(txt_file, info)
    elif info['op'] == 'Flatten':
        Flatten(txt_file, info)
    elif info['op'] == 'FullyConnected':
        FullyConnected(txt_file, info)
    elif info['op'] == 'SoftmaxOutput':
        SoftmaxOutput(txt_file, info)
    else:
        print("Warning!  Unknown mxnet op:{}".format(info['op']))
        # sys.exit("Warning!  Unknown mxnet op:{}".format(info['op']))




