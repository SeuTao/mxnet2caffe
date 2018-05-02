import json
from prototxt_basic import *
import sys, argparse
sys.path.append( '..../python')
import mxnet as mx
import caffe
import numpy as np

parser = argparse.ArgumentParser(description='Convert MXNet jason to Caffe prototxt')
parser.add_argument('--mx-json',     type=str, default='model.json')
parser.add_argument('--mx-model',    type=str, default='')
parser.add_argument('--mx-epoch',    type=int, default=0)
parser.add_argument('--cf-prototxt', type=str, default='model.prototxt')
parser.add_argument('--cf-model',    type=str, default='model.caffemodel')
args = parser.parse_args()

#-------------------------------------------
CAFFE_LAYERS = [ 'Convolution',
                 'ChannelwiseConvolution',
                 'BatchNorm',
                 'Scale',
                 'Activation',
                 'elemwise_add',
                 '_Plus',
                 'Concat',
                 'Pooling',
                 'Flatten',
                 'FullyConnected',
                 'SoftmaxOutput']
# ------------------------------------------

# Load
_, arg_params, aux_params = mx.model.load_checkpoint(args.mx_model, args.mx_epoch)
all_params = dict(arg_params.items()+aux_params.items())


all_keys = all_params.keys()
all_keys.sort()

def get_isinput_blob(info_list, name_list):
  index_list = []
  if len(name_list)==0:
    return index_list

  name = name_list[0]
  for i in range(len(info_list)):
    info = info_list[i]
    if name in info['bottom']:
      index_list.append(i)

  return index_list

def get_name_blob(info_list, name_list):
  index_list = []
  if len(name_list)==0:
    return index_list

  name = name_list[0]

  if name == '_mulscalar0':
    a = 0
  for i in range(len(info_list)):
    info = info_list[i]
    if name == info['name']:
      index_list.append(i)

  return index_list

def info_list_merge_bn(info_list):
  i = 0
  while i < len(info_list):
    info_first = info_list[i]
    info_second = info_list[i + 1]

    if info_first['name'] == 'pre_fc1':
      a = 0

    if (info_first['op'] == 'Convolution' or info_first['op'] == 'FullyConnected') and info_second['op'] == 'BatchNorm':
      if info_second['bottom'][0] == info_first['top']:
        info_first['attrs']['no_bias'] = 'False'

        weight = info_first['params'][info_first['name']+'_weight']
        if info_first['params'].has_key(info_first['name'] + '_bias'):
          bias = info_first['params'][info_first['name'] + '_bias']
        else:
          bias = np.zeros([weight.shape[0],1])

        bn_gamma = info_second['params'][info_second['name']+'_gamma']
        bn_beta = info_second['params'][info_second['name'] + '_beta']
        bn_moving_mean = info_second['params'][info_second['name'] + '_moving_mean']
        bn_moving_var = info_second['params'][info_second['name'] + '_moving_var']

        bn_variance_tmp = np.sqrt(bn_moving_var.reshape([-1]) + 0.00002)
        bn_mean_tmp = bn_moving_mean.reshape([-1])
        bn_gamma_tmp = bn_gamma.reshape([-1])
        bn_beta_tmp = bn_beta.reshape([-1])
        bias = bias.reshape([-1])

        fmap_num = weight.shape[0]
        # print(fmap_num)

        if info_first['op'] == 'Convolution':
          for f_index in range(fmap_num):
            weight[f_index, :, :, :] = weight[f_index, :, :, :] * bn_gamma_tmp[f_index] / bn_variance_tmp[f_index]
            bias[f_index] = (bias[f_index] - bn_mean_tmp[f_index]) * bn_gamma_tmp[f_index] / bn_variance_tmp[f_index] + bn_beta_tmp[f_index]
        else:
          for f_index in range(fmap_num):
            weight[f_index, :] = weight[f_index, :] * bn_gamma_tmp[f_index] / bn_variance_tmp[f_index]
            bias[f_index] = (bias[f_index] - bn_mean_tmp[f_index]) * bn_gamma_tmp[f_index] / bn_variance_tmp[f_index] + bn_beta_tmp[f_index]

        info_first['params'][info_first['name'] + '_weight'] = weight
        info_first['params'][info_first['name'] + '_bias'] = bias

        index_list = get_isinput_blob(info_list, [info_second['name']])
        for index in index_list:
          info_list[index]['bottom'][0] = info_first['top']
        info_list.remove(info_second)

    elif info_first['op'] == 'BatchNorm':
      info_first['op'] = 'Scale'

      bn_gamma = info_first['params'][info_first['name'] + '_gamma']
      bn_beta = info_first['params'][info_first['name'] + '_beta']
      bn_moving_mean = info_first['params'][info_first['name'] + '_moving_mean']
      bn_moving_var = info_first['params'][info_first['name'] + '_moving_var']

      bn_variance_tmp = np.sqrt(bn_moving_var.reshape([-1]) + 0.00002)
      bn_mean_tmp = bn_moving_mean.reshape([-1])
      bn_gamma_tmp = bn_gamma.reshape([-1])
      bn_beta_tmp = bn_beta.reshape([-1])

      gamma = bn_gamma_tmp / bn_variance_tmp
      beta =  ( 0 - bn_mean_tmp) * bn_gamma_tmp / bn_variance_tmp + bn_beta_tmp

      info_first['params'][info_first['name'] + '_gamma'] = gamma
      info_first['params'][info_first['name'] + '_beta'] = beta

    i += 1
    if i >= len(info_list)-1:
      break

  return info_list

def info_list_merge_act(info_list):
  i = 0
  while i < len(info_list):
    info_first = info_list[i]
    if info_first['op'] == 'Activation':
      info_first['name'] = info_first['top']
      info_first['top'] = info_first['bottom'][0]

      index_list = get_isinput_blob(info_list, [info_first['name']])
      for index in index_list:

        for j in range(len(info_list[index]['bottom'])):
          if info_list[index]['bottom'][j] == info_first['name']:
            info_list[index]['bottom'][j] = info_first['bottom'][0]

    i += 1
    if i >= len(info_list)-1:
      break

  return info_list

def info_list_exclude(info_list, name):
  new_list = []
  for info in info_list:
    if name not in info['name']:
      new_list.append(info)

  return new_list


def info_list_merge_unknown(info_list):
  i = 0
  while i < len(info_list):
    info_first = info_list[i]
    if info_first['op'] not in CAFFE_LAYERS and info_first['name'] != 'data':
      print('unknown')
      index_list = get_isinput_blob(info_list, [info_first['name']])

      for index in index_list:
        info_list[index]['bottom'] = info_first['bottom']

      info_list.remove(info_first)

    else:
      i += 1

    if i >= len(info_list)-1:
      break

  return info_list



def create_prototxt(merge_bn = False):

  with open(args.mx_json) as json_file:
    jdata = json.load(json_file)

  with open(args.cf_prototxt, "w") as prototxt_file:

    info_list = []

    for i_node in range(0,len(jdata['nodes'])):
      node_i    = jdata['nodes'][i_node]


      print('{}, \top:{}, name:{} -> {}'.format(i_node,node_i['op'].ljust(20),
                                          node_i['name'].ljust(30),
                                          node_i['name']).ljust(20))

      info = node_i
      info['top'] = info['name']
      info['bottom'] = []
      info['params'] = {}
      for input_idx_i in node_i['inputs']:
        input_i = jdata['nodes'][input_idx_i[0]]

        if str(input_i['op']) != 'null' or (str(input_i['name']) == 'data'):
          if str(input_i['name']) == 'id':
            info['bottom'].append('data')
          else:
            info['bottom'].append(str(input_i['name']))

        if str(input_i['op']) == 'null':

          if all_params.has_key(str(input_i['name'])):
            info['params'][str(input_i['name'])] = all_params[str(input_i['name'])].asnumpy()

          if not str(input_i['name']).startswith(str(node_i['name'])):
            print('use shared weight -> %s'% str(input_i['name']))
            info['share'] = True

      info_list.append(info)

    info_list = info_list_merge_unknown(info_list)
    info_list = info_list_merge_bn(info_list)
    info_list = info_list_merge_act(info_list)

    for info in info_list:
      write_node(prototxt_file, info)

    return info_list

def create_caffemodel(info_list):
  net = caffe.Net(args.cf_prototxt, caffe.TEST)
  for info in info_list:
    name = info['name']
    op = info['op']
    params = info['params']

    if op == 'Scale':
      net.params[name][0].data.flat = params[name + '_gamma'].flat
      net.params[name][1].data.flat = params[name + '_beta'].flat
    if op == 'Convolution':
      net.params[name][0].data.flat = params[name + '_weight'].flat
      if params.has_key(name + '_bias'):
        net.params[name][1].data.flat = params[name + '_bias'].flat
    if op == 'FullyConnected':
      net.params[name][0].data.flat = params[name + '_weight'].flat
      net.params[name][1].data.flat = params[name + '_bias'].flat
  # ------------------------------------------
  # Finish
  net.save(args.cf_model)
  print("\n- Finished.\n")

info_list = create_prototxt(True)
create_caffemodel(info_list)
