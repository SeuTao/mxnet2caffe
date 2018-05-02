import sys
import argparse
import json
from prototxt_basic import *

parser = argparse.ArgumentParser(description='Convert MXNet jason to Caffe prototxt')
parser.add_argument('--mx-json',     type=str, default='/media/st/DATA01/Projects/insightface/model/model/Ms_128_stride2_finalstride2_caffesame/'
                                                         'model-symbol.json')
parser.add_argument('--cf-prototxt', type=str, default='model_caffe/test.prototxt')
args = parser.parse_args()


def get_isinput_blob(info_list, name):
  index_list = []
  for i in range(len(info_list)):
    info = info_list[i]
    if name in info['bottom']:
      index_list.append(i)

  return index_list


def info_list_merge_bn(info_list):
  # batch_norm + scale -> scale
  # batch_norm -> scale

  # new_info_list = []
  i = 0
  while i < len(info_list):
    info_first = info_list[i]
    info_second = info_list[i + 1]

    if (info_first['op'] == 'Convolution' or info_first['op'] == 'FullyConnected') and info_second['op'] == 'BatchNorm':
      if info_second['bottom'][0] == info_first['top']:
        info_first['attrs']['no_bias'] = 'False'
        index_list = get_isinput_blob(info_list, info_second['name'])
        for index in index_list:
          info_list[index]['bottom'][0] = info_first['top']
        info_list.remove(info_second)
    elif info_first['op'] == 'BatchNorm':
      info_first['op'] = 'Scale'

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

      if str(node_i['op']) == 'null' and str(node_i['name']) != 'data':
        continue

      print('{}, \top:{}, name:{} -> {}'.format(i_node,node_i['op'].ljust(20),
                                          node_i['name'].ljust(30),
                                          node_i['name']).ljust(20))

      info = node_i
      info['top'] = info['name']
      info['bottom'] = []
      info['params'] = []
      for input_idx_i in node_i['inputs']:
        input_i = jdata['nodes'][input_idx_i[0]]
        if str(input_i['op']) != 'null' or (str(input_i['name']) == 'data'):
          info['bottom'].append(str(input_i['name']))
        if str(input_i['op']) == 'null':
          info['params'].append(str(input_i['name']))
          if not str(input_i['name']).startswith(str(node_i['name'])):
            print('           use shared weight -> %s'% str(input_i['name']))
            info['share'] = True

      info_list.append(info)

    if merge_bn:
      info_list = info_list_merge_bn(info_list)

    for info in info_list:
      write_node(prototxt_file, info)

create_prototxt(True)

