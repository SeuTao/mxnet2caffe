
import sys
sys.path.insert(0, '/media/st/DATA01/Projects/sphereface-master/tools/caffe-sphereface/python')
import caffe
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from utils import *
import lfw_validate.lfw as lfw
import numpy as np
import cv2

transformer = caffe.io.Transformer({'data': (1, 1, 128, 128)})
transformer.set_mean('data', np.array([0]))
transformer.set_raw_scale('data', 255.0)

def load_image(imgfile):
    image = caffe.io.load_image(imgfile, color=False)
    image = transformer.preprocess('data', image)
    image = image.reshape(1, 1, 128, 128)
    return image

def load_model(protofile, weightfile, is_cuda = False):
    if is_cuda:
        caffe.set_device(0)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()
    net = caffe.Net(protofile, weightfile, caffe.TEST)
    return net

def extract_feature(net,  img_path):
    print('Runnning forward pass on image')

    image = load_image(img_path)
    net.blobs['data'].data[...] = image

    output = net.forward()

    # bn_data = net.blobs['bn_data'].data
    # conv0_data = net.blobs['batchnorm2'].data
    #
    # concat0_data = net.blobs['concat0'].data
    # concat1_data = net.blobs['concat1'].data
    # plus0_data = net.blobs['_plus0'].data
    # concat2_data = net.blobs['concat2'].data
    #
    # pooling1 = net.blobs['pooling1'].data


    bn1 = net.blobs['pre_fc1'].data
    bn1 = bn1.reshape([-1])

    for i in range(256):
        print(str(i)+' ',bn1[i])

    # print(bn1)

    # pre_fc1 = net.blobs['pre_fc1'].data
    # pre_fc1 = pre_fc1.reshape([-1])
    # print(pre_fc1)


    # pre_fc1 = net.blobs['pre_fc1'].data

    # pre_fc1_bn = net.blobs['pre_fc1_bn'].data

    # stage4_unit3_conv3_data = net.blobs['stage4_unit3_conv3'].data
    # activation0_data =  net.blobs['activation0'].data
    # pooling0_data =  net.blobs['pooling0'].data
    #
    # _plus0_data = net.blobs['_plus0'].data
    # _plus1_data = net.blobs['_plus1'].data
    # _plus15_data = net.blobs['_plus15'].data
    #
    # pre_fc1_data = net.blobs['pre_fc1'].data

    # vis_square(feat)
    # np_features = np.array(output['bn_data'])

def check(protofile, weightfile):
    net = load_model(protofile, weightfile, is_cuda=True)
    image_path = r'./0001.jpg'
    extract_feature(net, image_path)

def extract_features(net, image_lists):
    paths = image_lists
    print('Runnning forward pass on images')
    nrof_images = len(paths)
    print(nrof_images)
    nrof_batches = int((nrof_images))

    feature_list = []

    for i in range(nrof_batches):
        if i % 1000 == 0 and i > 0:
            print(i)

        img_path = paths[i]

        image = load_image(img_path)
        net.blobs['data'].data[...] = image

        output = net.forward()
        np_features = np.array(output['pre_fc1'])
        feature_list.append(np_features)

    return feature_list

def default_list_reader(fileList):
    imgList = []
    with open(fileList, 'r') as file:
        for line in file.readlines():
            line = line.strip()
            # print(line)
            imgList.append(line)

    return imgList

def get_features( protofile, weightfile, output_name, emb_size = 512 ):

    net = load_model(protofile, weightfile, is_cuda=True)
    path = '/media/st/SSD01/DATA/TEST/zhongkong_align_128'
    import os
    img_list = os.listdir(path)
    img_list = [os.path.join(path, tmp) for tmp in img_list]
    # print(img_list)

    feature_list = extract_features(net, img_list)

    output = open(output_name, 'w')

    for i in range(len(img_list)):
        img_path = img_list[i]
        feature = feature_list[i]

        output.write(img_path.replace(r'/media/st/SSD01/DATA/TEST/zhongkong_align_128','zhongkong') + '\t')
        for j in range(emb_size-1):
            output.write('%4f'%feature[0,j]+ '\t')
        output.write(str('%4f'%feature[0,emb_size-1]) + '\n')

def validate_on_lfw(net, embedding_size, paths, actual_issame, lfw_nrof_folds=10):
        # Load the model
        print('Runnning forward pass on images')
        nrof_images = len(paths)
        print(nrof_images)
        nrof_batches = int((nrof_images))
        emb_array = np.zeros((nrof_images, embedding_size))

        for i in range(nrof_batches):
            img_path = paths[i]

            image = load_image(img_path)
            net.blobs['data'].data[...] = image

            output = net.forward()
            np_features = output['pre_fc1']

            norm = np.sqrt(max((np_features ** 2).sum(), 0.00001))
            np_features = np_features / norm
            emb_array[i:i + 1, :] = np_features

        tpr, fpr, accuracy, val1, far1, val2, far2, val3, far3, val4, far4 = lfw.evaluate(emb_array, actual_issame,
                                                                                          nrof_folds=lfw_nrof_folds)

        print('Accuracy: %1.4f+-%1.4f' % (np.mean(accuracy), np.std(accuracy)))
        print('TAR @ FAR(%2.4f) : %2.5f' % (far1, val1))
        print('TAR @ FAR(%2.4f) : %2.5f' % (far2, val2))
        print('TAR @ FAR(%2.4f) : %2.5f' % (far3, val3))
        print('TAR @ FAR(%2.4f) : %2.5f' % (far4, val4))

def val_lfw(protofile, weightfile, embedding_size=512):

        net = load_model(protofile, weightfile, is_cuda=True)

        print('lfw_test')
        lfw_dir = r'/media/st/SSD01/DATA/TEST/lfw_align_rgb'
        lfw_pairs = r'/media/st/SSD01/DATA/TEST/pairs.txt'
        lfw_file_ext = 'jpg'
        pairs = lfw.read_pairs(lfw_pairs)
        paths, actual_issame = lfw.get_paths(os.path.expanduser(lfw_dir), pairs, lfw_file_ext)
        validate_on_lfw(net, embedding_size, paths, actual_issame, lfw_nrof_folds=10)

if __name__ == '__main__':
    path = r'/media/st/DATA01/Projects/MXNet2Caffe_home/model_caffe/'

    protofile =  path + r'0502_resnet50_thin_400k_1.2_0.4_0.0_125k.prototxt'
    weightfile = path + r'0502_resnet50_thin_400k_1.2_0.4_0.0_125k.caffemodel'

    output_name = r'0502_resnet50_thin_400k_1.2_0.4_0.0_125k'
    get_features(protofile, weightfile, output_name, 512)
    val_lfw(protofile, weightfile, 512)





