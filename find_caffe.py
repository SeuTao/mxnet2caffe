try:
    import caffe
except ImportError:
    import os, sys
    curr_path = os.path.abspath(os.path.dirname(__file__))
    sys.path.append(os.path.join(curr_path, '/media/st/DATA01/Projects/sphereface-master/tools/caffe-sphereface/python'))
    import caffe
