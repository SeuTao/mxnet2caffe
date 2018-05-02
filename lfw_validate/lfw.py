from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import lfw_validate.facenet as facenet

def evaluate(embeddings, actual_issame, nrof_folds=10):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 1, 0.001)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy = facenet.calculate_roc(thresholds, embeddings1, embeddings2,np.asarray(actual_issame), nrof_folds=nrof_folds)
    thresholds = np.arange(0, 1, 0.001)
    val_10e_1, _, far_10e_1 = facenet.calculate_val(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), 0.1, nrof_folds=nrof_folds)
    val_10e_2, _, far_10e_2= facenet.calculate_val(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), 0.01, nrof_folds=nrof_folds)
    val_10e_3, _, far_10e_3= facenet.calculate_val(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), 0.001, nrof_folds=nrof_folds)
    val_10e_4, _, far_10e_4= facenet.calculate_val(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), 0.0001, nrof_folds=nrof_folds)
    return tpr, fpr, accuracy, val_10e_1, far_10e_1, val_10e_2,far_10e_2, val_10e_3,far_10e_3, val_10e_4,far_10e_4

def get_paths(lfw_dir, pairs, file_ext):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    for pair in pairs:
        if len(pair) == 3:
            path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])+'.'+file_ext)
            path1 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])+'.'+file_ext)
            issame = True
        elif len(pair) == 4:
            path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])+'.'+file_ext)
            path1 = os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])+'.'+file_ext)
            issame = False
        if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
            path_list += (path0,path1)
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs>0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)
    
    return path_list, issame_list

def get_separate_paths(vis_dir, nir_dir, pairs, file_ext):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    for pair in pairs:
        if len(pair) == 3:
            path0 = os.path.join(vis_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1]) + '.' + file_ext)
            path1 = os.path.join(nir_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2]) + '.' + file_ext)
            issame = True
        elif len(pair) == 4:
            path0 = os.path.join(vis_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1]) + '.' + file_ext)
            path1 = os.path.join(nir_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3]) + '.' + file_ext)
            issame = False
        if os.path.exists(path0) and os.path.exists(path1):  # Only add the pair if both paths exist
            path_list += (path0, path1)
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs > 0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)

    return path_list, issame_list

def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs)



