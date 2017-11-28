import glob
import numpy as np
#import cv2
import matplotlib.pyplot as plt
import zipfile

def read_miniImageNet_pathonly(TESTMODE = False, miniImageNetPath = '/media/fujenchu/dataset/miniImageNet/', imgPerCls = 600):
  '''
  input:  
    TESTMODE = False for training 
    TESTMODE = True for testing

  output:
    one list of data, each element is a (imgPerCls,1) array for image path of one class  
  '''

  if TESTMODE == True:
    dataType = 'test'
  else:
    dataType = 'trainval'

  # get training data
  trainCharListAll = glob.glob(miniImageNetPath + str(dataType) +'/*')

  train_images = []
  # per character class
  for (trainCharPath, clsIdx) in zip(trainCharListAll, range(len(trainCharListAll))):
    trainCharRepliList = glob.glob(trainCharPath + '/*')
    # append to list
    train_images.append(trainCharRepliList)

  return train_images



def get_combination_miniImageNet_5way1shot_random_pathonly_episode_variableWays(trainList, visualize=False, episode_num=1000, ways=5,
                                                                   query_num=15):
  nSample = episode_num * query_num * ways
  imgPerCls = 600

  # prepare 5-way 1-shot dataflow
  # train_images_lists[-1] is  train_images_t0
  # train_images_lists[0] is train_images_s0
  # train_images_lists[1] is train_images_s1
  train_images_lists = [[] for _ in range(ways+1)]
  trueLabelSet_list = []

  # per episode
  for episode_idx in range(episode_num):
    # random choose 5 classes
    charIdx = range(len(trainList))
    np.random.shuffle(charIdx)
    charIdx = charIdx[:ways]

    # random choose 1 for support set, and choose query_num for query
    choseIdx_lists = [[] for _ in range(ways)]
    for idx in range(ways):
        choseIdx = range(imgPerCls)
        np.random.shuffle(choseIdx)
        choseIdx_lists[idx] = choseIdx[:query_num + 1]

    supportSet = []
    for idx in range(ways):
        supportSet.append(trainList[charIdx[idx]][choseIdx_lists[idx][query_num]])


    # get each query and shuffle with selected supportSet
    for query_cls_idx in range(ways):
      for query_idx in range(query_num):
        cls = charIdx[query_cls_idx]
        cls_idx = choseIdx_lists[query_cls_idx][query_idx]
        train_images_lists[-1].append(trainList[cls][cls_idx])

        # true label is query_cls_idx
        trueLabel = np.random.randint(ways)
        trueLabelSet_list.append(trueLabel)

        supportSet_copy = supportSet[:]
        tmp = supportSet_copy[query_cls_idx]
        del supportSet_copy[query_cls_idx]
        np.random.shuffle(supportSet_copy)
        supportSet_copy.insert(trueLabel, tmp)

        for idx in range(ways):
          train_images_lists[idx].append(supportSet_copy[idx])

  tie = list(zip(trueLabelSet_list, *train_images_lists))
  np.random.shuffle(tie)
  trueLabel_supportSet_query = zip(*tie)

  trueLabelSet = np.asarray(trueLabel_supportSet_query[0])
  trueLabel_supportSet_query[0] = trueLabelSet.reshape(nSample, 1)

  return trueLabel_supportSet_query