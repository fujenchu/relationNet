import cv2
import numpy as np
from config import args
from utils import get_combination_miniImageNet_5way1shot_random_pathonly_episode_variableWays



def randomCrop(img, centralCrop=False):
  '''
  perform random crop and normalize an image
  :param 
     img: original image 
     centralCrop: crop from center if True, default False
  :return
     imgNormalize: random Cropped / normalized image
  '''
  dx = np.random.randint(4)
  dy = np.random.randint(4)

  if centralCrop is False:
      imgNormalize = img[dx:dx+80, dy:dy+80, :]
  else:
      imgNormalize = img[2:82,2:82,:]

  imgNormalize = imgNormalize.astype('float64')
  imgNormalize_R = (imgNormalize[:, :, 0]  / 255.0 - 0.4710) / 0.2741 #R
  imgNormalize_G = (imgNormalize[:, :, 1]  / 255.0 - 0.4498) / 0.2661 #G
  imgNormalize_B = (imgNormalize[:, :, 2]  / 255.0 - 0.4034) / 0.2809 #B

  imgNormalize[:, :, 0] = imgNormalize_R
  imgNormalize[:, :, 1] = imgNormalize_G
  imgNormalize[:, :, 2] = imgNormalize_B

  return imgNormalize


def Producer(list, ways_num, episode_num, mode):
    #print ('generating ' + mode + ' combination.. ')
    # [0] is trueLabel; [-1] is t0;
    trueLabel_supportSet_query = get_combination_miniImageNet_5way1shot_random_pathonly_episode_variableWays(list, visualize=False,episode_num=episode_num, ways=ways_num, query_num=args.num_query)
    return trueLabel_supportSet_query


def loadImg(sample_list):
    mode = "training"
    train_images_data = []
    for way in range(1, len(sample_list)):
        img = cv2.imread(sample_list[way])
        RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = RGB_img
        if np.random.choice([True, False]) and mode is "training":
            img = cv2.flip(RGB_img,1)
        imgNormalize = randomCrop(img, centralCrop=args.central_crop)
        train_images_data.append(imgNormalize.reshape(80, 80, 3))

    train_images_data.insert(0, sample_list[0])
    return tuple(train_images_data)


def loadImg_testing(sample_list):
    mode = "testing"
    train_images_data = []
    for way in range(1, len(sample_list)):
        img = cv2.imread(sample_list[way])
        RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = RGB_img
        if np.random.choice([True, False]) and mode is "training":
            img = cv2.flip(RGB_img, 1)
        imgNormalize = randomCrop(img, centralCrop=args.central_crop)
        train_images_data.append(imgNormalize.reshape(80, 80, 3))

    train_images_data.insert(0, sample_list[0])
    return tuple(train_images_data)
