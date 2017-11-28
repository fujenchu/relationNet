import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from nets import Net, RelationNet
from config import args
from dataloader import Producer
from utils import read_miniImageNet_pathonly
from Queue import Queue
import numpy as np
import threading
import os

def weighted_mse_loss(input, target, weight):
    return torch.sum(weight * (input - target) ** 2)

def main(args):
    '''
    main function
    '''
    EPOCH_SIZE = args.num_episode*args.num_query*args.way_train
    EPOCH_SIZE_TEST = args.num_episode*args.num_query*args.way_test


    '''define network'''
    net = Net(args.num_in_channel, args.num_filter)
    relationNet = RelationNet(args.num_filter*2, args.num_filter, 5*5*args.num_filter, args.num_fc)
    if torch.cuda.is_available():
        net.cuda()
        relationNet.cuda()


    '''
    load model if needed
    '''
    if args.model_load_path_net!='':
        net.load_state_dict(torch.load(args.model_load_path_net))
        net.cuda()
        relationNet.load_state_dict(torch.load(args.model_load_path_relationNet))
        relationNet.cuda()
        print('model loaded')


    ''' define loss, optimizer'''
    criterion = nn.MSELoss()
    params = list(net.parameters()) + list(relationNet.parameters())
    optimizer = optim.Adam(params, lr=args.learning_rate)


    '''get data'''
    trainList = read_miniImageNet_pathonly(TESTMODE=False,
                                           miniImageNetPath='/home/fujenchu/projects/dataset/miniImageNet_Ravi/',
                                           imgPerCls=600)
    testList = read_miniImageNet_pathonly(TESTMODE=True,
                                          miniImageNetPath='/home/fujenchu/projects/dataset/miniImageNet_Ravi/',
                                          imgPerCls=600)
    queue = Queue(maxsize=3)
    producer = threading.Thread(target=Producer, args=(queue, trainList, args.batch_size, EPOCH_SIZE, "training"))
    producer.start()
    queue_test = Queue(maxsize=100)
    producer_test = threading.Thread(target=Producer,args=(queue_test, testList, args.batch_size, EPOCH_SIZE_TEST, "testing"))
    producer_test.start()


    ''' training'''
    for epoch in range(1000):
        running_loss = 0.0
        avg_accu_Train = 0.0
        avg_accu_Test = 0.0
        total_batch = int(EPOCH_SIZE / args.batch_size)
        total_batch_test = int(EPOCH_SIZE_TEST / args.batch_size)

        for i in range(total_batch):
            # get inputs
            batch = queue.get()
            labels = torch.from_numpy(batch[0])
            images = batch[1:]
            images_all = torch.from_numpy(np.transpose(np.concatenate(images),(0,3,1,2))).float()

            labels_one_hot = torch.zeros(args.batch_size, args.way_train)
            labels_one_hot.scatter_(1, labels, 1.0)

            # wrap in Variable
            if torch.cuda.is_available():
                images_all, labels_one_hot = Variable(images_all.cuda()), Variable(labels_one_hot.cuda())
            else:
                images_all, labels_one_hot = Variable(images_all), Variable(labels_one_hot)

            # zero gradients
            optimizer.zero_grad()

            # forward + backward + optimizer
            feature_s_all_t0_p = net(images_all)
            feature_s_all_t0_p = torch.split(feature_s_all_t0_p, args.batch_size, 0)

            concatenatedFeat_list = [[] for _ in range(args.way_train)]
            for idx in range(args.way_train):
                concatenatedFeat_list[idx] = torch.cat((feature_s_all_t0_p[idx], feature_s_all_t0_p[-1]), 1)

            concatenatedFeat_all = torch.cat(concatenatedFeat_list, 0)
            relationScore_all = relationNet(concatenatedFeat_all)
            relationScore_list = torch.split(relationScore_all, args.batch_size, 0)
            relationScore = torch.cat(relationScore_list, 1)


            #loss = criterion(relationScore, labels_one_hot)
            weights = labels_one_hot.clone()
            weights[labels_one_hot == 0] = 1.0/(args.way_train)
            weights[labels_one_hot != 0] = (args.way_train-1.0)/(args.way_train)
            loss = weighted_mse_loss(relationScore, labels_one_hot, weights)
            loss.backward()
            optimizer.step()

            # summing up
            running_loss += loss.data[0]
            _, predicted = torch.max(relationScore.data, 1)
            labels = torch.squeeze(labels, 1)
            avg_accu_Train += (predicted == labels.cuda()).sum()
            if i % 1000 == 999:
                print('[%d, %5d] train loss: %.3f  train accuracy: %.3f' % (epoch + 1, i + 1, running_loss / 1000, avg_accu_Train/(1000*args.batch_size)))
                running_loss = 0.0
                avg_accu_Train = 0.0

            if (i+1) % args.save_step == 0:
                torch.save(net.state_dict(),
                           os.path.join(args.model_path,
                                        'net-model-%d-%d.pkl' %(epoch+1, i+1)))
                torch.save(relationNet.state_dict(),
                           os.path.join(args.model_path,
                                        'relationNet-model-%d-%d.pkl' %(epoch+1, i+1)))
        net.eval()
        for i in range(total_batch_test):
            # get inputs
            batch = queue_test.get()
            labels = torch.from_numpy(batch[0])
            images = batch[1:]
            images_all = torch.from_numpy(np.transpose(np.concatenate(images), (0, 3, 1, 2))).float()

            labels_one_hot = torch.zeros(args.batch_size, args.way_train)
            labels_one_hot.scatter_(1, labels, 1.0)

            # wrap in Variable
            if torch.cuda.is_available():
                images_all, labels_one_hot = Variable(images_all.cuda()), Variable(labels_one_hot.cuda())
            else:
                images_all, labels_one_hot = Variable(images_all), Variable(labels_one_hot)

            # forward
            feature_s_all_t0_p = net(images_all)
            feature_s_all_t0_p = torch.split(feature_s_all_t0_p, args.batch_size, 0)

            concatenatedFeat_list = [[] for _ in range(args.way_test)]
            for idx in range(args.way_test):
                concatenatedFeat_list[idx] = torch.cat((feature_s_all_t0_p[idx], feature_s_all_t0_p[-1]), 1)

            concatenatedFeat_all = torch.cat(concatenatedFeat_list, 0)
            relationScore_all = relationNet(concatenatedFeat_all)
            relationScore_list = torch.split(relationScore_all, args.batch_size, 0)
            relationScore = torch.cat(relationScore_list, 1)


            _, predicted = torch.max(relationScore.data, 1)
            avg_accu_Test += (predicted == torch.squeeze(labels, 1).cuda()).sum()

        print('test accuracy: %.3f' % (avg_accu_Test/(total_batch_test*args.batch_size)))
        avg_accu_Test = 0.0


if __name__ == '__main__':
    main(args)