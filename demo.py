import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

from nets import Net, RelationNet
from config import args
from dataloader import Producer, loadImg, loadImg_testing
from utils import read_miniImageNet_pathonly, mean_confidence_interval, Dashboard
from Queue import Queue
import numpy as np
import threading
import os
import torchnet as tnt
import time
from tqdm import tqdm

def weighted_mse_loss(input, target, weight):
    return torch.sum(weight * (input - target) ** 2)

def main(args):
    '''
    main function
    '''

    if args.logport:
        args.logport = Dashboard(args.logport, 'dashboard')

    EPOCH_SIZE = args.num_episode*args.num_query*args.way_train
    EPOCH_SIZE_TEST = args.num_episode_test*args.num_query*args.way_test


    '''define network'''
    net = Net(args.num_in_channel, args.num_filter)
    relationNet = RelationNet(args.num_filter*2, args.num_filter, 5*5*args.num_filter, args.num_fc, args.drop_prob)
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

    scheduler = StepLR(optimizer, step_size=40, gamma=0.5)
    ''' training'''
    for epoch in range(1000):
        scheduler.step()

        running_loss = 0.0
        avg_accu_Train = 0.0
        accu_Test_stats = []

        net.train()
        relationNet.train()
        # epoch training list
        trainList_combo = Producer(trainList, args.way_train, args.num_episode, "training") # combo contains [query_label, query_path ]
        list_trainset = tnt.dataset.ListDataset(trainList_combo, loadImg)
        trainloader = list_trainset.parallel(batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

        for i, data in enumerate(tqdm(trainloader), 0):
        #for i, data in enumerate(trainloader, 0):

            # get inputs
            batchSize = data[0].size()[0]
            labels = torch.unsqueeze(data[0], 1)
            images = data[1:]
            images_all = torch.cat(images).permute(0, 3, 1, 2).float()

            labels_one_hot = torch.zeros(data[0].size()[0], args.way_train)
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
            feature_s_all_t0_p = torch.split(feature_s_all_t0_p, batchSize, 0)

            concatenatedFeat_list = [[] for _ in range(args.way_train)]
            for idx in range(args.way_train):
                concatenatedFeat_list[idx] = torch.cat((feature_s_all_t0_p[idx], feature_s_all_t0_p[-1]), 1)

            concatenatedFeat_all = torch.cat(concatenatedFeat_list, 0)
            relationScore_all = relationNet(concatenatedFeat_all)
            relationScore_list = torch.split(relationScore_all, batchSize, 0)
            relationScore = torch.cat(relationScore_list, 1)


            #loss = criterion(relationScore, labels_one_hot)
            weights = labels_one_hot.clone()
            weights[labels_one_hot == 0] = 1.0/(args.way_train)
            weights[labels_one_hot != 0] = (args.way_train-1.0)/(args.way_train)
            loss = weighted_mse_loss(relationScore, labels_one_hot, weights)/data[0].size()[0]
            loss.backward()
            optimizer.step()

            # summing up
            running_loss += loss.data[0]
            _, predicted = torch.max(relationScore.data, 1)
            labels = torch.squeeze(labels, 1)
            avg_accu_Train += (predicted == labels.cuda()).sum()
            if i % args.log_step == args.log_step-1:
                #print('[%d, %5d] train loss: %.3f  train accuracy: %.3f' % (epoch + 1, i + 1, running_loss / args.log_step, avg_accu_Train/(args.log_step*batchSize)))

                if args.logport:
                    args.logport.appendlog(running_loss / args.log_step, 'Training Loss')
                    args.logport.appendlog(avg_accu_Train/(args.log_step*batchSize), 'Training Accuracy')
                    args.logport.image((images[-1][0, :, :, :]).permute(2, 0, 1), 'query img', mode='img')
                    for idx in range(args.way_train):
                        args.logport.image((images[idx][0, :, :, :]).permute(2, 0, 1), 'support img', mode='img')

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
        relationNet.eval()
        # epoch training list
        testList_combo = Producer(testList, args.way_test, args.num_episode_test, "testing") # combo contains [query_label, query_path ]
        list_testset = tnt.dataset.ListDataset(testList_combo, loadImg_testing)
        testloader = list_testset.parallel(batch_size=args.batch_size_test, num_workers=args.num_workers, shuffle=False)
        #for i, data in enumerate(tqdm(testloader), 0):
        for i, data in enumerate(testloader, 0):
            # get inputs
            batchSize = data[0].size()[0]

            labels = torch.unsqueeze(data[0], 1)
            images = data[1:]
            images_all = torch.cat(images).permute(0, 3, 1, 2).float()

            labels_one_hot = torch.zeros(batchSize, args.way_test)
            labels_one_hot.scatter_(1, labels, 1.0)

            # wrap in Variable
            if torch.cuda.is_available():
                images_all, labels_one_hot = Variable(images_all.cuda(), volatile=True), Variable(labels_one_hot.cuda(), volatile=True)
            else:
                images_all, labels_one_hot = Variable(images_all, volatile=True), Variable(labels_one_hot, volatile=True)

            # forward
            feature_s_all_t0_p = net(images_all)
            feature_s_all_t0_p = torch.split(feature_s_all_t0_p, batchSize, 0)

            concatenatedFeat_list = [[] for _ in range(args.way_test)]
            for idx in range(args.way_test):
                concatenatedFeat_list[idx] = torch.cat((feature_s_all_t0_p[idx], feature_s_all_t0_p[-1]), 1)

            concatenatedFeat_all = torch.cat(concatenatedFeat_list, 0)
            relationScore_all = relationNet(concatenatedFeat_all)
            relationScore_list = torch.split(relationScore_all, batchSize, 0)
            relationScore = torch.cat(relationScore_list, 1)


            _, predicted = torch.max(relationScore.data, 1)
            #avg_accu_Test += (predicted == torch.squeeze(labels, 1).cuda()).sum()
            accu_Test_stats.append((predicted == torch.squeeze(labels, 1).cuda()).sum()/float(batchSize))


        m, h = mean_confidence_interval(np.asarray(accu_Test_stats), confidence=0.95)
        print('[epoch %3d] test accuracy with 0.95 confidence: %.4f, +-: %.4f' % (epoch + 1, m, h))

        #avg_accu_Test = 0.0
        accu_Test_stats = []


if __name__ == '__main__':
    main(args)