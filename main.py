import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
import utils
from loss import SSIM, EdgeLoss
from NetModel import Net
from MyDataset import *


if __name__ == '__main__':

    random.seed(1234)
    torch.manual_seed(1234)
    EPOCH = 100
    BATCH_SIZE = 18
    LEARNING_RATE = 1e-3
    lr_list = []
    loss_list = []

    inputPathTrain = './data/inputTrain/'
    targetPathTrain = './data/targetTrain/'
    inputPathTest = './data/inputTest/'
    resultPathTest = './data/resultTest/'
    targetPathTest = './data/targetTest/'
    best_psnr = 0
    best_epoch = 0

    myNet = Net()
    myNet = myNet.cuda()

    criterion1 = SSIM().cuda()
    # criterion2 = nn.MSELoss().cuda()
    criterion3 = EdgeLoss().cuda()

    psnr = utils.PSNR()
    psnr = psnr.cuda()
    ssim = utils.SSIM()
    ssim = ssim.cuda()

    optimizer = optim.Adam(myNet.parameters(), lr=LEARNING_RATE)
    scheduler = MultiStepLR(optimizer, milestones=[30, 50, 80], gamma=0.2)

    datasetTrain = MyTrainDataSet(inputPathTrain, targetPathTrain)
    trainLoader = DataLoader(dataset=datasetTrain, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=6,
                             pin_memory=True)

    datasetValue = MyValueDataSet(inputPathTest, targetPathTest)
    valueLoader = DataLoader(dataset=datasetValue, batch_size=16, shuffle=True, drop_last=True, num_workers=6,
                             pin_memory=True)

    datasetTest = MyTestDataSet(inputPathTest)
    testLoader = DataLoader(dataset=datasetTest, batch_size=1, shuffle=False, drop_last=False, num_workers=6,
                            pin_memory=True)

    datasetCom = MyComDataSet(resultPathTest, targetPathTest)
    comLoader = DataLoader(dataset=datasetCom, batch_size=1, shuffle=False, drop_last=False, num_workers=6,
                           pin_memory=True)

    print('-------------------------------------------------------------------------------------------------------')
    if os.path.exists('./model_best.pth'):
        myNet.load_state_dict(torch.load('./model_best.pth'))

    for epoch in range(EPOCH):
        myNet.train()
        iters = tqdm(trainLoader, file=sys.stdout)
        epochLoss = 0
        timeStart = time.time()
        for index, (x, y) in enumerate(iters, 0):

            myNet.zero_grad()
            optimizer.zero_grad()

            input_train, target = Variable(x).cuda(), Variable(y).cuda()

            output_train = myNet(input_train)

            l_ssim = np.sum([criterion1(output_train[i], target) for i in range(len(output_train))])
            # l_2 = np.sum([criterion2(output_train[i], target) for i in range(len(output_train))])
            l_edge = np.sum([criterion3(output_train[i], target) for i in range(len(output_train))])

            # loss = (1 - l_ssim) + l_2
            loss = (1 - l_ssim) + 0.05*l_edge

            loss.backward()
            optimizer.step()
            epochLoss += loss.item()

            iters.set_description('Training !!!  Epoch %d / %d,  Batch Loss %.6f' % (epoch+1, EPOCH, loss.item()))

        myNet.eval()
        psnr_val_rgb = []
        for index, (x, y) in enumerate(valueLoader, 0):
            input_, target_value = x.cuda(), y.cuda()
            with torch.no_grad():
                output_value = myNet(input_)
            for output_value, target_value in zip(output_value[1], target_value):
                psnr_val_rgb.append(psnr(output_value, target_value))

        psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()

        if psnr_val_rgb > best_psnr:
            best_psnr = psnr_val_rgb
            best_epoch = epoch
            torch.save(myNet.state_dict(), 'model_best.pth')

        loss_list.append(epochLoss)
        lr_list.append(scheduler.get_last_lr())
        scheduler.step()
        torch.save(myNet.state_dict(), 'model.pth')
        timeEnd = time.time()
        print("------------------------------------------------------------")
        print("Epoch:  {}  Finished,  Time:  {:.4f} s,  Loss:  {:.6f}.".format(epoch+1, timeEnd-timeStart, epochLoss))
        print('-------------------------------------------------------------------------------------------------------')
    print("Training Process Finished ! Best Epoch : {} , Best PSNR : {:.2f}".format(best_epoch, best_psnr))

    print('--------------------------------------------------------------')
    myNet.load_state_dict(torch.load('./model_best.pth'))
    myNet.eval()

    with torch.no_grad():
        timeStart = time.time()
        for index, x in enumerate(tqdm(testLoader, desc='Testing !!! ', file=sys.stdout), 0):
            torch.cuda.empty_cache()
            input_test = x.cuda()
            output_test = myNet(input_test)
            save_image(output_test[1], resultPathTest + str(index+1).zfill(4) + tail)
        timeEnd = time.time()
        print('---------------------------------------------------------')
        print("Testing Process Finished !!! Time: {:.4f} s".format(timeEnd - timeStart))

    plt.figure(1)
    x = range(0, EPOCH)
    plt.xlabel('epoch')
    plt.ylabel('epoch loss')
    plt.plot(x, loss_list, 'r-')
    plt.figure(2)
    plt.xlabel('epoch')
    plt.ylabel('learning rate')
    plt.plot(x, lr_list, 'r-')

    plt.show()





