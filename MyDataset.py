import os
import random
import torch
import torchvision.transforms.functional as ttf
from torch.utils.data import Dataset
from PIL import Image


class MyTrainDataSet(Dataset):
    def __init__(self, inputPathTrain, targetPathTrain, patch_size=128):
        super(MyTrainDataSet, self).__init__()

        self.inputPath = inputPathTrain
        self.inputImages = os.listdir(inputPathTrain)

        self.targetPath = targetPathTrain
        self.targetImages = os.listdir(targetPathTrain)

        self.ps = patch_size

    def __len__(self):
        return len(self.targetImages)

    def __getitem__(self, index):

        ps = self.ps
        index = index % len(self.targetImages)

        inputImagePath = os.path.join(self.inputPath, self.inputImages[index])
        inputImage = Image.open(inputImagePath).convert('RGB')

        targetImagePath = os.path.join(self.targetPath, self.targetImages[index])
        targetImage = Image.open(targetImagePath).convert('RGB')

        inputImage = ttf.to_tensor(inputImage)
        targetImage = ttf.to_tensor(targetImage)

        hh, ww = targetImage.shape[1], targetImage.shape[2]

        rr = random.randint(0, hh-ps)
        cc = random.randint(0, ww-ps)

        input_ = inputImage[:, rr:rr+ps, cc:cc+ps]
        target = targetImage[:, rr:rr+ps, cc:cc+ps]

        return input_, target


class MyValueDataSet(Dataset):
    def __init__(self, inputPathTrain, targetPathTrain, patch_size=128):
        super(MyValueDataSet, self).__init__()

        self.inputPath = inputPathTrain
        self.inputImages = os.listdir(inputPathTrain)

        self.targetPath = targetPathTrain
        self.targetImages = os.listdir(targetPathTrain)

        self.ps = patch_size

    def __len__(self):
        return len(self.targetImages)

    def __getitem__(self, index):

        ps = self.ps
        index = index % len(self.targetImages)

        inputImagePath = os.path.join(self.inputPath, self.inputImages[index])
        inputImage = Image.open(inputImagePath).convert('RGB')

        targetImagePath = os.path.join(self.targetPath, self.targetImages[index])
        targetImage = Image.open(targetImagePath).convert('RGB')

        inputImage = ttf.center_crop(inputImage, (ps, ps))
        targetImage = ttf.center_crop(targetImage, (ps, ps))

        input_ = ttf.to_tensor(inputImage)
        target = ttf.to_tensor(targetImage)

        return input_, target


class MyTestDataSet(Dataset):
    def __init__(self, inputPathTest):
        super(MyTestDataSet, self).__init__()

        self.inputPath = inputPathTest
        self.inputImages = os.listdir(inputPathTest)

    def __len__(self):
        return len(self.inputImages)

    def __getitem__(self, index):
        index = index % len(self.inputImages)

        inputImagePath = os.path.join(self.inputPath, self.inputImages[index])
        inputImage = Image.open(inputImagePath).convert('RGB')

        input_ = ttf.to_tensor(inputImage)

        return input_


class MyComDataSet(Dataset):
    def __init__(self, resultPathTrain, targetPathTrain):
        super(MyComDataSet, self).__init__()

        self.resultPath = resultPathTrain
        self.resultImages = os.listdir(resultPathTrain) 

        self.targetPath = targetPathTrain
        self.targetImages = os.listdir(targetPathTrain)

    def __len__(self):
        return len(self.targetImages)

    def __getitem__(self, index):

        resultImagePath = os.path.join(self.resultPath, self.resultImages[index])
        resultImage = Image.open(resultImagePath).convert('L')

        targetImagePath = os.path.join(self.targetPath, self.targetImages[index])
        targetImage = Image.open(targetImagePath).convert('L')

        result = ttf.to_tensor(resultImage)
        target = ttf.to_tensor(targetImage)

        return result, target
