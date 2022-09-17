import load_data
#from .model.simpleNet import Net
from .model.vgg16_pre_trained import VGG16

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F

device_num = 0
train_loss = []
train_acc = []
val_loss = []
val_acc = []
train_epoch = []

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0
    acc = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        print('\rTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()), end='')

        pred = output.argmax(dim = 1, keepdim=True) #predicted answer
        correct += pred.eq(target.view_as(pred)).sum().item() #currect answer

    acc = round(correct / len(train_loader.dataset), 4)
    train_acc.append(acc)
    train_loss.append(loss.item())

    print('\nTrain acc: ', acc, 'Train loss: ', loss.item())
    

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)           
            test_loss += F.cross_entropy(output, target, reduction = 'sum').item()       
            pred = output.argmax(dim = 1, keepdim=True) #predicted answer
            correct += pred.eq(target.view_as(pred)).sum().item() #currect answer

    test_loss /= len(test_loader.dataset)
    acc = round(correct / len(test_loader.dataset), 4)
    val_loss.append(test_loss)
    val_acc.append(acc)
    print('\r\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * acc), end='')

def draw(title, xlabel, ylabel, x, y1, y2, filename):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.legend(['train', 'valid'], loc='upper left')
    plt.savefig(filename)

def main():
    train_dict, train_info_dict = load_data.read_image_folder((224, 224), "./train")
    whole_dataset = load_data.ImageDataset(train_dict["data"], train_dict["labels"])
    
    # Random split
    train_set_size = int(len(whole_dataset) * 0.8)
    valid_set_size = len(whole_dataset) - train_set_size
    train_set, valid_set = data.random_split(whole_dataset, [train_set_size, valid_set_size])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=32)

    with torch.cuda.device(device_num):
        #model = Net(num_classes = 12).cuda()
        model = VGG16(num_classes = 12).cuda()
        adam = torch.optim.Adam(model.parameters(), lr=1e-5)

        for epoch in range(1, 61):
            train_epoch.append(epoch)
            train(model, device_num, train_loader, adam, epoch)
            test(model, device_num, valid_loader)

        #print loss graph
        draw('train loss', 'epoch', 'loss', train_epoch, train_loss, val_loss,'VGG16_train_loss.jpg')
        draw('train accuracy', 'epoch', 'accuracy', train_epoch, train_acc, val_acc,'VGG16_train_acc.jpg')
        torch.save(model.state_dict(), 'VGG16.pt')

if __name__ == "__main__":
    main()
