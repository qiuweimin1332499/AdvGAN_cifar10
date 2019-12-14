import torch
import torchvision.datasets
from torchvision import datasets,models,transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from models import target_net
import time
import copy
import torch.optim as optim
import torch.nn as nn
import numpy as np

feature_extract = False
use_pretrained = True
model_name = 'densenet'
num_classes = 10
batch_size = 256
workers = 1
input_size = 32   # 数据大小

if __name__ == "__main__":
    # Define what device we are using
    use_cuda = True
    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    # train dataset
    train_data_transform = transforms.Compose([transforms.RandomResizedCrop(input_size),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    train_dataset = torchvision.datasets.CIFAR10('./dataset', train=True, transform=train_data_transform,
                                                 download=False)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    # test dataset
    val_data_transform = transforms.Compose([transforms.Resize(input_size),
                                             transforms.CenterCrop(input_size),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    test_dataset = torchvision.datasets.CIFAR10('./dataset', train=True, transform=val_data_transform, download=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    # training the target model
    target_model, input_size = target_net(model_name,
                                          num_classes,
                                          feature_extract,
                                          use_pretrained).initialize_model()
    target_model = target_model.to(device)  # 除非forward否则精确写出到函数才能to(device)
    target_model.train()  # set the model in train mode
    opt_model = torch.optim.Adam(target_model.parameters(), lr=0.001)
    epochs = 120
    for epoch in range(epochs):
        loss_epoch = 0
        if epoch == 80:
            opt_model = torch.optim.Adam(target_model.parameters(), lr=0.0001)
        for i, data in enumerate(train_dataloader, 0):
            train_imgs, train_labels = data
            train_imgs, train_labels = train_imgs.to(device), train_labels.to(device)
            logits_model = target_model(train_imgs)
            loss_model = F.cross_entropy(logits_model, train_labels)
            loss_epoch += loss_model
            opt_model.zero_grad()
            loss_model.backward()
            opt_model.step()
        print('loss in epoch %d: %f' % (epoch, loss_epoch))

    # save model
    targeted_model_file_name = './target_model.pth'
    torch.save(target_model.state_dict(), targeted_model_file_name)
    target_model.eval()  # set the model in eval mode

    num_correct = 0
    # testing the target model
    for i, data in enumerate(test_dataloader, 0):
        test_img, test_label = data
        test_img, test_label = test_img.to(device), test_label.to(device)
        pred_lab = torch.argmax(target_model(test_img), 1)
        num_correct += torch.sum(pred_lab==test_label,0)
    num_correct=num_correct.cuda().data.cpu().numpy()  # num_correct先转为cpu变量再转为numpy变量
    print('accuracy in testing set: %f\n'%(num_correct/len(test_dataset)))
