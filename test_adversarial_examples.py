import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import models
from train_target_model import target_net

batch_size = 128
use_pretrained=True
feature_extract=True
use_cuda = True
model_name = 'densenet'
num_classes = 10

if __name__ == "__main__":
    # Define what device we are using
    use_cuda = True
    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    # load the pretrained model
    pretrained_model = "./target_model.pth"
    target_model, input_size = target_net(model_name,
                                          num_classes,
                                          feature_extract,
                                          use_pretrained).initialize_model()
    target_model = target_model.to(device)  # 除非forward否则精确写出到函数才能to(device)
    target_model.load_state_dict(torch.load(pretrained_model))
    target_model.eval()

    # load the generator of adversarial examples
    pretrained_generator_path = './models/netG_epoch_40.pth'
    pretrained_G = models.Generator().to(device)
    pretrained_G.load_state_dict(torch.load(pretrained_generator_path))
    pretrained_G.eval()

    # test adversarial examples in MNIST training dataset
    dataset_train = torchvision.datasets.CIFAR10('./dataset', train=True, transform=transforms.ToTensor(), download=True)
    train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, num_workers=1)
    num_correct = 0
    for i, data in enumerate(train_dataloader, 0):
        test_img, test_label = data
        test_img, test_label = test_img.to(device), test_label.to(device)
        perturbation = pretrained_G(test_img)
        perturbation = torch.clamp(perturbation, -0.3, 0.3)
        adv_img = perturbation + test_img
        adv_img = torch.clamp(adv_img, 0, 1)
        pred_lab = torch.argmax(target_model(adv_img),1)
        num_correct += torch.sum(pred_lab==test_label,0)

    print('training dataset:')
    print('num_correct: ', num_correct.item())   # 使用item()函数将num_correct转为numpy变量
    print('accuracy of adv imgs in training set: %f\n'%(num_correct.item()/len(dataset_train)))

    # test adversarial examples in MNIST testing dataset
    dataset_test = torchvision.datasets.CIFAR10('./dataset', train=False, transform=transforms.ToTensor(), download=True)
    test_dataloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=1)
    num_correct = 0
    for i, data in enumerate(test_dataloader, 0):
        test_img, test_label = data
        test_img, test_label = test_img.to(device), test_label.to(device)
        perturbation = pretrained_G(test_img)
        perturbation = torch.clamp(perturbation, -0.3, 0.3)
        adv_img = perturbation + test_img
        adv_img = torch.clamp(adv_img, 0, 1)
        pred_lab = torch.argmax(target_model(adv_img),1)
        num_correct += torch.sum(pred_lab==test_label,0)

    print('testing dataset:')
    print('num_correct: ', num_correct.item())
    print('accuracy of adv imgs in testing set: %f\n'%(num_correct.item()/len(dataset_test)))

