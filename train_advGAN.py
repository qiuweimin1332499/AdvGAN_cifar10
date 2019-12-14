import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from advGAN import AdvGAN_Attack
from train_target_model import target_net, input_size

use_pretrained=True
feature_extract=True
model_name = 'resnet'
num_classes = 10
image_nc= 3
epochs = 60
batch_size = 64
BOX_MIN = 0
BOX_MAX = 256
# train advGAN

if __name__ == "__main__":
    # Define what device we are using
    use_cuda = True
    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    pretrained_model = "./target_model.pth"
    target_modeled, input_size = target_net(model_name,
                                            num_classes,
                                            feature_extract,
                                            use_pretrained).initialize_model()
    targeted_model = target_modeled.to(device)  # 除非forward否则精确写出到函数才能to(device)
    targeted_model.load_state_dict(torch.load(pretrained_model))
    targeted_model.eval()


    # train dataset and dataloader declaration
    train_data_transform = transforms.Compose([transforms.RandomResizedCrop(input_size),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    dataset = torchvision.datasets.CIFAR10('./dataset', train=True, transform=train_data_transform, download=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    advGAN = AdvGAN_Attack(device, targeted_model, num_classes, BOX_MIN, BOX_MAX)
    advGAN.train(dataloader, epochs)
