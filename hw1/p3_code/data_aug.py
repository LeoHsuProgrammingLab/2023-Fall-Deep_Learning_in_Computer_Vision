from torchvision import transforms

train_tfm = transforms.Compose([
    transforms.ToTensor(),
])

test_tfm = transforms.Compose([
    transforms.ToTensor(),
])