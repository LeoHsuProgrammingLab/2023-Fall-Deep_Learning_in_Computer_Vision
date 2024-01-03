from torchvision import transforms

train_tfm = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomGrayscale(0.1),
    transforms.ToTensor()
])

test_tfm = transforms.Compose([
    transforms.ToTensor()
])