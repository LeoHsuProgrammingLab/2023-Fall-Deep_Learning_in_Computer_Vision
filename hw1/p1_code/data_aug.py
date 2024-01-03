from torchvision import transforms
import matplotlib.pyplot as plt

test_tfm_B = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
])

test_tfm_A = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

train_tfm_B = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.RandomHorizontalFlip(p = 0.5),
    transforms.RandomGrayscale(p = 0.2),
    transforms.RandomRotation(30),
    transforms.GaussianBlur(3, sigma = (0.1, 2.0)),
    # transforms.ColorJitter(brightness = 0.5, contrast = 0.5, saturation = 0.5, hue = 0.5),
    # transforms.RandomCrop((60, 60), padding=8),
    transforms.ToTensor(),
    # transforms.Normalize(mean = [0.4157, 0.3942, 0.3590], std = [0.2465, 0.2378, 0.2289]),
])

train_tfm_A = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p = 0.5),
    transforms.RandomRotation(30),
    transforms.ToTensor()   
])

# write a function to show the data augmentation
def show_aug_img(train_set):
    axes = []
    fig = plt.figure()
    for i in range(6):
        axes.append(fig.add_subplot(2, 3, i+1))
        plt.imshow(train_set[i][0].permute(1, 2, 0))
    fig.tight_layout()
    plt.show()