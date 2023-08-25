import torch
import torch.optim as optim
import torchvision.transforms as transforms
from model import *
from transforms import *
from training_loop import *
from celeba_data import *
from sen12_data import *
from simple_season import *
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot
import pandas as pd
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

os.makedirs("results", exist_ok=True)
os.makedirs("/data/pgorry/checkpoints", exist_ok=True)
os.makedirs("/data/pgorry/losses", exist_ok=True)

checkpoint_dir = "/data/pgorry/checkpoints"
name = 'sen12_test'
batch_size = 32
gen_steps = 1
disc_steps = 1
epochs = 10
img_size = 256
lr = 0.0002
beta = 0.5
desired_attr = ['Attractive'] 
desired_season = ['fall']
label_size = len(desired_attr)
img_channels = 3

data_source = "/data/pgorry/sen12ms/s2"
source_labels = "/data/pgorry/sen12ms/seasons_labeled.csv"

imgtransform = BasicImageCropTransform(size = (img_size, img_size), scale = (1, 2))
#anntransform = celeb_label_transform(desired_attr)
#transform = TransformWrapper(imgtransform, anntransform)

transform_sen = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((img_size,img_size),antialias=False),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize([0.5 for _ in range(img_channels)],[0.5 for _ in range(img_channels)]),
    ])
anntransform_sen12 = sen12_label_transform(source_labels, desired_season)

dataset = SEN12MS(data_source, imgtransform, anntransform_sen12, "rgb")


#dataset = CelebDS(imgtransform, anntransform)

dataloader = DataLoader(dataset, batch_size, pin_memory = True)
print("Total Base Examples: " + str(len(dataset)))

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on a GPU")
else:
    device = torch.device("cpu")
    print("Running on a CPU")

def weights_init(m): 
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

generator = Generator(label_size).to(device)
generator.apply(weights_init)
discriminator = Discriminator(label_size).to(device)
discriminator.apply(weights_init)
optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(beta, 0.999))
optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta, 0.999))

gen_history, discrim_history = training_loop(dataloader, label_size, desired_attr, img_size, batch_size, 
                                             epochs, generator, discriminator, optimizerG, optimizerD, True,
                                             checkpoint_dir, name, gen_steps, disc_steps, device)