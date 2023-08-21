import torch
import torch.optim as optim
import torchvision.transforms as transforms
from model import *
from transforms import *
from training_loop import *
from celeba_data import *
from sen12_data import *
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot
import pandas as pd
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

checkpoint_dir = os.path.join(os.path.dirname(os.getcwd()), "checkpoints")
name = 'celebA50k_attr_40'
batch_size = 128
gen_steps = 1
disc_steps = 1
epochs = 1
img_size = 256
lr = 0.0001
beta = 0.5
desired_attr = ['Attractive','Bangs'] 
desired_season = ['fall']
label_size = len(desired_attr)
img_channels = 3

data_source = "C:/Users/Paddy/CRT/Github/input/SEN12MS"
source_labels = "C:/Users/Paddy/CRT/Github/input/SEN12MS/seasons_labeled_spring.csv"

imgtransform = BasicImageCropTransform(size = (img_size, img_size), scale = (1, 2))
anntransform = celeb_label_transform(desired_attr)
#transform = TransformWrapper(imgtransform, anntransform)

transform_sen = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((img_size,img_size),antialias=False),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize([0.5 for _ in range(img_channels)],[0.5 for _ in range(img_channels)]),
    ])
anntransform_sen12 = sen12_label_transform(source_labels, desired_season)

imgtransform = BasicImageCropTransform(size = (img_size, img_size), scale = (1, 2))

#dataset = SEN12MS(data_source, imgtransform, anntransform_sen12, "rgb")


dataset = CelebDS(imgtransform, anntransform)

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

gen_history, discrim_history = training_loop(dataloader, label_size, img_size, batch_size, 
                                             epochs, generator, discriminator, optimizerG, optimizerD, True,
                                             checkpoint_dir, name, gen_steps, disc_steps, device)

df_gen = pd.DataFrame(gen_history, columns=['gen_loss'])
df_discrim = pd.DataFrame(discrim_history, columns=['discrim_loss'])

df_gen.to_csv("gen_loss_history.csv")
df_discrim.to_csv("gen_loss_history.csv")