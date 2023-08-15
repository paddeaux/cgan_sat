import torch
import torchvision.transforms as transforms
import torch.optim as optim
from model import *
from transforms import *
from training_loop import *
from celeba_data import *
from sen12_data import *
from torch.utils.data import DataLoader
import os
import argparse
from torchinfo import summary

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Make output directorys
os.makedirs("results", exist_ok=True)

topn = 1
checkpoint_dir = os.path.join(os.path.dirname(os.getcwd()), "checkpoints")
name = 'sen12_cgan_overfit'
batch_size = 32
gen_steps = 1
disc_steps = 1
epochs = 5
img_size = 256
lr = 0.0002
beta = 0.5
desired_season = ['fall']
label_size = len(desired_season)
data_source = "C:/Users/Paddy/CRT/Github/input/sen12_overfit/ROIs1158_spring_s2_1_p30.tif"
source_labels = "C:/Users/Paddy/CRT/Github/input/sen12_overfit/seasons_labeled_overfit.csv"
bands = "rgb" # or "rgb"
img_channels = 3

def train_model():
    transform_sen = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((img_size,img_size),antialias=False),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize([0.5 for _ in range(img_channels)],[0.5 for _ in range(img_channels)])
    ])
    anntransform = sen12_overfit_label_transform(source_labels, desired_season)
    dataset = SEN12MS_overfit(data_source, transform_sen, anntransform, "rgb", 64)
    dataloader = DataLoader(dataset, batch_size, pin_memory = True)
    print("Total Base Examples: " + str(len(dataset)))

    print("image dimensions:", dataset[0][0].shape)

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
    return gen_history, discrim_history

def model_summary():
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

    print("Generator Summary:")
    summary(generator, input_data=[torch.randn(1,100,1,1).to(device), torch.randn(1,2,1,1).to(device)])

    discriminator = Discriminator(label_size).to(device)
    discriminator.apply(weights_init)

    print("Discriminator Summary:")
    summary(discriminator, input_data=[torch.randn(1,3,256,256).to(device), torch.randn(1,2,256,256).to(device)])

def main():
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-m", "--mode", help="train or load a model: 'train' or 'load'")
    argParser.add_argument("-p", "--plot", help="plot example image: 'rgb' or 'full'")
    argParser.add_argument("-s", "--summary", help="display model structure summary")
    args = argParser.parse_args()
    
    # Choosing model mode
    if args.mode in ('train', 'TRAIN'):
        gen_history, discrim_history = train_model()
    elif args.mode in ('summary', 'SUMMARY'):
        model_summary()
    else:
        print("Invalid input: --mode <train OR load>")

    # Plotting sample
    if args.plot in ('rgb', 'RGB'):
        print("Not implemented.")
    elif args.plot in ('full', 'FULL'):
        print("Not implemented.")
    else:
        print("Invalid input: --plot rgb/full")

if __name__ == '__main__':
    main()
