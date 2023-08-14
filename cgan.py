import torch
import torch.optim as optim
from model import *
from transforms import *
from training_loop import *
from celeba_data import *
from torch.utils.data import DataLoader
import os
import argparse
from torchinfo import summary

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

classes = 3
topn = 1
checkpoint_dir = 'Model_Checkpoints/labels'
name = 'facegan'
batch_size = 32
gen_steps = 1
disc_steps = 1
epochs = 1
img_size = 256
lr = 0.0002
beta = 0.5
desired_attr = ['Male', 'Chubby', 'Bald']
label_size = len(desired_attr)

def train_model():
    imgtransform = BasicImageCropTransform(size = (img_size, img_size), scale = (1, 2))
    anntransform = celeb_label_transform(desired_attr)
    #transform = TransformWrapper(imgtransform, anntransform)
    dataset = CelebDS(imgtransform, anntransform)
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
