import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np

def gradient_penalty(critic, gen, real_samples, fake_samples, labels, device):
    # Random weight term for interpolation between real and fake samples
    alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = critic(interpolates, labels)
    fake = torch.Tensor(real_samples.shape[0], 1).fill_(1.0).to(device)
    fake.requires_grad = False
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )
    gradients = gradients[0].view(gradients[0].size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def training_loop(dataloader, label_size, desired_attr, img_size, b_size, epochs, generator, discriminator, optimizerG, optimizerD, show_after_epoch = True,
                  checkpoint_directory = None, name = None, gen_steps = 1, disc_steps = 1, device = torch.device("cpu")):
    criterion = torch.nn.BCELoss()
    generator_history = []
    discriminator_history = []
    wass_history = []
    batch_size = b_size

    lambda_gp = 10
    channels = 3
    crit_repeats = 5

    for epoch in range(epochs):
        gen_loss = 0
        dis_loss = 0
        batch = tqdm(enumerate(dataloader), desc = "Epoch " + str(epoch), total = len(dataloader.dataset)//b_size)

        for i, (images, labels) in batch:
            #setup input
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.shape[0]

            #get labels that are the right shape and broadcast along 2 more dimensions for deconvolutional step
            broadcasted_labels = torch.zeros(batch_size, label_size, img_size, img_size, device = device)
            g_labels = labels.unsqueeze(-1).unsqueeze(-1).to(device)
            d_labels = broadcasted_labels + g_labels

            print("gen_labels:", g_labels.shape)
            print("d_labels:", d_labels.shape)

            for _ in range(crit_repeats):
                ##### Train Discriminator #####
                optimizerD.zero_grad()

                # fake loss
                z = torch.randn(batch_size, 100, device = device).view(-1, 100, 1, 1)
                fake_images = generator(z, g_labels)
                print("fake images shape:", fake_images.shape)
                fake_pred = discriminator(fake_images, d_labels)
                print("fake pred shape:", fake_pred.shape)
                d_loss_fake = torch.mean(fake_pred)

                # real loss
                real_pred = discriminator(images, d_labels)
                d_loss_real = -torch.mean(real_pred)

                gp = gradient_penalty(discriminator, generator, images, fake_images, labels, device)

                d_loss = d_loss_fake - d_loss_real
                was_loss = (d_loss_fake + d_loss_real) + lambda_gp*gp
                was_loss.backward()
                optimizerD.step()

                dis_loss += d_loss.item()/b_size

            ##### Train Generator #####
            optimizerG.zero_grad()

            z = torch.randn(batch_size, 100, device = device).view(-1, 100, 1, 1)
            fake_images = generator(z, g_labels)
            fake_pred = discriminator(fake_images, d_labels)
            g_loss = -torch.mean(fake_pred)
            g_loss.backward()
            optimizerG.step()

            gen_loss += g_loss.item()/b_size

            batch.set_postfix({'Disc loss': dis_loss, 'Gen loss': gen_loss,
                               'Wass loss': was_loss.item(), 'Grad Pen':gp.item()})
            
            #append losses to the histories
        generator_history.append(gen_loss)
        discriminator_history.append(dis_loss)
        wass_history.append(was_loss)

        if show_after_epoch:
            _, labels = next(iter(dataloader))
            g_labels = labels.unsqueeze(-1).unsqueeze(-1).to(device)
            noise = torch.randn(b_size, 100, device = device).view(-1, 100, 1, 1)
            gen_image = generator(noise, g_labels)
            images = [gen_image[i].squeeze().to('cpu').detach() for i in range(min(b_size, 8))]
            labels = [labels[i].squeeze().to('cpu').detach() for i in range(min(b_size, 8))]
            images = {'Image'+str(i): [image.numpy(), label.numpy()] for i, (image, label) in enumerate(zip(images, labels))}
            display_multiple_img(images, epoch, desired_attr, 2, 4)



        if checkpoint_directory:
            if not name:
                name = 'unnamed'
            torch.save({
                        'epoch': epoch,
                        'generator_state_dict': generator.state_dict(),
                        'discriminator_state_dict': discriminator.state_dict(),
                        'generator_optimizer_state_dict': optimizerG.state_dict(),
                        'discriminator_optimizer_state_dict': optimizerD.state_dict(),
                        }, checkpoint_directory + '/' + name + '_' + str(epoch) + '.pt')
        print("Saving loss history...")
        df_gen = pd.DataFrame(generator_history, columns=['gen_loss'])
        df_discrim = pd.DataFrame(discriminator_history, columns=['discrim_loss'])

        df_gen.to_csv(f"/data/pgorry/losses/gen_loss_history_epoch{epoch}.csv")
        df_discrim.to_csv(f"/data/pgorry/losses/discrim_loss_history_epoch{epoch}.csv")

    return generator_history, discriminator_history

def display_multiple_img(images, epoch, desired, rows = 2, cols= 4):
    fig, ax = plt.subplots(nrows=rows,ncols=cols, figsize = (30, 15))
    for ind,title in enumerate(images):
        ax.flatten()[ind].imshow((images[title][0].swapaxes(0, -1).swapaxes(0, 1) + 1) / 2)
        ax.flatten()[ind].set_title(f"{desired[0]}" if images[title][1] == 1 else f"Not {desired[0]}")
        ax.flatten()[ind].set_axis_off()
    plt.savefig('results/epoch' + str(epoch) + '.png')
