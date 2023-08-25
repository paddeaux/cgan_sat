import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

def training_loop(dataloader, label_size, desired_attr, img_size, b_size, epochs, generator, discriminator, optimizerG, optimizerD, show_after_epoch = True,
                  checkpoint_directory = None, name = None, gen_steps = 1, disc_steps = 1, device = torch.device("cpu")):
    criterion = torch.nn.BCELoss()
    def wasserstein_loss(y_true, y_pred):
        return torch.mean(y_true * y_pred)
    generator_history = []
    discriminator_history = []
    batch_size = b_size

    for epoch in range(epochs):
        batch = tqdm(enumerate(dataloader), desc = "Epoch " + str(epoch), total = len(dataloader.dataset)//b_size)

        #learning rate decay
        if epoch == 5 or epoch == 10:
            optimizerG.param_groups[0]['lr'] /= 10
            optimizerD.param_groups[0]['lr'] /= 10

        for i, (images, labels) in batch:
            #setup input
            images = images.to(device)
            batch_size = images.shape[0]
            y_real = torch.full((batch_size,), 1, dtype=torch.float, device=device)
            y_fake = torch.full((batch_size,), 0, dtype=torch.float, device=device)

            #get labels that are the right shape and broadcast along 2 more dimensions for deconvolutional step
            broadcasted_labels = torch.zeros(batch_size, label_size, img_size, img_size, device = device)
            g_labels = labels.unsqueeze(-1).unsqueeze(-1).to(device)
            d_labels = broadcasted_labels + g_labels

            #discriminator training
            disc_losses = []
            disc_accuracies = []
            for i in range(disc_steps):
                discriminator.zero_grad()

                #get the loss on real images (and metrics)
                D_out_real = discriminator(images, d_labels).squeeze()
                D_loss_real = criterion(D_out_real, y_real)
                #D_loss_real = wasserstein_loss(D_out_real, y_real)

                real_accuracy = torch.mean(1 - torch.abs(D_out_real - y_real)).item()

                #get the loss on fake images (and metrics)
                noise = torch.randn(batch_size, 100, device = device).view(-1, 100, 1, 1)
                gen_image = generator(noise, g_labels)

                D_out_fake = discriminator(gen_image, d_labels).squeeze()
                D_loss_fake = criterion(D_out_fake, y_fake)
                #D_loss_fake = wasserstein_loss(D_out_fake, y_fake)

                fake_accuracy = torch.mean(1 - torch.abs(D_out_fake - y_fake)).item()

                #add losses and backprop
                D_loss = D_loss_real + D_loss_fake
                D_loss.backward()
                optimizerD.step()

                #recording for metrics
                disc_losses.append(D_loss.item())
                disc_accuracies.append((real_accuracy, fake_accuracy))

            #generator training
            gen_losses = []
            for i in range(gen_steps):
                generator.zero_grad()
                
                #get discriminator predictions on faked images, and take loss between real y
                noise = torch.randn(batch_size, 100, device = device).view(-1, 100, 1, 1)
                gen_image = generator(noise, g_labels)

                D_out_gen = discriminator(gen_image, d_labels).squeeze()
                gen_loss = criterion(D_out_gen, y_real)
                #gen_loss = wasserstein_loss(D_out_gen, y_real)

                #backprop and record metric
                gen_loss.backward()
                optimizerG.step()

                gen_losses.append(gen_loss.item())

            batch.set_postfix({'Disc loss': torch.mean(torch.tensor(disc_losses[-disc_steps:])).item(),
                  'Gen loss': torch.mean(torch.tensor(gen_losses[-disc_steps:])).item(),
                  'Disc Real Accuracy': torch.mean(torch.tensor([disc_accuracies[-i-1][0] for i in range(disc_steps)])).item(),
                  'Disc Fake Accuracy': torch.mean(torch.tensor([disc_accuracies[-i-1][1] for i in range(disc_steps)])).item()})
            
            #append losses to the histories
            generator_history.append(gen_losses)
            discriminator_history.append(disc_losses)

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
        df_discrim.to_csv(f"/data/pgorrylosses/discrim_loss_history_epoch{epoch}.csv")

    return generator_history, discriminator_history

def display_multiple_img(images, epoch, desired, rows = 2, cols= 4):
    fig, ax = plt.subplots(nrows=rows,ncols=cols, figsize = (30, 15))
    for ind,title in enumerate(images):
        ax.flatten()[ind].imshow((images[title][0].swapaxes(0, -1).swapaxes(0, 1) + 1) / 2)
        ax.flatten()[ind].set_title(f"{desired[0]}" if images[title][1] == 1 else f"Not {desired[0]}")
        ax.flatten()[ind].set_axis_off()
    plt.savefig('results/epoch' + str(epoch) + '.png')
