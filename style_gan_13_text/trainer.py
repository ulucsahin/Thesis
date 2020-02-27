"""
This method trains NVidia's StyleGAN.
Implementation taken from https://github.com/SiskonEmilia/StyleGAN-PyTorch and edited.
"""


import os
from torchvision import transforms
import torch
from style_gan_13_text.model import StyleBased_Generator, Discriminator
import torch.optim as optim
from tqdm import tqdm
from PIL import Image
import numpy as np

# my code
from json_manager import JSONManager
import data_manager
from args import Args
from embedding_manager import Embedder
from losses import CondWGAN_GP
from util import create_grid

# Parameters
learning_rate = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
device = "cuda"
img_size = 1024 # max image size.
batch_size = 8
n_fc = 8
dim_latent = 300+100
dim_input = 4
max_step = 7
n_sample = 600000  # number of samples to show before doubling resolution
n_sample_total = 10_000_000  # number of samples train model in total
DGR = 1 # n_critic value
n_show_loss = 360
n_gpu = 1
is_continue = True

save_folder_path = "E:/TezOutputs/StyleGanOutputs_2/train_result/images/"
model_save_folder = "E:/TezOutputs/StyleGanModels_2/"
model_name = "style_gan_13_text"

def imsave(tensor, i):
    grid = tensor[0]
    grid.clamp_(-1, 1).add_(1).div_(2)
    # Add 0.5 after normalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    img = Image.fromarray(ndarr)
    img.save(f'{save_folder_path}{model_name}/sample-iter{i}.png')


def set_grad_flag(module, flag):
    for p in module.parameters():
        p.requires_grad = flag

def gain_sample(batch_size, image_size=4, shuffle=True):
    json_manager = JSONManager(Args.descriptions_json_path)

    data_transforms = transforms.Compose([
        transforms.Resize(image_size),  # Resize to the same size
        transforms.CenterCrop(image_size),  # Crop to get square area
        transforms.RandomHorizontalFlip(),  # Increase number of samples
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    celebA_dataset = data_manager.CelebADataset(Args.celeba_path, json_manager, transform=data_transforms)
    dataloader_celeba = torch.utils.data.DataLoader(celebA_dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader_celeba


def reset_LR(optimizer, lr):
    for pam_group in optimizer.param_groups:
        mul = pam_group.get('mul', 1)
        pam_group['lr'] = lr * mul


def fixed_descriptions():
    dataloader_celeba = iter(gain_sample(8, 8, False))
    _, description, _ = next(dataloader_celeba)
    f = open(f"E:/TezOutputs/StyleGanOutputs_2/train_result/images/{model_name}/fixed_examples/fixed_descriptions.txt", "a+", encoding="utf-8")
    for i, desc in enumerate(description):
        f.write(f"{i} : {desc} \n")
    f.close()


def fixed_examples(generator, embedder, iteration, step, alpha=0):
    resolution = 4 * 2 ** step
    dataloader_celeba = iter(gain_sample(8, resolution, False))
    _, description, _ = next(dataloader_celeba)

    # get embeddings
    latent_w1 = [torch.tensor(embedder.get_embeddings(description), device=device).float()]

    # Generate noise
    noise = torch.randn((latent_w1[0].shape[0], 100), device=device)
    noise = [torch.mul(noise, 0.07)]
    gan_input = [torch.cat((latent_w1[0], noise[0]), dim=-1)]

    # uluc: This is the noise appended to each generator block. Not as gan input at beginning.
    noise_1 = []

    for m in range(step + 1):
        size = 4 * 2 ** m  # Due to the upsampling, size of noise will grow
        noise_1.append(torch.randn((latent_w1[0].shape[0], 1, size, size), device=device))

    fake_image = generator(gan_input, step, 0, noise_1).detach()

    save_path = f"E:/TezOutputs/StyleGanOutputs_2/train_result/images/{model_name}/fixed_examples/{iteration}.jpg"
    upscale_factor = int(np.power(2, 6 - step - 1))
    create_grid(fake_image, upscale_factor, save_path)


def train():
    fixed_descriptions()
    # Initialize Embedder
    embedder = Embedder()

    # Used to continue training from last checkpoint
    iteration = 0
    startpoint = 0
    used_sample = 0
    alpha = 0

    step = 1  # train from 8x8
    d_losses = [float('inf')]
    g_losses = [float('inf')]

    generator = StyleBased_Generator(n_fc, dim_latent, dim_input).to(device)
    discriminator = Discriminator().to(device)

    # loss
    wgan_gp_loss = CondWGAN_GP(device, discriminator)

    # Optimizers
    g_optim = optim.Adam([{
        'params': generator.convs.parameters(),
        'lr': 0.001
    }, {
        'params': generator.to_rgbs.parameters(),
        'lr': 0.001
    }], lr=0.001, betas=(0.0, 0.99))

    g_optim.add_param_group({
        'params': generator.fcs.parameters(),
        'lr': 0.001 * 0.01,
        'mul': 0.01
    })
    d_optim = optim.Adam(discriminator.parameters(), lr=0.001, betas=(0.0, 0.99))

    if is_continue:
        if os.path.exists(f'{model_save_folder}{model_name}/trained.pth'):
            # Load data from last checkpoint
            print('Loading pre-trained model...')
            checkpoint = torch.load(f'{model_save_folder}{model_name}/trained.pth')
            generator.load_state_dict(checkpoint['generator'])
            discriminator.load_state_dict(checkpoint['discriminator'])
            g_optim.load_state_dict(checkpoint['g_optim'])
            d_optim.load_state_dict(checkpoint['d_optim'])
            step, iteration, startpoint, used_sample, alpha = checkpoint['parameters']
            d_losses = checkpoint.get('d_losses', [float('inf')])
            g_losses = checkpoint.get('g_losses', [float('inf')])
            print('Start training from loaded model...')
        else:
            print('No pre-trained model detected, restart training...')

    generator.train()
    discriminator.train()

    # train initial setup
    resolution = 4 * 2 ** step

    dataloader_celeba = gain_sample(batch_size, resolution)
    data_loader = iter(dataloader_celeba)

    reset_LR(g_optim, learning_rate.get(resolution, 0.001))
    reset_LR(d_optim, learning_rate.get(resolution, 0.001))

    progress_bar = tqdm(total=n_sample_total, initial=used_sample)

    # start train
    while used_sample < n_sample_total:
        iteration += 1
        alpha = min(1, alpha + batch_size / (n_sample))

        # increase resolution, reset data_loader
        if (used_sample - startpoint) > n_sample and step < max_step:
            step += 1
            alpha = 0
            startpoint = used_sample

            resolution = 4 * 2 ** step

            # Avoid possible memory leak
            del dataloader_celeba
            del data_loader

            # Change batch size # uluc - i removed changing batch size but will add back later
            dataloader_celeba = gain_sample(batch_size, resolution)
            data_loader = iter(dataloader_celeba)

            reset_LR(g_optim, learning_rate.get(resolution, 0.001))
            reset_LR(d_optim, learning_rate.get(resolution, 0.001))

        # Get Data from DataLoader
        try:
            # Try to read next image
            real_image, description, mismatched_description = next(data_loader)
        except (OSError, StopIteration):
            # Dataset exhausted, train from the first image
            data_loader = iter(dataloader_celeba)
            real_image, description, mismatched_description = next(data_loader)

        # Count used sample
        used_sample += real_image.shape[0]
        progress_bar.update(real_image.shape[0])

        # Send image to GPU
        real_image = real_image.to(device)

        # get embeddings
        latent_w1 = [torch.tensor(embedder.get_embeddings(description), device=device).float()]
        latent_w2 = [torch.tensor(embedder.get_embeddings(description), device=device).float()]

        # D Module ---
        # Train discriminator first
        discriminator.zero_grad()
        set_grad_flag(discriminator, True)
        set_grad_flag(generator, False)

        # Real image predict & backward
        # We only implement non-saturating loss with R1 regularization loss
        real_image.requires_grad = True

        # Generate noise
        noise = torch.randn((latent_w1[0].shape[0], 100), device=device)
        noise = [torch.mul(noise, 0.07)]
        gan_input = [torch.cat((latent_w1[0], noise[0]), dim=-1)]

        # This is the noise appended to each generator block. Not as gan input at beginning.
        noise_1 = []
        for m in range(step + 1):
            size = 4 * 2 ** m  # Due to the upsampling, size of noise will grow
            noise_1.append(torch.randn((latent_w1[0].shape[0], 1, size, size), device=device))

        # optimize the discriminator:
        d_optim.zero_grad()
        set_grad_flag(discriminator, True)
        set_grad_flag(generator, False)

        fake_image = generator(gan_input, step, alpha, noise_1).detach()

        mis_match_embed = [torch.tensor(embedder.get_embeddings(mismatched_description), device=device).float()]
        d_loss = wgan_gp_loss.dis_loss_uluc(real_image, fake_image, latent_w1, mis_match_embed, step, alpha)
        d_optim.step()

        if iteration % n_show_loss == 0:
            d_losses.append((d_loss).item())

        # optimize the generator
        generator.zero_grad()
        set_grad_flag(discriminator, False)
        set_grad_flag(generator, True)

        gan_input = [torch.cat((latent_w1[0], noise[0]), dim=-1)]
        fake_image = generator(gan_input, step, alpha, noise_1)
        g_loss = wgan_gp_loss.gen_loss_uluc(None, fake_image, latent_w1, step, alpha)
        g_optim.step()


        if iteration % n_show_loss == 0:
            g_losses.append(g_loss.item())
            # Save Generated Image
            imsave(fake_image.data.cpu(), iteration)
            # Save Description of Generated Image
            f = open(f"E:/TezOutputs/StyleGanOutputs_2/train_result/images/{model_name}/descriptions.txt", "a+", encoding="utf-8")
            f.write(f"{iteration} : {description[0]} \n")
            f.close()
            # Save Mismatched Description of Generated Image (Not needed)
            f = open(f"E:/TezOutputs/StyleGanOutputs_2/train_result/images/{model_name}/mismatched_descriptions.txt", "a+", encoding="utf-8")
            f.write(f"{iteration} : {mismatched_description[0]} \n")
            f.close()
            fixed_examples(generator, embedder, iteration, step)

        # Avoid possible memory leak
        del fake_image, latent_w2

        if iteration % 1000 == 0:
            # Save the model every 1000 iterations
            torch.save({
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'g_optim': g_optim.state_dict(),
                'd_optim': d_optim.state_dict(),
                'parameters': (step, iteration, startpoint, used_sample, alpha),
                'd_losses': d_losses,
                'g_losses': g_losses
            }, f'{model_save_folder}{model_name}/trained.pth')
            print(f'Model successfully saved.')

        progress_bar.set_description((f'Resolution: {resolution}*{resolution}  D_Loss: {d_losses[-1]:.4f}  G_Loss: {g_losses[-1]:.4f}  Alpha: {alpha:.4f}'))

    # After training completely finished (code should not reach here with 10 million iteration limit)
    torch.save({
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'g_optim': g_optim.state_dict(),
        'd_optim': d_optim.state_dict(),
        'parameters': (step, iteration, startpoint, used_sample, alpha),
        'd_losses': d_losses,
        'g_losses': g_losses
    }, f'{model_save_folder}{model_name}/trained_finished.pth')
    print(f'Final model successfully saved.')