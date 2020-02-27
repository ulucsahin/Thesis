import matplotlib.pyplot as plt
import torch
import numpy as np
from torchvision.utils import save_image
from torch.nn.functional import interpolate

def plot_img(array, number=None):
    img_size = 32
    array = array.detach()
    array = array.reshape(img_size, img_size)

    plt.imshow(array.cpu().detach().numpy(), cmap='binary')
    plt.xticks([])
    plt.yticks([])
    if number:
        plt.xlabel(number, fontsize='x-large')
    plt.show()

def make_some_noise(batch_size, output_size):
    return torch.rand(batch_size, output_size)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def create_grid(samples, scale_factor, img_file, real_imgs=False):
    """
    utility function to create a grid of GAN samples
    :param samples: generated samples for storing
    :param scale_factor: factor for upscaling the image
    :param img_file: name of file to write
    :param real_imgs: turn off the scaling of images
    :return: None (saves a file)
    """

    samples = torch.clamp((samples / 2) + 0.5, min=0, max=1)

    # upsample the image
    if not real_imgs and scale_factor > 1:
        samples = interpolate(samples, scale_factor=scale_factor)

    # save the images:
    save_image(samples, img_file, nrow=int(np.sqrt(len(samples))))

def sentence_to_indexes(sentence, word_to_idx_mapping):
    result = []
    for word in sentence:
        result.append(word_to_idx_mapping[word])

    return result

def indexes_to_sentence(indexes, idx_to_word_mapping):
    result = []
    for index in indexes:
        result.append(idx_to_word_mapping[index])

    return

def annotations_to_text(annots, img_index):
    """
    This method converts celebA annotations to readable form.
    """
    labels = ["5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
              "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup",
              "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose",
              "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick",
              "Wearing_Necklace", "Wearing_Necktie", "Young" ]

    result = ""
    ones = ""
    minus_ones = ""

    for i, annot in enumerate(annots):
        if annot == 1:
            ones += " " + labels[i]
        if annot == -1:
            minus_ones += " " + labels[i]

    result = str(img_index) + ":\n" + "========= ONES ========= \n" + ones + "\n======== MINUS ONES ======== \n" + minus_ones

    return result
