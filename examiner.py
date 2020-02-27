"""
This file has methods that help us examine the GAN.
Call execute method with hardcoded sentences to get outputs for sentences.
This file is not well written.
"""

import torch
import numpy as np
from torchvision.utils import save_image
from style_gan_13_text.model import StyleBased_Generator, Discriminator
import os
from torch.nn.functional import interpolate

# my code
from embedding_manager import Embedder, Encoder
from ConditionAugmenter import ConditionAugmentor
from json_manager import JSONManager
from args import Args
from util import sentence_to_indexes

# parameters
device = "cuda"
save_folder_path = "E:/TezOutputs/StyleGanOutputs_2/train_result/images/"
model_save_folder = "E:/TezOutputs/StyleGanModels_2/"
model_name = "style_gan_13_text"
sentence_file_path = f"E:/TezOutputs/StyleGanOutputs_2/train_result/images/{model_name}/examination/sentences_tried.txt"
n_fc = 8
dim_latent = 400 #400 #256 + 100 # 600 # 300  # 512

dim_input = 4
noise_dim = 100 # dim_latent - 300

encoder_available = False
condition_augmenter_available = False

# Encoder parameters
hidden_size = 300
num_layers = 3

# Conditioning Augmentation hyperparameters
compressed_latent_size = 128
ca_out_size = 256


def save_example(fake_image, save_path):
    #save_image(fake_image, save_path, nrow=int(np.sqrt(len(fake_image))))
    save_image(fake_image, save_path)


def generate_input_noise(input_noise_save_path, latent_w1):
    noise = torch.randn((latent_w1[0].shape[0], noise_dim), device=device)
    #noise = [torch.mul(noise, 0.07)]
    noise = [torch.mul(noise, 0.01)]
    torch.save(noise, input_noise_save_path)

    return noise


def generate_middle_noise(middle_noise_save_path, latent_w1, step):
    # uluc: This is the noise appended to each generator block. Not as gan input at beginning.
    noise_1 = []
    for m in range(step + 1):
        size = 4 * 2 ** m  # Due to the upsampling, size of noise will grow
        noise_1.append(torch.randn((latent_w1[0].shape[0], 1, size, size), device=device))
    torch.save(noise_1, middle_noise_save_path)

    return noise_1


def load_input_noise(input_noise_file_path):
    return torch.load(input_noise_file_path)


def load_middle_noise(middle_noise_file_path):
    return torch.load(middle_noise_file_path)


def save_sentence(sentence, img_id):
    img_id = int(get_current_id())

    f = open(sentence_file_path, "a+", encoding="utf-8")
    f.write(f"{img_id} : {sentence}\n")
    f.close()


def get_current_id():
    if os.path.exists(f'{sentence_file_path}'):
        with open(sentence_file_path, 'r') as f:
            lines = f.read().splitlines()
            last_line = lines[-1]
            id = last_line.split(" ")[0]
    else:
        id = -1

    return int(id) + 1


def examine(sentence, new_input_noise, new_middle_noise, embedder, generator, step, alpha, model_name, img_id=0, condition_augmenter=None, raw_sentence=None):
    # save sentence
    if(encoder_available):
        save_sentence(raw_sentence, img_id)
    else:
        save_sentence(sentence, img_id)

    # get embeddings
    embedding_path = f"E:/TezOutputs/StyleGanOutputs_2/train_result/images/{model_name}/examination/embedding.pt"
    if encoder_available:
        assert condition_augmenter is not None
        embedding = embedder(sentence).to(device)
        c_not_hats, mus, sigmas = condition_augmenter(embedding)
        latent_w1 = [c_not_hats]
    else:
        embedding = embedder.get_embeddings(sentence)
        latent_w1 = [torch.tensor(embedding, device=device).float()]

    torch.save(embedding, embedding_path)

    # Generate or load noise
    input_noise_path = f"E:/TezOutputs/StyleGanOutputs_2/train_result/images/{model_name}/examination/input_noise.pt"
    middle_noise_path = f"E:/TezOutputs/StyleGanOutputs_2/train_result/images/{model_name}/examination/middle_noise.pt"
    if new_input_noise:
        print("Generating new input noise.")
        noise = generate_input_noise(input_noise_path, latent_w1)

    else:
        print("Using previously generated input noise.")
        noise = load_input_noise(input_noise_path)

    #print(torch.mul(noise[0], 0.07))
    if new_middle_noise:
        print("Generating new middle noise.")
        noise_1 = generate_middle_noise(middle_noise_path, latent_w1, step)
    else:
        print("Using previously generated middle noise.")
        noise_1 = load_middle_noise(middle_noise_path)

    gan_input = [torch.cat((latent_w1[0], noise[0]), dim=-1)]

    # save image
    fake_image = generator(gan_input, step, alpha, noise_1).detach()
    fake_image = torch.clamp((fake_image / 2) + 0.5, min=0, max=1)
    scale_factor = int(np.power(2, 6 - step - 1)) # correct colors
    image_path = f"E:/TezOutputs/StyleGanOutputs_2/train_result/images/{model_name}/examination/image{img_id}.jpg"
    save_example(fake_image, image_path)


def execute():
    word_to_idx = None
    temp_json = None

    import data_manager
    from torchvision import transforms

    if encoder_available:
        temp_json = JSONManager(Args.descriptions_json_path)
        vocab_size = temp_json.get_vocab_size() + 2  # +2 for unknown and pad
        embedder = Encoder(300, vocab_size, hidden_size, num_layers).to(device)
        word_to_idx = temp_json.get_word_to_idx_mapping()
    else:
        embedder = Embedder()

    condition_augmenter = None
    if condition_augmenter_available:
        condition_augmenter = ConditionAugmentor(input_size=300, latent_size=ca_out_size, use_eql=True, device=device)

    generator = StyleBased_Generator(n_fc, dim_latent, dim_input).to(device)

    # load model that we want to test
    if os.path.exists(f'{model_save_folder}{model_name}/trained.pth'):
        print('Loading pre-trained model.')
        checkpoint = torch.load(f'{model_save_folder}{model_name}/trained.pth')
        generator.load_state_dict(checkpoint['generator'])
        if encoder_available:
            embedder.load_state_dict(checkpoint['encoder'])
        if condition_augmenter_available:
            condition_augmenter.load_state_dict(checkpoint["condition_augmenter"])
        step, iteration, startpoint, used_sample, alpha = checkpoint['parameters']
        print('Model loaded succesfully.')


    new_input_noise = True
    new_middle_noise = True

    sentence1 = ["A young girl with straight long blonde hair . Her eyes are brown and small and her lips are thin . She has got a heavy lower lip . She looks serious ."]
    sentence2 = ["A young girl with straight long dark hair . Her eyes are brown and small and her lips are thin . She has got a heavy lower lip . She is smiling and her upper teeth is visible ."]
    sentence3 = ["A young girl with straight long dark hair . Her eyes are brown and small and her lips are thin . She has got a heavy lower lip . She looks serious ."]
    sentence4 = ["A man with a chiselled jaw . He has narrow eyes and short , curly , black hair . He is smiling ."]
    sentence5 = ["A dark skinned man with short , black , curly hair , thick eyebrows , a wide nose and a stubble beard . He is smiling and his upper teeth is visible ."]
    sentence6 = ["A pale skinned man with short , black , curly hair , thick eyebrows , a wide nose and a stubble beard . He is smiling and his upper teeth is visible ."]

    #sentences = [sentence1, sentence2, sentence3, sentence4, sentence5, sentence6]
    sentences = [sentence1]

    raw_sentence = None
    if encoder_available:
        raw_sentence = sentence.copy()
        #sentence = [torch.tensor(sentence_to_indexes(sentence[0].split(" "), word_to_idx)).to(device)]
        sentence = sentence[0].split(" ")
        sentence = temp_json.transform_sentence(sentence, 100)
        sentence = torch.tensor([sentence], dtype=torch.long).to(device)

    for i in range(len(sentences)):
        print(sentences[i])
        examine(sentences[i], new_input_noise, new_middle_noise, embedder, generator, step, alpha, model_name, img_id=int(get_current_id()), condition_augmenter=None, raw_sentence=raw_sentence)