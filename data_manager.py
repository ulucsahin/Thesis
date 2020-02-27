from torchvision import transforms
import torchvision.datasets as datasets
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
from skimage import io
import PIL
import numpy as np
from json_manager import JSONManager
import random


class CelebADataset(Dataset):
    def __init__(self, data_path, JSON_manager, transform=None, tokenize=False, encode=False, return_sentence_lengths=False, damsm_mode=False, desired_length=100):
        """
        Args:
            data_path (string): Directory with all the images.
            JSON_manager: JSONManager class holding all annotation information
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        self.data_path = data_path
        self.JSON_manager = JSON_manager
        self.transform = transform
        self.tokenize = tokenize
        self.encode = encode
        self.damsm_mode = damsm_mode
        self.desired_length = desired_length

        self.return_sentence_lengths = return_sentence_lengths
        #files = next(os.walk(data_path))[2]  # dir is your directory path as string

    def __len__(self):
        return len(self.JSON_manager.data)

    def __getitem__(self, index):
        """
        This method takes index and returns corresponding image and descriptions.
        tokenize: either tokenize sentence to words or not
        encode: encode words to numeric indexes or not
        return_sentence_lengths: returns sentence lengths along with sentences
        damsm_mode: should be set to True if this dataset is being used for DAMSM model (for attention GAN, currently not used)
        """
        # index: index in json file, filename is different than index. for example at index 0 we may have image 000035.jpg

        image_name = self.JSON_manager.get_imagename_from_idx(index)
        description = self.JSON_manager.get_description_from_idx(index)
        mismatched_description = self.JSON_manager.get_random_description()

        img_path = os.path.join(self.data_path, image_name)
        image = PIL.Image.open(img_path)

        raw_description = ""
        raw_mismatched_description = ""

        if self.transform:
            image = self.transform(image)

        if self.tokenize:
            raw_description = description
            raw_mismatched_description = mismatched_description
            description = description.split(" ")
            mismatched_description = mismatched_description.split(" ")

        if self.encode and self.tokenize: # cant encode without tokenizing
            description = self.JSON_manager.transform_sentence(description, self.desired_length)
            mismatched_description = self.JSON_manager.transform_sentence(mismatched_description, self.desired_length)

            description = torch.tensor(description, dtype=torch.long)
            mismatched_description = torch.tensor(mismatched_description, dtype=torch.long)

            if self.damsm_mode:
                return image, description, len(description), img_path

            if self.return_sentence_lengths:
                return image, description, mismatched_description, len(description), len(mismatched_description), raw_description, raw_mismatched_description

            return image, description, mismatched_description


        return image, description, mismatched_description


    def get_index(self, index):
        # Wrapper
        return self.__getitem__(index)

    def get_vocab_size(self):
        return self.JSON_manager.vocab_size

    def get_word_to_idx(self):
        return self.JSON_manager.word_to_idx

    def get_idx_to_word(self):
        return self.JSON_manager.idx_to_word


class CelebADataset_AnnotationVersion(Dataset):
    """
    This class is for celeba dataset with 40 annotations. Has all 200k data from celebA.
    """
    def __init__(self, data_path, annotations_path, transform=None):
        """
        Args:
            data_path (string): Directory with all the images.
            JSON_manager: JSONManager class holding all annotation information
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        self.data_path = data_path
        self.annotations_path = annotations_path
        self.transform = transform
        self.annotations = self.get_annotations()

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        """
        This method takes index and returns corresponding image and descriptions.
        tokenize: either tokenize sentence to words or not
        encode: encode words to numeric indexes or not
        """
        # index: index in json file, filename is different than index. for example at index 0 we may have image 000035.jpg

        # in celebA dataset are images are named like "000001.jpg" etc, image number is "000001" part here
        image_name = self.get_image_name_from_index(index)
        img_path = os.path.join(self.data_path, image_name)
        image = PIL.Image.open(img_path)

        true_annotations = self.annotations[image_name]

        rand_idx = random.randint(0, len(self.annotations) - 1)

        mismatched_image_name = self.get_image_name_from_index(rand_idx)
        mismatched_annotations = self.annotations[mismatched_image_name]

        if self.transform:
            image = self.transform(image)

        return image, true_annotations, mismatched_annotations

    def get_image_name_from_index(self, index):
        image_number = index + 1  # images in celebA start from 000001.jpg
        image_number = str(image_number).zfill(6)  # pad zeros to beginning so that length of str is 6, for example, 35 becomes 000035
        image_name = image_number + ".jpg"

        return image_name

    def get_annotations(self):
        filepath = self.annotations_path
        annotations = {}
        with open(filepath) as fp:
            # first two lines are number of images and annotation labels
            line = fp.readline()
            no_images = line
            line = fp.readline()
            annotation_labels = line

            while line:
                line = fp.readline()
                if line == "":
                    break

                line_ = line.split(" ")

                current_image_name = line_[0]

                current_annotations = line_[1:]  # first element is image name
                current_annotations = [int(a) for a in current_annotations]
                annotations[current_image_name] = np.array(current_annotations)

        return annotations


    def get_index(self, index):
        # Wrapper
        return self.__getitem__(index)


class SimpleShapesDataset(Dataset):
    def __init__(self, data_path, JSON_manager, transform=None, tokenize=False, encode=False, return_sentence_lengths=False, damsm_mode=False, desired_length=100):
        """
        Args:
            data_path (string): Directory with all the images.
            JSON_manager: JSONManager class holding all annotation information
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        self.data_path = data_path
        self.JSON_manager = JSON_manager
        self.transform = transform
        self.tokenize = tokenize
        self.encode = encode
        self.damsm_mode = damsm_mode
        self.desired_length = desired_length

        self.return_sentence_lengths = return_sentence_lengths
        #files = next(os.walk(data_path))[2]  # dir is your directory path as string

    def __len__(self):
        return len(self.JSON_manager.data)

    def __getitem__(self, index):
        """
        This method takes index and returns corresponding image and descriptions.
        TODO: Multiple descriptions.
        tokenize: either tokenize sentence to words or not
        encode: encode words to numeric indexes or not
        """
        # index: index in json file, filename is different than index. for example at index 0 we may have image 000035.jpg

        image_name = self.JSON_manager.get_imagename_from_idx(index)
        description = self.JSON_manager.get_description_from_idx(index)
        mismatched_description = self.JSON_manager.get_random_description()

        # in celebA dataset are images are named like "000001.jpg" etc, image number is "000001" part here
        #image_number = str(image_number).zfill(6) # pad zeros to beginning so that length of str is 6, for example, 35 becomes 000035
        img_path = os.path.join(self.data_path, image_name)
        image = PIL.Image.open(img_path)

        raw_description = ""
        raw_mismatched_description = ""

        if self.transform:
            image = self.transform(image)

        if self.tokenize:
            raw_description = description
            raw_mismatched_description = mismatched_description
            description = description.split(" ")
            mismatched_description = mismatched_description.split(" ")

        if self.encode and self.tokenize: # cant encode without tokenizing
            description = self.JSON_manager.transform_sentence(description, self.desired_length)
            mismatched_description = self.JSON_manager.transform_sentence(mismatched_description, self.desired_length)

            description = torch.tensor(description, dtype=torch.long)
            mismatched_description = torch.tensor(mismatched_description, dtype=torch.long)

            if self.damsm_mode:
                return image, description, len(description), img_path

            if self.return_sentence_lengths:
                return image, description, mismatched_description, len(description), len(mismatched_description), raw_description, raw_mismatched_description

            return image, description, mismatched_description


        return image, description, mismatched_description


    def get_index(self, index):
        # Wrapper
        return self.__getitem__(index)

    def get_vocab_size(self):
        return self.JSON_manager.vocab_size

    def get_word_to_idx(self):
        return self.JSON_manager.word_to_idx

    def get_idx_to_word(self):
        return self.JSON_manager.idx_to_word