import json
from random import randrange
import re
import random
from args import Args


class JSONManager(object):
    """
    This class handles everything about json description file
    """
    word_frequencies = {}
    word_to_idx = {}
    idx_to_word = {}
    vocab_size = -1


    def __init__(self, json_path):
        self.json_path = json_path
        self.data = self.read_json(json_path)
        self.preprocess_text_data()
        self.frequencies = self.get_word_frequencies()
        self.word_to_idx = self.get_word_to_idx_mapping()
        self.idx_to_word = self.get_idx_to_word_mapping()
        self.vocab_size = self.get_vocab_size()


    def read_json(self, path):
        with open(path) as json_file:
            data = json.load(json_file)

        return data

    def get_number_of_descriptions_at_idx(self, index):
        return len(self.data[index]["descriptions"])

    def get_description_from_idx(self, index):
        """
        This method returns a random description for given image index.
        """
        idx = random.randint(0, len(self.data[index]["descriptions"])-1)

        return self.data[index]["descriptions"][idx]["text"]


    def get_description_from_idx_with_idx(self, index, index2):
        """
        This method return description number "index2" for data instance number "index".
        """
        return self.data[index]["descriptions"][index2]["text"]


    def get_imagename_from_idx(self, idx):
        return self.data[idx]["image"]


    def get_random_description(self):
        """
        Returns description of random image
        Used for giving images with mismatched description to discriminator
        """
        rand_idx = randrange(len(self.data))

        return self.get_description_from_idx(rand_idx)


    def preprocess_text_data(self):
        """
        Insert space before special characters such as "," or "."  etc.
        """
        #desc = "A woman with a chiselled jaw, prominent cheekbones, a long, narrow nose and thin eyebrows. She has long, messy, black hair and she is wearing makeup."

        for i in range(len(self.data)):
            for j in range(len(self.data[i]["descriptions"])):
                desc = self.get_description_from_idx_with_idx(i,j)
                desc = re.sub(r"([^a-zA-Z])", r" \1 ", desc)
                desc = re.sub('\s{2,}', ' ', desc)
                self.data[i]["descriptions"][j]["text"] = desc


    def get_word_frequencies(self):
        frequencies = {}
        for i in range(len(self.data)):
            for j in range(len(self.data[i]["descriptions"])):
                desc = self.get_description_from_idx_with_idx(i, j)
                words = desc.split(" ")
                for word in words:
                    try:
                        frequencies[word] += 1
                    except:
                        frequencies[word] = 1

        # sort according to frequency
        frequencies = {k: v for k, v in sorted(frequencies.items(), key=lambda item: item[1], reverse=True)}

        return frequencies


    def get_word_to_idx_mapping(self):
        mapping = {}
        count = 2 # 0 and 1 are reserved for pad and unknown
        for k, v in self.frequencies.items():
            word = k
            mapping[word] = count
            count += 1

        return mapping


    def get_idx_to_word_mapping(self):
        idx_to_word = {}
        for k, v in self.word_to_idx.items():
            idx_to_word[v] = k

        return idx_to_word


    def get_vocab_size(self):
        return len(self.frequencies)


    def transform_sentence(self, tokenized_sentence, desired_length):
        # Transforms sentence to word indexes
        unknown = 1
        pad = 0
        transformed = []
        for word in tokenized_sentence:
            if word in self.word_to_idx:
                transformed.append(self.word_to_idx[word])
            else:
                transformed.append(unknown)

        if len(transformed) < desired_length:
            while(len(transformed) < desired_length):
                transformed.append(pad)
        else:
            transformed = transformed[0:desired_length]

        # print("len(transformed)", len(transformed))
        return transformed


    def get_shortest_sentence_length(self):
        result = 9e9

        for item in self.data:
            for desc in item["descriptions"]:
                sentence_len = len(desc["text"].split(" "))
                if sentence_len < result:
                    result = sentence_len

        print(result)
        return result


    def get_average_sentence_length(self):
        result = 0
        count = 0
        for item in self.data:
            for desc in item["descriptions"]:
                sentence_len = len(desc["text"].split(" "))
                result += sentence_len
                count += 1

        result = result / count
        print(result)

        return result


class SimpleShapesDescriptionReader(object):
    data_dir = Args.shapes_dataset_dir
    descriptions_dir = Args.shapes_descriptions_dir
    data = None

    def __init__(self):
        self.data = self.get_descriptions_data()


    def get_descriptions_data(self):
        result = {}
        filepath = self.descriptions_dir
        with open(filepath) as fp:
            line = fp.readline()
            item = line.split(":")
            image_name = item[0][:-1] + ".jpg"
            description = item[1]
            description = description[1:-2]  # remove unnecessary space at the beginning of each description and \n at the end.
            result[image_name] = description
            cnt = 1
            while line:
                line = fp.readline()
                if(not line):
                    break

                item = line.split(":")
                image_name = item[0][:-1] + ".jpg"
                description = item[1]
                description = description[1:-2] # remove unnecessary space at the beginning of each description and \n at the end.
                cnt += 1

                result[image_name] = description

        return result


    def get_description_from_image_name(self, image_name):
        return self.data[image_name]


    def get_imagename_from_idx(self, idx):
        return str(idx) + ".jpg"


    def get_description_from_idx(self, idx):
        img_name = self.get_imagename_from_idx(idx)
        return self.data[img_name]

    def get_random_description(self):
        rand_idx = randrange(len(self.data))

        return self.get_description_from_idx(rand_idx)


class JSONManager_V2(object):
    """
    This class is used to parse json data generated by combining 5685 data and
    descriptions generated from celebA annotations.
    """
    json_path = None
    data = None
    word_frequencies = {}
    word_to_idx = {}
    idx_to_word = {}
    vocab_size = -1

    def __init__(self, json_path):
        self.json_path = json_path
        self.data = self.read_json(self.json_path)
        self.preprocess_text_data()
        self.frequencies = self.get_word_frequencies()
        self.word_to_idx = self.get_word_to_idx_mapping()
        self.idx_to_word = self.get_idx_to_word_mapping()
        self.vocab_size = self.get_vocab_size()

    def read_json(self, path):
        with open(path) as json_file:
            temp_data = json.load(json_file)

        data = []
        for k, v in temp_data.items():
            data.append({"imagename": k, "descriptions": v})

        return data


    def get_number_of_descriptions_at_idx(self, index):
        return len(self.data[index]["descriptions"])


    def get_description_from_idx(self, index):
        """
        This method returns a random description for given image index.
        """
        idx = random.randint(0, len(self.data[index]["descriptions"])-1)

        return self.data[index]["descriptions"][idx]

    def get_description_from_idx_with_idx(self, index, index2):
        """
        This method return description number "index2" for data instance number "index".
        """
        return self.data[index]["descriptions"][index2]

    def get_imagename_from_idx(self, idx):
        return self.data[idx]["imagename"]


    def get_random_description(self):
        """
        Returns description of random image
        Used for giving images with mismatched description to discriminator
        """
        rand_idx = randrange(len(self.data))

        return self.get_description_from_idx(rand_idx)


    def preprocess_text_data(self):
        """
        Insert space before special characters such as "," or "."  etc.
        """
        #desc = "A woman with a chiselled jaw, prominent cheekbones, a long, narrow nose and thin eyebrows. She has long, messy, black hair and she is wearing makeup."

        for i in range(len(self.data)):
            for j in range(len(self.data[i]["descriptions"])):
                desc = self.get_description_from_idx_with_idx(i,j)
                desc = re.sub(r"([^a-zA-Z])", r" \1 ", desc)
                desc = re.sub('\s{2,}', ' ', desc)
                self.data[i]["descriptions"][j] = desc

    def get_word_frequencies(self):
        frequencies = {}
        for i in range(len(self.data)):
            for j in range(len(self.data[i]["descriptions"])):
                desc = self.get_description_from_idx_with_idx(i, j)
                words = desc.split(" ")
                for word in words:
                    try:
                        frequencies[word] += 1
                    except:
                        frequencies[word] = 1

        # sort according to frequency
        frequencies = {k: v for k, v in sorted(frequencies.items(), key=lambda item: item[1], reverse=True)}

        return frequencies

    def get_word_to_idx_mapping(self):
        mapping = {}
        count = 2 # 0 and 1 are reserved for pad and unknown
        for k, v in self.frequencies.items():
            word = k
            mapping[word] = count
            count += 1

        return mapping


    def get_idx_to_word_mapping(self):
        idx_to_word = {}
        for k, v in self.word_to_idx.items():
            idx_to_word[v] = k

        return idx_to_word


    def get_vocab_size(self):
        return len(self.frequencies)





