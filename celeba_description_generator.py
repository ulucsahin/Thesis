from args import Args
from json_manager import JSONManager
import json
from tqdm import tqdm
import os
import re
from random import randint
import itertools
# Parameters
skip_under = 10

from annotation_manager import *

def cleanup_face2text():
    json_manager = JSONManager(Args.descriptions_json_path)

    data = {}
    for i in range(len(json_manager.data)):
        image_name = json_manager.get_imagename_from_idx(i)
        no_descriptions = json_manager.get_number_of_descriptions_at_idx(i)
        descriptions = []
        for j in range(no_descriptions):
            description = json_manager.get_description_from_idx_with_idx(i, j)
            try:
                data[image_name].append(description)
            except:
                data[image_name] = [description]

    with open(Args.cleanedup_lfw_descriptions, 'w') as outfile:
        json.dump(data, outfile)


def generate_new_dataset_json():
    """
    Generate json file of new descriptions dataset combined with old descriptions (from 5685 data)
    """
    filepath = Args.uluc_generated_descriptions
    dataset_5685 = Args.descriptions_json_path
    savepath = Args.uluc_combined_dataset_json

    # old json manager for extracting old descriptions
    json_manager = JSONManager(dataset_5685)

    data = {}
    for i in range(len(json_manager.data)):
        image_name = json_manager.get_imagename_from_idx(i)
        no_descriptions = json_manager.get_number_of_descriptions_at_idx(i)
        descriptions = []
        for j in range(no_descriptions):
            description = json_manager.get_description_from_idx_with_idx(i, j)
            try:
                data[image_name].append(description)
            except:
                data[image_name] = [description]

    with open(filepath) as fp:
        line = fp.readline()

        while line:
            item = line.split(":")
            image_name = item[0]

            # we should not have any image that exists both in old descriptions and new descriptions
            assert image_name not in data.keys()

            description = item[1]
            description = description[1:]  # remove unnecessary space at the beginning of each description
            data[image_name] = [description] # each image has only 1 description in new descriptions so we dont need to use .append()

            line = fp.readline()

    with open(savepath, 'w') as outfile:
        json.dump(data, outfile)




def get_annotation_dict():
    result = {}
    with open(Args.annotation_dict_path) as fp:
        line = fp.readline()
        while line:
            line = line.split(":")
            line[1] = line[1][0:-1] # get rid of "\n" at the end of string
            result[line[0]] = line[1]
            line = fp.readline()

    return result


def get_annotation_sentences():
    result = {}
    with open(Args.annotation_sentence_parts) as fp:
        line = fp.readline()
        while line:
            line = line.split(":")
            line[1] = line[1][0:-1]  # get rid of "\n" at the end of string
            result[line[0]] = line[1]
            line = fp.readline()
    print(result)
    return result

class CelebaDescriptionGenerator(object):
    """
    Merge descriptions generated from annotations with original data.
    If a description is present in original data, skip that image.
    Only generate descriptions for images without a description.
    Merge new data with old data to have a bigger dataset.
    """
    lfw_data = None
    annotation_dict = None
    annotation_sentence_parts = None


    def __init__(self, cleanup_data = False):
        self.annotation_dict = get_annotation_dict()
        self.annotation_sentence_parts = get_annotation_sentences()

        if cleanup_data:
            cleanup_face2text()

        with open(Args.cleanedup_lfw_descriptions) as json_file:
            self.lfw_data = json.load(json_file)

    def rule_check(self, annotations):
        """
        return False if there are annotations marked as 1 which doesnt make any sense
        (such as blond hair and bald both marked as 1)
        """

        # removed this part since partly bald people marked as bald too.
        # # check hair (if bald and has hair)
        # if annotations[bald] == "1" and (annotations[black_hair] == "1" or annotations[blond_hair] == "1" or annotations[brown_hair] == "1"
        #                           or annotations[gray_hair] == "1" or annotations[straight_hair] == "1" or annotations[wavy_hair] == "1" or annotations[bangs] == "1"):
        #
        #     return False

        # check if woman and has beard (reaaally low chance)
        if (not annotations[male] == "1") and (annotations[goatee] == "1" or annotations[mustache] == "1"):
            return False

        # if has beard and has no beard
        if annotations[24] == "1" and (annotations[goatee] == "1" or annotations[mustache] == "1"):
            return False

        # if has no info about hair
        if (not annotations[black_hair] == "1") and (not annotations[blond_hair] == "1") and (not annotations[brown_hair] == "1") and (not annotations[gray_hair] == "1"):
            return False

        return True

    def count_ones(self, annotations):
        count = 0
        for item in annotations:
            if item == "1":
                count += 1

        return count


    def generate_description(self, annotations):
        # info from annotations
        intro = get_introduction_info(annotations)
        he_she = get_gender_auxiliary(annotations)
        his_her = get_gender_auxiliary2(annotations)
        hair = get_hair(annotations)
        face = get_face(annotations)
        nose = get_nose(annotations)
        beard = get_beard(annotations)
        eye_area = get_eye_information(annotations, he_she, his_her)
        makeup = get_makeup(annotations)
        mouth_area = get_mouth_area(annotations, he_she)

        # gender and hair description
        combination1 = f"{intro} {hair} ."

        if bangs != "":
            combination1 = f"{intro} {hair} ."

        # face information
        combination2 = ""
        if face != "":
            combination2 = f" {he_she} has {face} ."

        # nose
        combination3 = ""
        if nose != "":
            combination3 = f" {he_she} has {nose} ."

        # beard stuff
        combination4 = ""
        if beard != "" and he_she != "she": # we dont need to specify that a girl has no beard
            combination4 = f" {he_she} has {beard} ."

        # eye area
        combination5 = ""
        if eye_area != "":
            combination5 = f"{eye_area}"

        combination6 = f"{makeup}"
        combination7 = f"{mouth_area}"


        # add descriptive sentences in a random order (except introduction sentence)
        middle_choices = [combination2, combination3, combination4, combination5, combination6, combination7]
        middle_choice_combinations = list(itertools.permutations(middle_choices))

        choice = randint(0, len(middle_choice_combinations)-1)
        description = combination1 \
                      + middle_choice_combinations[choice][0] \
                      + middle_choice_combinations[choice][1] \
                      + middle_choice_combinations[choice][2] \
                      + middle_choice_combinations[choice][3] \
                      + middle_choice_combinations[choice][4] \
                      + middle_choice_combinations[choice][5]

        description = re.sub('\s+', ' ', description)
        description = re.sub('( . . )+', ' . ', description)
        description = re.sub('( . . )+', ' . ', description)

        return description


    def execute(self, filepath):
        no_male = 0.001
        no_female = 0.001
        desired_ratio = 1.0

        with open(filepath) as fp:
            line = fp.readline() # we don't need first line
            annotation_names = fp.readline().split(" ") # second line has annotation names
            annotation_names = annotation_names[0:-1] # remove "\n" from array

            line = fp.readline()
            current = 1
            progress_bar = tqdm(total=198523, initial=current)
            progress_bar.set_description("Generating descriptions from annotations")

            # file operations
            if os.path.exists(Args.uluc_generated_descriptions):
                os.remove(Args.uluc_generated_descriptions)
            f = open(Args.uluc_generated_descriptions, "a+", encoding="utf-8")

            # some debugging variables
            skipped_already_has_desc = 0
            skipped_no_annons = 0
            skipped_rule_check = 0
            skipped_ratio_check = 0
            generated = 0
            skip = False
            while line:
                item = line.split(" ")
                image_name = item[0]

                # only generate descriptions for images without any description
                if image_name in self.lfw_data:
                    # TODO: Add existing description for these files
                    skipped_already_has_desc += 1
                    skip = True

                annotations = item[1:]
                no_ones = self.count_ones(annotations)

                if no_ones < skip_under:
                    skipped_no_annons += 1
                    skip = True

                if not self.rule_check(annotations):
                    skipped_rule_check += 1
                    skip = True


                m_to_f_ratio = no_male / no_female
                if m_to_f_ratio < desired_ratio and annotations[20] != "1":
                    skipped_ratio_check += 1
                    skip = True


                if skip:
                    skip = False
                    line = fp.readline()
                    current += 1
                    progress_bar.update(1)
                    continue

                if annotations[20] == "1":
                    no_male += 1
                else:
                    no_female += 1

                description = self.generate_description(annotations)
                generated += 1
                f.write(f"{image_name}: {description}\n")

                line = fp.readline()
                current += 1
                progress_bar.update(1)


            f.close()
            print(f"Skipped {skipped_already_has_desc} descriptions since they already had descriptions.")
            print(f"Skipped {skipped_no_annons} descriptions since they had less than {skip_under} annotations marked as '1'.")
            print(f"Skipped {skipped_rule_check} descriptions since their annotations made no sense (broke rules).")
            print(f"Skipped {skipped_ratio_check} descriptions to preserve male to female ratio.")
            print(f"Generated a dataset of descriptions with male to female ratio of {m_to_f_ratio} .")
            print(f"Generated {generated} descriptions.")
