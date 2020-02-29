from args import Args
from json_manager import JSONManager
import json
from tqdm import tqdm
import os

# Parameters
skip_under = 11

# Annotation indexes
bald = 4
bangs = 5
has_bangs_hair = 5
black_hair = 8
blond_hair = 9
brown_hair = 11
gray_hair = 17
straight_hair = 32
wavy_hair = 33

# gender and beard
male = 20
goatee = 16
mustache = 22
no_beard = 24

# face
chubby = 13
oval_face = 25
high_cheekbones = 19
rosy_cheeks = 29

# nose
big_nose = 7
pointy_nose = 27

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
        #                           or annotations[gray_hair] == "1" or annotations[straight_hair] == "1" or annotations[wavy_hair] == "1" or annotations[has_bangs_hair] == "1"):
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


    def get_gender(self, annotations):
        if annotations[male] == "1":
            return "male"
        else:
            return "female"


    def get_gender_auxiliary(self, annotations):
        if annotations[male] == "1":
            return "he"
        else:
            return "she"


    def get_gender_auxiliary2(self, annotations):
        if annotations[male] == "1":
            return "his"
        else:
            return "her"


    def get_age(self, annotations):
        if annotations[39] == "1\n":
            return "young"
        else:
            return "old"


    def get_bangs(self, annotations):
        if annotations[bangs] == "1":
            return "bangs"
        else:
            return ""


    def get_hair_color(self, annotations):
        if annotations[bald] == "1":
            return "bald"
        elif annotations[black_hair] == "1":
            return "black"
        elif annotations[blond_hair] == "1":
            return "blond"
        elif annotations[brown_hair] == "1":
            return "brown"
        elif annotations[gray_hair] == "1":
            return "gray"
        else:
            return ""


    def get_hair_type(self, annotations):
        if annotations[straight_hair] == "1":
            return "straight"
        elif annotations[wavy_hair] == "1":
            return "wavy"
        else:
            return ""


    def get_face(self, annotations):
        face = ""
        face_middle = "a face with "
        face_end = ""

        if annotations[chubby] == "1":
            face = "chubby"
            face_middle = "face with"
            face_end = "face"

        if annotations[oval_face] == "1":
            face += " oval"
            face_middle = "face with"
            face_end = "face"

        # if annotations[high_cheekbones] == "1":
        #     face += f" {face_middle} high cheekbones"
        #     face_middle = "and"
        #     face_end = ""

        if annotations[rosy_cheeks] == "1":
            face += f" {face_middle} rosy cheeks "
            face_end = ""

        return face, face_end

    def get_nose(self, annotations):
        nose = ""
        nose_middle = ""

        if annotations[pointy_nose] == "1":
            nose = "pointy"
            nose_middle = "and"

        if annotations[big_nose] == "1":
            nose += f"{nose_middle} big"

        if not nose != "":
            nose += " nose"

        return nose


    def generate_description(self, annotations):
        # info from annotations
        gender = self.get_gender(annotations)
        he_she = self.get_gender_auxiliary(annotations)
        his_her = self.get_gender_auxiliary2(annotations)
        hair_type = self.get_hair_type(annotations)
        hair_color = self.get_hair_color(annotations)
        bangs = self.get_bangs(annotations)
        age = self.get_age(annotations)
        face, face_end = self.get_face(annotations)
        nose = self.get_nose(annotations)

        # TODO: Add randomness
        # combinations that will form description

        # hair description
        combination1 = f"A {age} {gender} with {hair_color} {hair_type} hair ."

        # description about bangs

        if bangs != "":
            combination1 = f"A {age} {gender} with {hair_color} {hair_type} hair with bangs ."


        combination2 = ""
        if face != "":
            combination2 = f"{he_she} has {face} {face_end} ."


        combination3 = ""

        if nose != "":
            combination3 = f"{he_she} has {nose}"



        description = f"{combination1}  {combination2} {combination3}"

        return description


    def execute(self, filepath):
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

                if skip:
                    skip = False
                    line = fp.readline()
                    current += 1
                    progress_bar.update(1)
                    continue

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
            print(f"Generated {generated} descriptions.")
