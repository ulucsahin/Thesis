from random import randint

# Annotation indexes
bald = 4
bangs = 5
black_hair = 8
blond_hair = 9
brown_hair = 11
gray_hair = 17
receding_hairline = 28
straight_hair = 32
wavy_hair = 33

# gender and beard
goatee = 16
male = 20
mustache = 22
no_beard = 24
sideburns = 34

# face
chubby = 13
high_cheekbones = 19
oval_face = 25
rosy_cheeks = 29
double_chin = 14

# nose
big_nose = 7
pointy_nose = 27

# mouth
mouth_open = 21
smiling = 31

# eye area
arched_eyebrows = 1
bags_under_eyes = 3
bushy_eyebrows = 12
narrow_eyes = 23

# attractive and makeup + young
attractive = 2
make_up = 18
young = 39

# skin
pale = 26

# not added: 6,15,30,35,36,37,38

def get_introduction_info(annotations):
    intro = ""
    skin = ""
    age = ""
    gender = ""

    if annotations[39] == "1\n":
        age = " young"
    else:
        rand = randint(0, 1)
        ages = ["n old", " middle aged"]
        age = ages[rand]

    if annotations[male] == "1":
        gender = "male"
    else:
        gender = "female"

    if annotations[pale] == "1":
        rand = randint(0, 2)

        skin1 = " with pale skin and"
        skin2 = " who has a pale skin and"
        skin3 = " that has pale skin"
        skins = [skin1, skin2, skin3]

        skin = skins[rand]


    intro = f"A{age} {gender} {skin}"

    return intro


def get_gender_auxiliary(annotations):
    if annotations[male] == "1":
        return "he"
    else:
        return "she"


def get_gender_auxiliary2(annotations):
    if annotations[male] == "1":
        return "his"
    else:
        return "her"


def get_hair(annotations):
    hair_color = get_hair_color(annotations)
    hair_type = get_hair_type(annotations)
    gender = "woman"
    if annotations[20] == "1":
        gender = "man"

    result = f" with {hair_color} {hair_type} hair "

    if annotations[receding_hairline] == "1":
        rand = randint(0, 1)
        x = [" and a receding hairline", f" , the {gender} has a receding hairline"]
        result += x[rand]

    if annotations[bangs] == "1":
        result += " with bangs"

    if annotations[20] == "1" and annotations[sideburns] == "1":
        result += " and sideburns"

    return result + " ."


def get_hair_color(annotations):
    if annotations[bald] == "1":
        return "balding"
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


def get_hair_type(annotations):
    if annotations[straight_hair] == "1":
        return "straight"
    elif annotations[wavy_hair] == "1":
        return "wavy"
    else:
        return ""


def get_face(annotations):
    face = ""
    face_middle = "a face with "
    face_end = ""

    if annotations[chubby] == "1":
        face = " chubby"
        face_middle = " face with"
        face_end = " face"

    if annotations[oval_face] == "1":
        face += " oval"
        face_middle = " face with"
        face_end = " face"

    # Almost all of them has this so I removed it. Bad annotations
    # if annotations[high_cheekbones] == "1":
    #     face += f" {face_middle} high cheekbones"
    #     face_middle = "and"
    #     face_end = ""

    if annotations[rosy_cheeks] == "1":
        face += f" {face_middle} rosy cheeks "
        face_middle = " and"
        face_end = ""

    if annotations[double_chin] == "1":
        face += f" {face_middle} double chin "
        face_end = ""

    if face != "":
        face = face + face_end + " ."

    return face


def get_nose(annotations):
    nose = ""
    nose_middle = ""

    if annotations[pointy_nose] == "1":
        nose = " pointy"
        nose_middle = " and"

    if annotations[big_nose] == "1":
        nose += f"{nose_middle} big"

    if nose != "":
        nose += " nose"

    return nose


def get_beard(annotations):
    beard = ""

    if annotations[goatee] == "1":
        beard = " a goatee"

    if annotations[mustache] == "1":
        beard = " a mustache"

    if annotations[no_beard] == "1":
        beard = " no beard"

    return beard


def get_eye_information(annotations, he_she, his_her):
    bags = False
    narrow = False
    eyes = ""
    eyes_middle = ""
    eyes_end = ""
    bushy = ""

    if annotations[narrow_eyes] == "1":
        eyes = f" {he_she} has narrow eyes"
        narrow = True

    if annotations[bags_under_eyes] == "1":
        if not narrow:
            eyes = f" {he_she} has bags under {his_her} eyes"
        else:
            eyes += f" with bags under {his_her} eyes"
        bags = True

    if annotations[bushy_eyebrows] == "1":
        bushy = " , bushy"

    if annotations[arched_eyebrows] == "1":
        if bags:
            eyes += f" and {he_she} has arched {bushy} eyebrows"
        else:
            eyes = f" {he_she} has arched {bushy} eyebrows"

    return eyes + " ."


def get_makeup(annotations):
    makeup_ = ""
    young_ = ""
    gender_ = " woman"
    attractive_ = ""
    is_none = True

    if annotations[young] == "1":
        _ = " young"

    if annotations[20] == "1":
        gender_ = " man"

    if annotations[attractive] == "1":
        attractive_ = " attractive"
        is_none = False

    if annotations[make_up] == "1":
        makeup_ = f" The {young_} {attractive_} {gender_} is wearing heavy make up ."
        is_none = False
    else:
        makeup_ = f" The {young_} {gender_} is {attractive_} ."

    if is_none:
        return ""
    else:
        return makeup_

def get_mouth_area(annotations, he_she):
    mouth_middle = ""
    gender = " woman"
    is_none = True
    mouth_open = False

    if annotations[20] == "1":
        gender = " man"

    mouth = f" The {gender}"

    if annotations[mouth_open] == "1":
        mouth += f" has a slightly open mouth"
        mouth_middle = " and"
        is_none = False
        mouth_open = True

    if annotations[smiling] == "1":
        if not mouth_open:
            he_she = ""
        mouth += f" {mouth_middle} {he_she} is smiling ."
        is_none = False
    else:
        x = randint(0,9)
        # add serious with 10% chance because annotations are bad and not all not smiling people are serious looking
        if x==1:
            if not mouth_open:
                he_she = ""
            mouth += f" {mouth_middle} {he_she} looks serious ."
            is_none = False

    if is_none:
        return ""
    else:
        return mouth
