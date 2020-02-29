class Args():
    # hyper-parameters
    latent_dim = 100
    batch_size = 16
    channels = 3

    # Paths
    celeba_path = "E:/TezData/img_align_celeba"
    infersent_path = "C:/Users/ulucs/PycharmProjects/uluc_gan_github_version/Pretrained/infersent1.pkl"
    glove_path = "D:/Downloads/Tez vs/glove.840B.300d/glove.840B.300d.txt"
    descriptions_json_path = "E:/TezData/LFW/Face2Text/face2text_v1.0/raw_uluc.json"
    cleanedup_lfw_descriptions = "E:/TezData/cleanedup_lfw.json" # cleaner hierarchy version of raw_uluc.json file
    celeba_annotations_path = "E:/TezData/list_attr_celeba.txt"
    uluc_generated_descriptions = "E:/TezData/generated_attr_celeba.txt"
    annotation_dict_path = "E:/TezData/celeba_annotation_index.txt"
    annotation_sentence_parts = "E:/TezData/sentence_version_annotations.txt"


    # StyleGan Stuff
    sg_img_save_dir = "C:/Users/ulucs/PycharmProjects/uluc_gan_github_version/StyleGanOutputs/train_result/images/"
    sg_model_save_dir = "C:/Users/ulucs/PycharmProjects/uluc_gan_github_version/StyleGanModels/"

    # SimpleShapesDataset Stuff
    shapes_dataset_dir = "E:/TezData/SimpleShapesDataset/images/"
    shapes_descriptions_dir = "E:/TezData/SimpleShapesDataset/descriptions.txt"

    # pretraining rnn-cnn encoder for attnGan stuff
    manualSeed = 1234
    damsm_model_dir = "E:/TezOutputs/DAMSM/model"
    damsm_images_dir = "E:/TezOutputs/DAMSM/img"
    damsm_other_dir = "E:/TezOutputs/DAMSM/othershit"
    damsm_batch = 8
    damsm_epoch = 600
    damsm_encoder_LR = 2e-4