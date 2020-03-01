import examiner
from args import Args

def main():
    # Uses sentences in execute method in examiner.py to generate outputs if set to True. Used for examining the model.
    examine = False
    if examine:
        examiner.execute()
        return

    description_generate = False
    if description_generate:
        from celeba_description_generator import CelebaDescriptionGenerator, generate_new_dataset_json

        description_generator = CelebaDescriptionGenerator(cleanup_data=True)
        description_generator.execute(Args.celeba_annotations_path)
        generate_new_dataset_json()
        return

    from style_gan_13_text.trainer import train
    train()


if __name__ == "__main__":
    main()
