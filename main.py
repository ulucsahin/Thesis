import examiner

def main():
    # Uses sentences in execute method in examiner.py to generate outputs if set to True. Used for examining the model.
    examine = False
    if examine:
        examiner.execute()
        return

    from style_gan_13_text.trainer import train
    train()


if __name__ == "__main__":
    main()
