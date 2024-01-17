from dataset import Dataset

if __name__ == "__main__":
    folder = "./data/raw/RandomMotionBlur"
    folder_to_save = "./dataset/RandomMotionBlur"

    Dataset(path=folder, path_to_save=folder_to_save).split_dataset()
