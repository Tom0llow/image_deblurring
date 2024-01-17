from dataset import Dataset

if __name__ == "__main__":
    folder = "./data/results_sharp/mnist"
    folder_to_save = "./dataset/mnist"

    Dataset(path=folder, path_to_save=folder_to_save).split_dataset()
