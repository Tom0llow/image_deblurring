from dataset import Dataset

if __name__ == "__main__":
    folder = "./data/results_sharp/celebahq_256"
    folder_to_save = "./dataset/celebahq_256"

    Dataset(path=folder, path_to_save=folder_to_save).split_dataset()
