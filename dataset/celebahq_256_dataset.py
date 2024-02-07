from dataset import Dataset

if __name__ == "__main__":
    folder = "/root/workspace/workspace/data/results_sharp/celebahq_256"
    folder_to_save = "/root/workspace/workspace/dataset/celebahq_256"

    Dataset(path=folder, path_to_save=folder_to_save).split_dataset()
