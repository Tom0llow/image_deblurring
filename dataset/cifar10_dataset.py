import os
import shutil
import random


def merge_dir(path):
    files = []
    for label in os.listdir(path):
        for file in os.listdir(os.path.join(path, label)):
            files.append((label, file))
    random.shuffle(files)
    return files


if __name__ == "__main__":
    path = "/root/workspace/workspace/data/raw/cifar10"
    path_to_save = "/root/workspace/workspace/dataset/cifar10"

    train_path = os.path.join(path, "train")
    test_path = os.path.join(path, "test")
    train_files = merge_dir(train_path)
    test_files = merge_dir(test_path)

    # make dir
    try:
        os.mkdir(path_to_save)
    except FileExistsError:
        print(f"{path_to_save} is already exist.")

    try:
        train_dir = os.path.join(path_to_save, "train")
        os.mkdir(train_dir)
    except FileExistsError:
        print(f"{train_dir} is already exist.")
    try:
        test_dir = os.path.join(path_to_save, "test")
        os.mkdir(test_dir)
    except FileExistsError:
        print(f"{test_dir} is already exist.")

    # copy to train
    for label, fname in train_files:
        src = os.path.join(train_path, label, fname)
        dst = os.path.join(train_dir, fname)
        shutil.copyfile(src, dst)
    # copy to test
    for label, fname in test_files:
        src = os.path.join(test_path, label, fname)
        dst = os.path.join(test_dir, fname)
        shutil.copyfile(src, dst)
