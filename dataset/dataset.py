import os
import shutil
import random


class Dataset:
    def __init__(self, path, path_to_save, train_size=0.7, val_size=0.2, test_size=0.1):
        self.path = path
        self.path_to_save = path_to_save

        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size

    def train_test_split(self, path, train_dir, val_dir, test_dir):
        files = os.listdir(path)
        random.shuffle(files)

        sample_num = len(files)
        train_num = 0
        val_num = 0
        test_num = 0

        tr_split_id = int(sample_num * self.train_size)
        te_split_id = -int(sample_num * self.test_size)

        # copy to train
        for fname in files[:tr_split_id]:
            src = os.path.join(path, fname)
            dst = os.path.join(train_dir, fname)
            shutil.copyfile(src, dst)
            train_num += 1
        # copy to val
        for fname in files[tr_split_id:te_split_id]:
            src = os.path.join(path, fname)
            dst = os.path.join(val_dir, fname)
            shutil.copyfile(src, dst)
            val_num += 1
        # copy to test
        for fname in files[te_split_id:]:
            src = os.path.join(path, fname)
            dst = os.path.join(test_dir, fname)
            shutil.copyfile(src, dst)
            test_num += 1
        # check to see if the dataset is split
        class_name = os.path.basename(os.path.basename(path))
        self.check_split(train_num, val_num, test_num, sample_num, class_name)

    def check_split(self, train_num, val_num, test_num, sample_num, class_name):
        print(f"train: {train_num}")
        print(f"val  : {val_num}")
        print(f"test : {test_num}")

        if train_num + val_num + test_num == sample_num:
            print(f"{class_name}: Complete split datasets!")
        else:
            print(f"{class_name}: The sum of the training size and test size does not match the number of samples.")

    def split_dataset(self):
        try:
            os.mkdir(self.path_to_save)
        except FileExistsError:
            print(f"{self.path_to_save} is already exist.")

        dir_lists = os.listdir(self.path)
        dir_lists = [f for f in dir_lists if os.path.isdir(os.path.join(self.path, f))]
        path_list = [os.path.join(self.path, p) for p in dir_lists]

        try:
            train_dir = os.path.join(self.path_to_save, "train")
            os.mkdir(train_dir)
        except FileExistsError:
            print(f"{train_dir} is already exist.")
        try:
            val_dir = os.path.join(self.path_to_save, "val")
            os.mkdir(val_dir)
        except FileExistsError:
            print(f"{val_dir} is already exist.")
        try:
            test_dir = os.path.join(self.path_to_save, "test")
            os.mkdir(test_dir)
        except FileExistsError:
            print(f"{test_dir} is already exist.")

        train_dir_path_lists = []
        val_dir_path_lists = []
        test_dir_path_lists = []
        for D in dir_lists:
            train_class_dir_path = os.path.join(train_dir, D)
            try:
                os.mkdir(train_class_dir_path)
            except FileExistsError:
                print(f"{train_class_dir_path} is already exist.")
            train_dir_path_lists += [train_class_dir_path]

            val_class_dir_path = os.path.join(val_dir, D)
            try:
                os.mkdir(val_class_dir_path)
            except FileExistsError:
                print(f"{val_class_dir_path} is already exist.")
            val_dir_path_lists += [val_class_dir_path]

            test_class_dir_path = os.path.join(test_dir, D)
            try:
                os.mkdir(test_class_dir_path)
            except FileExistsError:
                print(f"{test_class_dir_path} is already exist.")
            test_dir_path_lists += [test_class_dir_path]

        # split dataset
        if len(path_list) >= 1:
            for i, path in enumerate(path_list):
                self.train_test_split(path=path, train_dir=train_dir_path_lists[i], val_dir=val_dir_path_lists[i], test_dir=test_dir_path_lists[i])
        else:
            path = self.path
            self.train_test_split(path=path, train_dir=train_dir, val_dir=val_dir, test_dir=test_dir)
