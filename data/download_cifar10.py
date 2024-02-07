import os
import tarfile
import requests
import pickle
import joblib
from PIL import Image
import numpy as np
import tqdm


# Download the tar.gz file
def download_cifar10(save_dir):
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    header = requests.head(url).headers
    size = int(header["Content-Length"])
    with open(os.path.join(save_dir, "cifar-10-python.tar.gz"), "wb") as f:
        pbar = tqdm.tqdm(total=size, unit="B", unit_scale=True)
        for chunk in requests.get(url, stream=True).iter_content(chunk_size=1024):
            ff = f.write(chunk)
            pbar.update(len(chunk))
        pbar.close()


# Unpickle the file
def unpickle(file):
    with open(file, "rb") as f:
        dict = pickle.load(f, encoding="bytes")
    return dict


# Save cifar10 image
def save_image(label, data, fname, type="train"):
    img = data.reshape(3, 32, 32)
    img = np.transpose(img, (1, 2, 0))
    pil_img = Image.fromarray(img)
    dir = f"./raw/cifar10/{type}/{label}"
    os.makedirs(dir, exist_ok=True)
    pil_img.save(f"{dir}/{fname.decode('utf-8')}")


if __name__ == "__main__":
    save_dir = "./raw/cifar10"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Process start: save directory={save_dir}")

    print("Download the CIFAR10.")
    if not os.path.isfile(os.path.join(save_dir, "cifar-10-python.tar.gz")):
        download_cifar10(save_dir)

    print("Extract tar.gz.")
    with tarfile.open(os.path.join(save_dir, "cifar-10-python.tar.gz"), "r:gz") as tar:
        tar.extractall(path=save_dir)

    print("Save the training data as png.")
    for i in range(5):
        unp = unpickle(os.path.join(save_dir, f"cifar-10-batches-py/data_batch_{i+1}"))
        joblib.Parallel(n_jobs=-1)(joblib.delayed(save_image)(label, data, filename) for label, data, filename in zip(unp[b"labels"], unp[b"data"], unp[b"filenames"]))

    print("Save the test data as png.")
    unp = unpickle(os.path.join(save_dir, "cifar-10-batches-py/test_batch"))
    joblib.Parallel(n_jobs=-1)(joblib.delayed(save_image)(label, data, filename, "test") for label, data, filename in zip(unp[b"labels"], unp[b"data"], unp[b"filenames"]))

    print("Done.")
