import time
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader


# ImageNet RGB mean and std
IMAGENET_RGB_MEAN = (0.485, 0.456, 0.406)
IMAGENET_RGB_STD = (0.229, 0.224, 0.225)

# Mean and std to put images in the range [-1, 1]
TANH_RGB_MEAN = (0.5, 0.5, 0.5)
TANH_RGB_STD = (0.5, 0.5, 0.5)


def timer(start, end):
    # Return the difference between <start> time and <end> time as a string in the format of "mm:ss"
    minutes, seconds = divmod(end - start, 60)
    return "{:0>2}:{:0>2}".format(int(minutes), int(seconds))


def accuracy(y_true, y_pred):
    # Calculate accuracy between tensors. Meant for classification tasks
    correct = torch.eq(y_true, y_pred).sum().item()  # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100
    return acc


def train_step(model, data_loader, loss_fn, optimizer, accuracy_fn, device, use_tqdm=False):
    train_loss, train_acc = 0, 0
    start = time.time()
    model.train()
    for batch in tqdm(data_loader, disable=not use_tqdm):
        # Send data to GPU
        X = batch["image"].to(device)
        y = batch["label"].to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))  # Go from logits -> pred labels

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    end = time.time()
    # Calculates loss and accuracy per epoch and print out what's happening
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}% | Time elapsed: {timer(start, end)}")


def test_step(model, data_loader, loss_fn, accuracy_fn, device, use_tqdm=False):
    test_loss, test_acc = 0, 0
    model.eval()  # put model in eval mode
    # Turn on inference context manager
    with torch.inference_mode():
        start = time.time()
        for batch in tqdm(data_loader, disable=not use_tqdm):
            # Send data to GPU
            X = batch["image"].to(device)
            y = batch["label"].to(device)

            # 1. Forward pass
            test_pred = model(X)

            # 2. Calculate loss and accuracy
            loss = loss_fn(test_pred, y)
            test_loss += loss
            test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))  # Go from logits -> pred labels

        end = time.time()
        # Adjust metrics and print out
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Valid loss: {test_loss:.5f} | Valid accuracy: {test_acc:.2f}% | Time elapsed: {timer(start, end)}")


def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def get_dataloader(dset, batch_size=32, shuffle=True, num_workers=0):
    dl = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dl


def tensor_2_numpy(t):
    # convert a (C, H, W) pytorch tensor to a (H, W, C) numpy array
    return torch.permute(t, (1, 2, 0)).numpy()


def numpy_2_tensor(arr):
    # convert a (H, W, C) numpy array to a (C, H, W) pytorch tensor
    return torch.from_numpy(np.transpose(arr, (0, 1, 2)))


def convolutional_output_size(in_size, padding, kernel_size, stride):
    # find the size (H/W) of the output of a convolutional layer
    return np.floor((in_size + 2 * padding - kernel_size) / stride) + 1


def find_same_padding(in_size, kernel_size, stride):
    # find the size of the padding for "same" padding
    return np.ceil(((stride - 1) * in_size - stride + kernel_size) / 2)
