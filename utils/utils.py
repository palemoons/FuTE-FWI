"""Utility functions for training and testing"""

import torch
from torch.nn import functional as F
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from einops import rearrange
from tqdm.auto import tqdm
from pytorch_msssim import ssim, ms_ssim
import lpips
import time
import os
from utils.dataset import Dataset, LargeDataset, TestDataset

vel_families = ["FlatVel_A", "FlatVel_B", "CurveVel_A", "CurveVel_B"]
fault_familyies = ["FlatFault_A", "FlatFault_B", "CurveFault_A", "CurveFault_B"]


def create_training_dataset(dataset, gaussian_noise, memmap=True):
    """Create datasets according to different dataset family.

    Memory mapping (LargeDataset) is applied for the sake of the limited RAM. Note that adding gaussian noise is not
    implemented in LargeDataset.

    For who has enough memory capacity, set `memmap=False` to accelerate training.
    """
    if dataset in vel_families:
        train_range = range(0, 48)
        val_range = range(48, 60)
    elif dataset in fault_familyies:
        train_range = range(0, 96)
        val_range = range(96, 108)
    else:
        raise NotImplementedError("Unsupport dataset")
    if memmap:
        if gaussian_noise:
            train_set = LargeDataset(os.path.join("data", f"{dataset}_ND"), list(train_range))
            val_set = LargeDataset(os.path.join("data", f"{dataset}_ND"), list(val_range))
        else:
            train_set = LargeDataset(os.path.join("data", f"{dataset}_D"), list(train_range))
            val_set = LargeDataset(os.path.join("data", f"{dataset}_D"), list(val_range))
    else:
        if gaussian_noise:
            train_set = Dataset(os.path.join("data", f"{dataset}_ND"), list(train_range))
            val_set = Dataset(os.path.join("data", f"{dataset}_ND"), list(val_range))
        else:
            train_set = Dataset(os.path.join("data", f"{dataset}_D"), list(train_range))
            val_set = Dataset(os.path.join("data", f"{dataset}_D"), list(val_range))
    return train_set, val_set


def create_testing_dataset(dataset, gaussian_noise):
    if dataset in vel_families:
        val_range = range(48, 60)
    elif dataset in fault_familyies:
        val_range = range(96, 108)
    else:
        raise NotImplementedError("Unsupport dataset")

    if gaussian_noise:
        dataset = TestDataset(os.path.join("data", f"{dataset}_ND"), list(val_range))
    else:
        dataset = TestDataset(os.path.join("data", f"{dataset}_D"), list(val_range))
    return dataset


def train(model, device, dataloader, loss_fn, optimizer, epoch, epochs):
    model.train()
    batch_num = len(dataloader)
    train_loss = 0
    for data, target in tqdm(dataloader, desc=f"Epoch [{epoch}/{epochs}]", ncols=70):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        l = loss_fn(output, target)
        l.backward()
        optimizer.step()
        train_loss += l.item()
    train_loss /= batch_num
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{now}] Train set: Average loss: {train_loss:.6f}")
    return train_loss


def test(model, device, dataloader, loss_fn):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in tqdm(dataloader, desc="Testing", ncols=70):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()  # sum up batch loss
    test_loss /= len(dataloader)
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{now}] Test set: Average loss: {test_loss:.6f}\n")
    return test_loss


def train_gan(
    model,
    model_d,
    device,
    dataloader,
    loss_g,
    loss_d,
    optimizer_g,
    optimizer_d,
    epoch,
    epochs,
    update_interval,
):
    """Train function for VelocityGAN"""
    model.train()
    train_loss = 0
    for batch_idx, (data, label) in enumerate(tqdm(dataloader, desc=f"Epoch [{epoch}/{epochs}]", ncols=70)):
        data, label = data.to(device), label.to(device)
        optimizer_d.zero_grad()
        output = model(data)
        ld = loss_d(output, label, model_d)[0]
        ld.backward()
        optimizer_d.step()
        train_loss += ld.item()

        if (batch_idx + 1) % update_interval == 0:
            optimizer_g.zero_grad()
            pred = model(data)
            lg = loss_g(pred, label, model_d)
            lg.backward()
            optimizer_g.step()

    train_loss /= len(dataloader)
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{now}] Train set: Average discriminator loss: {train_loss:.6f}")
    return train_loss


def test_gan(model, device, dataloader, loss_fn):
    """Test function for VelocityGAN"""
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, label in tqdm(dataloader, desc="Testing", ncols=70):
            data = data.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            output = model(data)
            test_loss += loss_fn(output, label).item()
    test_loss /= len(dataloader)
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{now}] Test set: Average generator loss: {test_loss:.6f}\n")
    return test_loss


def evaluate(dataloader, model, device):
    mae, mse, ssim_loss, mssim_loss, ps = 0, 0, 0, 0, 0
    total_samples = len(dataloader.dataset)
    lpips_loss = lpips.LPIPS(net="alex", verbose=False)
    if device == torch.device("cuda"):
        lpips_loss = lpips_loss.cuda()
    for data, label, _ in tqdm(dataloader, desc="Evaluating", ncols=60):
        batch_size = label.size(0)
        data, label = data.to(device), label.to(device)
        output = model(data)
        mae += torch.mean(torch.abs(output - label) * batch_size).item()
        mse += torch.mean((output - label) ** 2 * batch_size).item()
        ssim_loss += torch.sum(
            ssim(output / 2 + 0.5, label / 2 + 0.5, data_range=1, size_average=False)
        )  # [-1, 1] -> [0, 1]
        mssim_loss += torch.sum(
            ms_ssim(output / 2 + 0.5, label / 2 + 0.5, data_range=1, size_average=False, win_size=3)
        )  # [-1, 1] -> [0, 1]
        ps += torch.sum(lpips_loss(output, label))
    mae /= total_samples
    mse /= total_samples
    ssim_loss /= total_samples
    mssim_loss /= total_samples
    ps /= total_samples
    print(
        f"MAE: {mae:.4f}",
        f"MSE: {mse:.4f}",
        f"SSIM: {ssim_loss:.4f}",
        f"MS-SSIM: {mssim_loss:.4f}",
        f"LPIPS: {ps:.4f}",
        sep="\n",
    )


def plot_vmodel(dataloader, model, save_name, device):
    # Generate sample list
    data, _, label = next(iter(dataloader))
    output = torch.squeeze(model(data.to(device)).cpu()).numpy()
    label = torch.squeeze(label).numpy()

    # Renormalize
    max_vals = np.max(label)
    min_vals = np.min(label)
    output = (output / 2 + 0.5) * (max_vals - min_vals) + min_vals

    fig, axs = plt.subplots(1, 2, figsize=(6, 6.5))
    vmin = min(output.min(), min_vals)
    vmax = max(output.max(), max_vals)
    norm = matplotlib.colors.Normalize(vmin, vmax)
    mappable = matplotlib.cm.ScalarMappable(norm)

    axs[0].imshow(output, norm=norm)
    axs[0].set_title(f"Prediction", {"fontsize": 12})
    axs[1].imshow(label, norm=norm)
    axs[1].set_title(f"Label", {"fontsize": 12})

    cb_ax = fig.add_axes([0.1, 0.1, 0.8, 0.02])
    fig.colorbar(mappable, cax=cb_ax, orientation="horizontal")
    fig.suptitle(save_name, y=0.95, fontsize=20, fontweight=500)
    plt.show()
    plt.savefig(f"{save_name}.png")


def _plot_loss(epoch, train_loss, val_loss, model, dataset):
    _, ax = plt.subplots()
    ax.plot(range(1, epoch + 1), train_loss, "b", label="Training loss")
    ax.plot(range(1, epoch + 1), val_loss, "r", label="Validation loss")
    plt.title("Training and validation loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend()
    plt.savefig(f"loss_{model}_{dataset}.png")
    plt.close()


def _median_filter(img, kernel_size=3):
    """img: b c h w"""
    pad_size = kernel_size // 2
    padded_img = F.pad(img, (pad_size, pad_size, pad_size, pad_size), mode="reflect")
    unfolded_img = padded_img.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)
    unfolded_img = rearrange(unfolded_img, "b c h w k1 k2 -> b c h w (k1 k2)")

    median_filtered_img = unfolded_img.median(dim=-1)[0]
    return median_filtered_img
