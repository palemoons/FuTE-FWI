import torch
from torch.nn import functional as F
import numpy as np
from tqdm import trange
import os
from deepwave import scalar
from deepwave.wavelets import ricker
from utils import parse_gen_dataset_args


def forward_modeling(vmodel_fpath, seis_fpath, device, gaussian_noise=None):
    vmodels = np.load(vmodel_fpath).squeeze(1)
    seis = []
    for vmodel in vmodels:
        vmodel = torch.from_numpy(vmodel.T).to(device)

        # set parameters
        epsilon = 1e-6  # prevent source out of index
        dx = 10.0  # grid spacing
        n_shots = 5  # shot number
        n_sources_per_shot = 1  # source number in each shot
        d_source = 17.5 - epsilon  # source interval
        first_source = 0  # location of first source
        source_depth = 0  # depth of source
        n_receivers_per_shot = 70  # receiver number in each shot
        d_receiver = 1  # receiver interval
        first_receiver = 0  # location of first receiver
        receiver_depth = 0  # depth of receiver
        freq = 15  # source frequency
        nt = 2000  # time steps
        dt = 0.001  # time spacing
        peak_time = 1 / freq  # peak time of Ricker wavelet

        # set location
        # source_locations
        source_locations = torch.zeros(n_shots, n_sources_per_shot, 2, dtype=torch.long, device=device)
        source_locations[..., 1] = source_depth
        source_locations[:, 0, 0] = torch.arange(n_shots) * d_source + first_source
        # receiver_locations
        receiver_locations = torch.zeros(n_shots, n_receivers_per_shot, 2, dtype=torch.long, device=device)
        receiver_locations[..., 1] = receiver_depth
        receiver_locations[:, :, 0] = (torch.arange(n_receivers_per_shot) * d_receiver + first_receiver).repeat(
            n_shots, 1
        )
        # source_amplitudes
        source_amplitudes = ricker(freq, nt, dt, peak_time).repeat(n_shots, n_sources_per_shot, 1).to(device)

        # Propagate
        out = scalar(
            vmodel,
            dx,
            dt,
            source_amplitudes=source_amplitudes,
            source_locations=source_locations,
            receiver_locations=receiver_locations,
            accuracy=8,
            pml_freq=freq,
        )

        # save seismic record
        seis.append(np.transpose(out[-1].cpu().numpy(), (0, 2, 1)))
    seis = np.array(seis)
    if gaussian_noise is not None:
        max_vals, min_vals = np.max(seis, axis=(1, 2, 3), keepdims=True), np.min(seis, axis=(1, 2, 3), keepdims=True)
        normalized = ((seis - min_vals) / (max_vals - min_vals) - 0.5) * 2
        noise = np.random.normal(0, np.sqrt(gaussian_noise), normalized[0].shape).astype(normalized.dtype)
        seis = ((normalized + noise) / 2 + 0.5) * (max_vals - min_vals) + min_vals
    np.save(seis_fpath, seis)


def interpolate_vmodel(vmodel_fpath, new_vmodel_fpath):
    vmodels = torch.from_numpy(np.load(vmodel_fpath))
    _, _, h, w = vmodels.shape
    new_vmodels = F.interpolate(vmodels, [2 * h, w], mode="nearest").numpy()
    np.save(new_vmodel_fpath, new_vmodels)


if __name__ == "__main__":
    args = parse_gen_dataset_args()
    dataset_name = f"{args.dataset}_{args.version}"
    gn = args.gaussian_noise

    vmodel_dir = os.path.join("./data", dataset_name, "model")
    if gn is None:
        new_dir = os.path.join("./data", f"{dataset_name}_D")
        new_vmodel_dir = os.path.join("./data", f"{dataset_name}_D", "model")
        new_seis_dir = os.path.join("./data", f"{dataset_name}_D", "data")
    else:
        new_dir = os.path.join("./data", f"{dataset_name}_ND")
        new_vmodel_dir = os.path.join("./data", f"{dataset_name}_ND", "model")
        new_seis_dir = os.path.join("./data", f"{dataset_name}_ND", "data")

    # create target dir is not exists
    if not os.path.exists(new_dir):
        os.mkdir(os.path.join(new_dir))
    if not os.path.exists(new_seis_dir):
        os.mkdir(new_seis_dir)
    if not os.path.exists(new_vmodel_dir):
        os.mkdir(new_vmodel_dir)

    # determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    vmodels = os.listdir(vmodel_dir)
    vmodels.sort()
    if "Vel" in args.dataset:
        """Vel Family"""
        for i in trange(len(vmodels), ncols=60):
            vmodel_id = vmodels[i][5:-4]
            interpolate_vmodel(
                os.path.join(vmodel_dir, f"model{vmodel_id}.npy"),
                os.path.join(new_vmodel_dir, f"model{vmodel_id}.npy"),
            )
            forward_modeling(
                os.path.join(new_vmodel_dir, f"model{vmodel_id}.npy"),
                os.path.join(new_seis_dir, f"data{vmodel_id}.npy"),
                device,
                gn,
            )
    elif "Fault" in args.dataset:
        """Fault Family"""
        for i in trange(len(vmodels), ncols=60):
            vmodel_id = vmodels[i][3:-4]
            interpolate_vmodel(
                os.path.join(vmodel_dir, f"vel{vmodel_id}.npy"),
                os.path.join(new_vmodel_dir, f"vel{vmodel_id}.npy"),
            )
            forward_modeling(
                os.path.join(new_vmodel_dir, f"vel{vmodel_id}.npy"),
                os.path.join(new_seis_dir, f"seis{vmodel_id}.npy"),
                device,
                gn,
            )
