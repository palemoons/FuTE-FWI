import torch
from torch.utils.data import DataLoader
from torch.nn import DataParallel
import os
from models import Generator, Discriminator
from utils import parse_train_velocitygan_args, create_training_dataset, train_gan, test_gan
from utils import Wasserstein_GP, UnionLoss


if __name__ == "__main__":
    # parse args
    args = parse_train_velocitygan_args()

    # create output dir
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # load dataset
    print("Loading dataset...")
    dataset = f"{args.dataset}_{args.version}"
    train_set, val_set = create_training_dataset(dataset, args.gaussian_noise)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, pin_memory=True, shuffle=True)

    # create model
    device = torch.device(args.device)
    print("Using device:", device)
    model, model_d = Generator(), Discriminator()
    if args.device == "cuda":
        model = DataParallel(model)
        model_d = DataParallel(model_d)
    model.to(device)
    model_d.to(device)

    # set optimizer
    loss_g = UnionLoss(args.lambda_g1v, args.lambda_g2v)
    loss_d = Wasserstein_GP(device, args.lambda_gp)
    optimizer_g = torch.optim.AdamW(model.parameters(), lr=args.lr_g, weight_decay=args.weight_decay)
    optimizer_d = torch.optim.AdamW(model_d.parameters(), lr=args.lr_d, weight_decay=args.weight_decay)

    # training
    train_loss, val_loss = [], []
    epochs = range(1, args.epochs + 1)
    for epoch in epochs:
        train_loss.append(
            train_gan(
                model,
                model_d,
                device,
                train_loader,
                loss_g,
                loss_d,
                optimizer_g,
                optimizer_d,
                epoch,
                args.epochs,
                args.update_interval,
            )
        )
        val_loss.append(test_gan(model, device, val_loader, loss_g))

    # save model
    model_to_save = model.module if isinstance(model, DataParallel) else model
    if args.gaussian_noise:
        save_path = os.path.join(args.output, f"{args.name}_{dataset}_ND.pt")
    else:
        save_path = os.path.join(args.output, f"{args.name}_{dataset}_D.pt")
    torch.save(model_to_save.state_dict(), save_path)
