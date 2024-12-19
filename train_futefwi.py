import os
import torch
from torch.nn import DataParallel, functional as F
from torch.utils.data import DataLoader
from utils import parse_train_futefwi_args, create_training_dataset, train, test
from models import FuteFWI


if __name__ == "__main__":
    # parse args
    args = parse_train_futefwi_args()

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
    model = FuteFWI(hidden_size=args.hidden_size, num_layers=args.layers, num_heads=args.heads)
    if args.device == "cuda":
        model = DataParallel(model)
    model = model.to(device)

    # set optimizer
    loss = F.l1_loss
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    # training
    train_loss, val_loss = [], []
    epochs = range(1, args.epochs + 1)
    for epoch in epochs:
        train_loss.append(train(model, device, train_loader, loss, optimizer, epoch, args.epochs))
        val_loss.append(test(model, device, val_loader, loss))

    # save model
    model_to_save = model.module if isinstance(model, DataParallel) else model
    if args.gaussian_noise:
        save_path = os.path.join(args.output, f"{args.name}_{dataset}_ND.pt")
    else:
        save_path = os.path.join(args.output, f"{args.name}_{dataset}_D.pt")
    torch.save(model_to_save.state_dict(), save_path)
