import argparse


def parse_gen_dataset_args():
    parser = argparse.ArgumentParser(
        prog="Deep-OpenFWI-Generator", description="Generate Deep-OpenFWI dataset from velocity models in OpenFWI"
    )
    parser.add_argument("-d", "--dataset", type=str, required=True, help="dataset of OpenFWI")
    parser.add_argument("-v", "--version", choices=["A", "B"], help="version of dataset, easy (A) or hard (B)")
    parser.add_argument(
        "-gn", "--gaussian-noise", type=float, default=None, help="variance of gaussian noise (default: None)"
    )
    return parser.parse_args()


def parse_train_futefwi_args():
    """Args parser for training FuteFWI"""
    parser = argparse.ArgumentParser(prog="FuteFWI-Trainer", description="Pytorch trainer for FuteFWI")
    # for dataset
    parser.add_argument("-d", "--dataset", type=str, required=True, help="family of dataset")
    parser.add_argument(
        "-v", "--version", choices=["A", "B"], required=True, help="version of dataset, easy (A) or hard (B)"
    )
    parser.add_argument("-gn", "--gaussian-noise", action="store_true", help="add gaussian noise")
    # for model
    parser.add_argument("--hidden-size", type=int, default=768, help="number of dimensions of MLP (default: 768)")
    parser.add_argument("--layers", type=int, default=4, help="number of layers of transformer module (default: 4)")
    parser.add_argument("--heads", type=int, default=12, help="number of heads of multiheaded attention (default: 12)")
    # for training hyperparameters
    parser.add_argument("--device", default="cuda", help="device (default: cuda)")
    parser.add_argument("--batch-size", type=int, default=64, help="batch size for training (default: 50)")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs to train (default: 100)")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate (default: 1e-4)")
    # for saving
    parser.add_argument("-o", "--output", type=str, default="out", help="parent dir of output (default: out)")
    parser.add_argument("-n", "--name", type=str, default="FuteFWI", help="name of saved model (default: FuteFWI)")
    return parser.parse_args()


def parse_train_inversionnet_args():
    """Args parser for training InversionNet"""
    parser = argparse.ArgumentParser(prog="InversionNet-Trainer", description="Pytorch trainer for InversionNet")
    # for dataset
    parser.add_argument("-d", "--dataset", type=str, required=True, help="family of dataset")
    parser.add_argument(
        "-v", "--version", choices=["A", "B"], required=True, help="version of dataset, easy (A) or hard (B)"
    )
    parser.add_argument("-gn", "--gaussian-noise", action="store_true", help="add gaussian noise")
    # for training hyperparameters
    parser.add_argument("--device", default="cuda", help="device (default: cuda)")
    parser.add_argument("--batch-size", type=int, default=128, help="batch size for training (default: 256)")
    parser.add_argument("--epochs", type=int, default=120, help="number of epochs to train (default: 120)")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate (default: 1e-4)")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    # for saving
    parser.add_argument(
        "-o", "--output", type=str, default="out", help="path to parent folder to save models (default: out)"
    )
    parser.add_argument(
        "-n", "--name", type=str, default="InversionNet", help="name of saved model (default: InversionNet)"
    )
    return parser.parse_args()


def parse_train_velocitygan_args():
    """Args parser for training VelocityGAN"""
    parser = argparse.ArgumentParser(prog="VelocityGAN-Trainer", description="Pytorch trainer for VelocityGAN")
    # for dataset
    parser.add_argument("-d", "--dataset", type=str, required=True, help="family of dataset")
    parser.add_argument(
        "-v", "--version", choices=["A", "B"], required=True, help="version of dataset, easy (A) or hard (B)"
    )
    parser.add_argument("-gn", "--gaussian-noise", action="store_true", help="add gaussian noise")
    # for training hyperparameters
    parser.add_argument("--device", default="cuda", help="device (default: cuda)")
    parser.add_argument("--batch-size", type=int, default=128, help="batch size for training (default: 64)")
    parser.add_argument("--epochs", type=int, default=480, help="number of epochs to train (default: 480)")
    parser.add_argument("--lr-g", type=float, default=1e-4)
    parser.add_argument("--lr-d", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("-gp", "--lambda_gp", type=float, default=10.0)
    parser.add_argument("-g1v", "--lambda_g1v", type=float, default=50.0)
    parser.add_argument("-g2v", "--lambda_g2v", type=float, default=100.0)
    parser.add_argument("--update-interval", type=int, default=3)
    # for saving
    parser.add_argument(
        "-o", "--output", type=str, default="out", help="path to parent folder to save models (default: out)"
    )
    parser.add_argument(
        "-n", "--name", type=str, default="VelocityGAN", help="name of saved model (default: VelocityGAN)"
    )
    return parser.parse_args()


def parse_test_args():
    parser = argparse.ArgumentParser(prog="Test", description="Test program for FuteFWI, InversionNet and VelocityGAN")
    # for model
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=["FuteFWI", "InversionNet", "VelocityGAN"],
        required=True,
        help="model to test",
    )
    parser.add_argument("-p", "--path", type=str, default="out", help="path of model (default: out)")
    parser.add_argument("-n", "--name", type=str, default=None, help="name of model (default: None)")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="family of dataset")
    parser.add_argument(
        "-v", "--version", choices=["A", "B"], required=True, help="version of dataset, easy (A) or hard (B)"
    )
    parser.add_argument("-gn", "--gaussian-noise", action="store_true", help="add gaussian noise")
    # for testing
    parser.add_argument("--device", default="cuda", help="device (default: cuda)")
    parser.add_argument("--batch-size", type=int, default=64, help="batch size for testing (default: 64)")
    parser.add_argument("--seed", type=int, default=42, help="seed for plotting velocity model randomly (default: 42)")
    # for drawing comparison figure
    parser.add_argument("--draw", action="store_true", help="draw comparison example")
    return parser.parse_args()
