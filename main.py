import argparse
from general import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # VAE Definition
    parser.add_argument("--hidden_layer", help="hidden layer dimension")
    parser.add_argument("--features", help="bottleneck dimension")
    parser.add_argument("--dropout", help="dropout")

    # Setup options
    parser.add_argument("--loss", help="loss function", choices=["bce", "mse"], default="mse")
    parser.add_argument("--lr", help="learning rate", default=0.0001)

    # Training options
    parser.add_argument("--epochs", help="epochs per cycle")
    parser.add_argument("--beta", help="beta")
    parser.add_argument("--c", help="channel capacity")

    # Data
    parser.add_argument("--data_path", help="path to data csv")
    parser.add_argument("--batch_size", help="batch size", default=32)
    parser.add_argument("--scaling", help="scaling data", default=0)

    # Plots

    parser.add_argument("--plots", help="loss plots", choices=["0", "1"], default="0")

    args = parser.parse_args()

    # input_dim = int(args.input_dim)
    mid_dim = int(args.hidden_layer)
    features = int(args.features)
    dropout = float(args.dropout)

    lr = float(args.lr)

    ch_epochs = int(args.epochs)
    ch_beta = float(args.beta)
    ch_c = float(args.c)

    ch_batch_size = int(args.batch_size)
    ch_scaling = int(args.scaling)

    tr_l, tt_l, tr_kl, tt_kl, tr_r, tt_r, id_string, dev = training(mid_dim=mid_dim, features=features,
                                                                    dropout=dropout,
                                                                    data_path=args.data_path,
                                                                    batch_size=ch_batch_size,
                                                                    scaling=ch_scaling,
                                                                    epochs=ch_epochs,
                                                                    beta=ch_beta,
                                                                    c=ch_c,
                                                                    learning_rate=lr)

    if args.plots == "1":
        loss_plots(tr_l, tt_l, tr_kl, tt_kl, tr_r, tt_r, id_string)
