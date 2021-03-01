import argparse
import pandas as pd
import erpp
import util


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Input
    parser.add_argument('--name', type=str, default='default')
    parser.add_argument('--filename', type=str, default=None)
    parser.add_argument('--model', type=str, default='erpp', choices=['erpp', 'rmtpp'])

    # Feature list
    parser.add_argument('--ev_ctg_features', type=str, default=None)
    parser.add_argument('--ts_ctg_features', type=str, default=None)
    parser.add_argument('--ev_num_features', type=str, default=None)
    parser.add_argument('--ts_num_features', type=str, default=None)

    # Model architecture
    parser.add_argument('--ev_seq_len', type=int, default=10)
    parser.add_argument('--ts_seq_len', type=int, default=10)
    parser.add_argument('--ev_emb_dim', type=int, default=10)
    parser.add_argument('--ts_emb_dim', type=int, default=10)
    parser.add_argument('--ev_hid_dim', type=int, default=32)
    parser.add_argument('--ts_hid_dim', type=int, default=32)
    parser.add_argument('--mlp_dim', type=int, default=16)

    # Learning setting
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--event_class', type=int, default=7)
    parser.add_argument('--verbose_step', type=int, default=350)
    parser.add_argument('--importance_weight', action='store_true')
    parser.add_argument('--lr', type=int, default=1e-3)
    parser.add_argument('--epochs', type=int, default=30)

    # Output
    parser.add_argument('--save_model', type=str, default='out/model.pth')

    config = parser.parse_args()

    data = pd.read_csv(config.filename)
    print(data.head())

    data_handler = util.DatasetHandler(data, config)
    print('CATEGORICAL FEATURES IN EVENT SERIES')
    print(data_handler.ev_ctg_features)
    print(data_handler.ev_num_features)
    print(data_handler.ts_ctg_features)
    print(data_handler.ts_num_features)