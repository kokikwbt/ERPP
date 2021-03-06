import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

import erpp
import util


def evaluate():
    model.eval()
    pred_times, pred_events = [], []
    gold_times, gold_events = [], []
    for i, batch in enumerate(tqdm(test_loader)):
        gold_times.append(batch[0][:, -1].numpy())
        gold_events.append(batch[1][:, -1].numpy())
        pred_time, pred_event = model.predict(batch)
        pred_times.append(pred_time)
        pred_events.append(pred_event)
    pred_times = np.concatenate(pred_times).reshape(-1)
    gold_times = np.concatenate(gold_times).reshape(-1)
    pred_events = np.concatenate(pred_events).reshape(-1)
    gold_events = np.concatenate(gold_events).reshape(-1)
    time_error = util.abs_error(pred_times, gold_times)
    acc, recall, f1 = util.clf_metric(pred_events, gold_events, n_class=config.event_class)
    print(f"epoch {epc}")
    print(f"time_error: {time_error}, PRECISION: {acc}, RECALL: {recall}, F1: {f1}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Input
    parser.add_argument('--name', type=str, default='default')
    parser.add_argument('--filename', type=str, default=None)  # df
    parser.add_argument('--model', type=str, default='erpp', choices=['erpp', 'rmtpp'])

    # Feature list
    parser.add_argument('--time_col', type=str, default='time')
    parser.add_argument('--event_col', type=str, default='event')
    parser.add_argument('--ev_ctg_features', type=str, default=None)
    parser.add_argument('--ev_num_features', type=str, default=None)
    parser.add_argument('--ts_ctg_features', type=str, default=None)
    parser.add_argument('--ts_num_features', type=str, default=None)
    parser.add_argument('--freq', type=str, default='D')

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
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--event_class', type=int, default=7)
    parser.add_argument('--verbose_step', type=int, default=1)
    parser.add_argument('--lr', type=int, default=1e-3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--importance_weight', action='store_true')
    parser.add_argument('--calibration_date', type=str, default=None)

    # Output
    parser.add_argument('--outdir', type=str, default='out/tmp/')
    parser.add_argument('--save_model', type=str, default='out/model.pth')
    parser.add_argument('--save_inputs', action='store_true')
    parser.add_argument('--save_config', action='store_true')
    parser.add_argument('--save_prediction', action='store_true')

    config = parser.parse_args()
    data = pd.read_csv(config.filename)

    print('Original data:')
    print(data.head())
    print()

    if config.seed > 0:
        pass

    # Make inputs

    event_handler = util.EventHandler(data, config)
    train_loader = DataLoader(event_handler,
                              batch_size=config.batch_size,
                              shuffle=True,
                              collate_fn=util.EventHandler.to_features)

    weight = np.ones(event_handler.config['event_class'])
    if config.importance_weight:
        weight = event_handler.importance_weight()
        # print('Importance weight:', weight)

    epochs = config.epochs
    model = erpp.ERPP(event_handler.config, lossweight=weight)
    model.set_optimizer(total_step=len(train_loader) * epochs,
                        use_bert=True)
    model.cuda()

    for epc in range(epochs):
        model.train()
        range_loss1 = range_loss2 = range_loss = 0
        desc = 'Epoch {}'.format(epc + 1)
        for i, batch in enumerate(tqdm(train_loader, desc=desc)):
            l1, l2, l = model.train_batch(batch)
            range_loss1 += l1
            range_loss2 += l2
            range_loss += l

        if epc % config.verbose_step == 0:
            print("time  loss:", range_loss1 / config.verbose_step)
            print("event loss:", range_loss2 / config.verbose_step)
            print("total loss:", range_loss  / config.verbose_step)
            range_loss1 = range_loss2 = range_loss = 0

        # evaluate()

    # https://pytorch.org/tutorials/beginner/saving_loading_models.html
    # .pt or .pth
    # How to load: model.load_state_dict(torch.load("model.pth"))
    util.set_workspace(config.outdir)

    if config.save_model is not None:
        torch.save(model.state_dict(), config.outdir + 'model.pth')

    if config.save_inputs is not None:
        pass

    if config.save_config is not None:
        util.saveas_json(event_handler.config,
                         config.outdir + 'config.json')

    if config.save_prediction is not None:
        pass
