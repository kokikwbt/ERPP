import torch
from torch import nn
from torch.optim import Adam
import numpy as np
from optimization import BertAdam


class ERPP(nn.Module):
    def __init__(self, config, lossweight):
        super(ERPP, self).__init__()

        self.config = config  # dict
        self.n_class = config['event_class']
        self.emb_drop = nn.Dropout(p=config['dropout'])

        self.event_embeddings = [
            nn.Embedding(num_embeddings=config['ev_ndim_' + s],
                         embedding_dim=config['ev_embdim_' + s])
            for s in config['ev_ctg_features']
        ]

        input_size = config['ev_embdim_total'] + config['ev_ndim_num'] + 1  # +1 for time
        self.event_lstm = nn.LSTM(input_size=input_size,
                                  hidden_size=config['ev_hid_dim'],
                                  batch_first=True,
                                  bidirectional=False)

        self.ts_embeddings = [
            nn.Embedding(num_embeddings=config['ts_ndim_' + s],
                         embedding_dim=config['ts_embdim_' + s])
            for s in config['ts_ctg_features']
        ]

        input_size = config['ts_embdim_total'] + config['ts_ndim_num']
        self.ts_lstm = nn.LSTM(input_size=input_size,
                               hidden_size=config['ts_hid_dim'],
                               batch_first=True,
                               bidirectional=False)

        # Feature mapping layer
        self.mlp = nn.Linear(in_features=config['ev_hid_dim']+config['ts_hid_dim'],
                             out_features=config['mlp_dim'])

        self.mlp_drop = nn.Dropout(p=config['dropout'])

        # Output layer
        self.event_linear = nn.Linear(in_features=config['mlp_dim'],
                                      out_features=config['event_class'])
        self.time_linear = nn.Linear(in_features=config['mlp_dim'], out_features=1)
        self.set_criterion(lossweight)

    def set_optimizer(self, total_step, use_bert=True):
        if use_bert:
            self.optimizer = BertAdam(params=self.parameters(),
                                      lr=self.config['lr'],
                                      warmup=0.1,
                                      t_total=total_step)
        else:
            self.optimizer = Adam(self.parameters(), lr=self.config['lr'])

    def set_criterion(self, weight):
        self.event_criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(weight))
        if self.config['model'] == 'rmtpp':
            self.intensity_w = nn.Parameter(torch.tensor(0.1, dtype=torch.float, device='cuda'))
            self.intensity_b = nn.Parameter(torch.tensor(0.1, dtype=torch.float, device='cuda'))
            self.time_criterion = self.RMTPPLoss
        else:
            self.time_criterion = nn.MSELoss()

    def RMTPPLoss(self, pred, gold):
        loss = torch.mean(pred + self.intensity_w * gold + self.intensity_b +
                          (torch.exp(pred + self.intensity_b) -
                           torch.exp(pred + self.intensity_w * gold + self.intensity_b)) / self.intensity_w)
        return -1 * loss

    def forward(self, input_time, input_events,
                ctg_event_tensor, num_event_tensor,
                ctg_ts_tensor, num_ts_tensor):

        # Event series features
        event_lstm_input = []

        if self.config['ev_ctg_features']:
            ctg_event_embeddings = [
                self.emb_drop(event_emb(ctg_event_tensor[..., i]))
                for i, event_emb in enumerate(self.event_embeddings)
            ]
            ctg_event_embeddings = torch.cat(ctg_event_embeddings, dim=-1).cuda()
            event_lstm_input.append(ctg_event_embeddings)

        if self.config['ev_num_features']:
            event_lstm_input.append(num_event_tensor.cuda())

        event_lstm_input.append(input_time.unsqueeze(-1))
        event_lstm_input = torch.cat(event_lstm_input, dim=-1)

        event_hidden_state, _ = self.event_lstm(event_lstm_input)

        ts_lstm_input = []

        if self.config['ts_ctg_features']:
            ctg_ts_embeddings = [
                ts_emb(ctg_ts_tensor[..., i])
                for i, ts_emb in enumerate(self.ts_embeddings)
            ]
            ctg_ts_embeddings = torch.cat(ctg_ts_embeddings, dim=-1).cuda()
            ts_lstm_input.append(ctg_ts_embeddings)

        if self.config['ts_num_features']:
            ts_lstm_input.append(num_ts_tensor.cuda())

        ts_lstm_input = torch.cat(ts_lstm_input, dim=-1)
        ts_hidden_state, _ = self.ts_lstm(ts_lstm_input)

        hidden_state = []

        if self.config['ev_ctg_features'] or self.config['ev_num_features']:
            hidden_state.append(event_hidden_state[:, -1])
        if self.config['ts_ctg_features'] or self.config['ts_num_features']:
            hidden_state.append(ts_hidden_state[:, -1])
        if len(hidden_state) == 2:
            hidden_state = torch.cat(hidden_state, dim=-1)

        # Mapping the above two features
        mlp_output = torch.tanh(self.mlp(hidden_state))
        mlp_output = self.mlp_drop(mlp_output)

        # Outputs
        event_logits = self.event_linear(mlp_output)
        time_logits = self.time_linear(mlp_output)
        return time_logits, event_logits

    def dispatch(self, tensors):
        for i in range(len(tensors)):
            tensors[i] = tensors[i].cuda().contiguous()
        return tensors

    def train_batch(self, batch):
        (time_tensor, event_tensor,
         ctg_event_tensor, num_event_tensor,
         ctg_ts_tensor, num_ts_tensor) = batch

        # print('Batch')
        # print(ctg_event_tensor.shape)
        # print(ctg_ts_tensor.shape)

        time_input, time_target = self.dispatch(
            [time_tensor[:, :-1], time_tensor[:, -1]])
        event_input, event_target = self.dispatch(
            [event_tensor[:, :-1], event_tensor[:, -1]])
        ctg_event_tensor = self.dispatch(ctg_event_tensor)
        num_event_tensor = self.dispatch(num_event_tensor)
        ctg_ts_tensor = self.dispatch(ctg_ts_tensor)
        num_ts_tensor = self.dispatch(num_ts_tensor)

        time_logits, event_logits = self.forward(
            time_input, event_input,
            ctg_event_tensor, num_event_tensor,
            ctg_ts_tensor, num_ts_tensor)

        loss1 = self.time_criterion(time_logits.view(-1), time_target.view(-1))
        loss2 = self.event_criterion(event_logits.view(-1, self.n_class), event_target.view(-1))
        loss = self.config['alpha'] * loss1 + loss2
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss1.item(), loss2.item(), loss.item()

    def predict(self, batch):
        time_tensor, event_tensor = batch
        time_input, time_target = self.dispatch([time_tensor[:, :-1], time_tensor[:, -1]])
        event_input, event_target = self.dispatch([event_tensor[:, :-1], event_tensor[:, -1]])
        time_logits, event_logits = self.forward(time_input, event_input)
        event_pred = np.argmax(event_logits.detach().cpu().numpy(), axis=-1)
        time_pred = time_logits.detach().cpu().numpy()
        return time_pred, event_pred
