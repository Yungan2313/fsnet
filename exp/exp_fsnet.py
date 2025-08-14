from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from exp.exp_basic import Exp_Basic
from models.ts2vec.fsnet import TSEncoder, GlobalLocalMultiscaleTSEncoder
from models.ts2vec.losses import hierarchical_contrastive_loss
from tqdm import tqdm
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric, cumavg
import pdb
import numpy as np
from einops import rearrange
from collections import OrderedDict, defaultdict
import time
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader,ConcatDataset, Sampler
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, train_test_split

import os
import time
import random
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')

class StepByPredLenSampler(Sampler):
    """Single dataset: each batch is one index; indices step by pred_len."""
    def __init__(self, dataset_len, step):
        self.indices = list(range(0, dataset_len, int(step)))
    def __iter__(self):
        for i in self.indices:
            yield [i]                 # single-sample batch
    def __len__(self):
        return len(self.indices)
    
class StepByPredLenConcatSampler(Sampler):
    """ConcatDataset: step within each sub-dataset by pred_len."""
    def __init__(self, datasets, step):
        self.batches = []
        offset = 0
        step = int(step)
        for d in datasets:
            n = len(d)
            for i in range(0, n, step):
                self.batches.append([offset + i])  # single-sample batch
            offset += n
    def __iter__(self):
        return iter(self.batches)
    def __len__(self):
        return len(self.batches)
    
def _custom_dataset_test_bounds(df_len, seq_len, pred_len):
    """
    Dataset_Custom 的 test 邊界與 __len__ 對照。
    """
    num_train = int(df_len * 0.75)
    num_test  = int(df_len * 0.2)
    num_vali  = df_len - num_train - num_test
    border1s  = [0, num_train - seq_len, df_len - num_test - seq_len]
    border2s  = [num_train, num_train + num_vali, df_len]
    border1, border2 = border1s[2], border2s[2]      # test set
    usable = (border2 - border1) - seq_len - pred_len + 1
    return border1, border2, max(0, usable)

def _build_t0_list_for_custom(csv_path, date_from, date_to, seq_len, label_len, pred_len):
    """
    讀原始 CSV（不做標準化），用 Dataset_Custom 的切法，
    回傳 test 階段每個 sample 的目標窗起始日 t0（以日期日粒度對齊）。
    """
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    if date_from: df = df[df['date'] >= pd.to_datetime(date_from)]
    if date_to:   df = df[df['date'] <= pd.to_datetime(date_to)]

    border1, border2, usable = _custom_dataset_test_bounds(
        len(df), seq_len, pred_len
    )
    # test 區間的日期
    dates = df['date'].iloc[border1:border2].reset_index(drop=True)
    # 每個 sample 左端點 i 對應 t0 = r_begin 的日期
    t0_list = []
    for i in range(usable):
        r_begin = i + seq_len - label_len
        t0_list.append(dates.iloc[r_begin].normalize())  # 僅取日期(去時間)
    return t0_list

class TS2VecEncoderWrapper(nn.Module):
    def __init__(self, encoder, mask):
        super().__init__()
        self.encoder = encoder
        self.mask = mask

    def forward(self, input):
        return self.encoder(input, mask=self.mask)[:, -1]

class net(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device = device
        
        # for limiting output
        self.pct_limit = args.pct_limit          # e.g. 0.10
        self.limit_col = args.limit_col          # e.g. -1 (last column)
        self.pred_len = args.pred_len
        self.c_out    = args.c_out

        self.col_mean: torch.Tensor | None = None
        self.col_std:  torch.Tensor | None = None
        
        encoder = TSEncoder(input_dims=args.enc_in + 7,
                             output_dims=320,  # standard ts2vec backbone value
                             hidden_dims=64, # standard ts2vec backbone value
                             depth=10) 
        self.encoder = TS2VecEncoderWrapper(encoder, mask='all_true').to(self.device)
        self.dim = args.c_out * args.pred_len
        
        #self.regressor = nn.Sequential(nn.Linear(320, 320), nn.ReLU(), nn.Linear(320, self.dim)).to(self.device)
        self.regressor = nn.Linear(320, self.dim).to(self.device)
        
    def forward(self, feat_x, prev_price_flat = None, col_mean=None, col_std=None):
        """
        limit output:
        feat_x           : (B,  seq_len, D+7)  ← concat(x, x_mark)
        prev_price_flat  : (B,  pred_len*C)    ← 最後真值攤平
        """
        rep = self.encoder(feat_x)
        logits = self.regressor(rep)
        # print("DBG:", self.pct_limit, prev_price_flat is None)  # ← 只跑一次就知道
        if col_mean is None or col_std is None:
            col_mean = self.col_mean
            col_std  = self.col_std
        assert (col_mean is not None) and (col_std is not None), "need mean/std"
        
        # 轉成本裝置/型別的 tensor（讓後面可 broadcasting）
        if not torch.is_tensor(col_mean):
            col_mean = torch.as_tensor(float(col_mean), device=self.device, dtype=logits.dtype)
        if not torch.is_tensor(col_std):
            col_std  = torch.as_tensor(float(col_std),  device=self.device, dtype=logits.dtype)

        B, T, C = logits.size(0), self.pred_len, self.c_out
        col     = self.limit_col % C

        logits = logits.view(B, T, C)
        prev   = prev_price_flat.view(B, T, C)

        # 1) 反標準化目標欄位
        prev_real  = prev[:, :, col] * col_std + col_mean           # (B,T)
        logits_pct = torch.tanh(logits[:, :, col]) * self.pct_limit # (B,T) ∈ ±pct
        bound_real = prev_real * (1 + logits_pct)                   # 真實價格 ±10 %

        # 2) 再標準化回來，覆蓋到 price
        price_norm = prev.clone()                                   # (B,T,C)
        price_norm[:, :, col] = (bound_real - col_mean) / col_std

        return price_norm.view(B, -1)
        
    def store_grad(self):
        for name, layer in self.encoder.named_modules():    
            if 'PadConv' in type(layer).__name__:
                #print('{} - {}'.format(name, type(layer).__name__))
                layer.store_grad()
    
    def set_col_stats(self, mean, std):
        """Set or clear mean/std used by forward() for de-standardization in pct-limit mode."""
        if mean is None or std is None:
            self.col_mean = None
            self.col_std  = None
        else:
            self.col_mean = torch.as_tensor(float(mean), device=self.device, dtype=torch.float32)
            self.col_std  = torch.as_tensor(float(std),  device=self.device, dtype=torch.float32)    
        
class Exp_TS2VecSupervised(Exp_Basic):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.online = args.online_learning
        assert self.online in ['none', 'full', 'regressor']
        self.n_inner = args.n_inner
        self.opt_str = args.opt
        self.model = net(args, device = self.device)
        self.C   = self.args.c_out                      # 每筆資料的特徵數
        self.col = self.args.limit_col % self.C              # -1 → 最後一欄
        
        # load pretrained model if specified
        if getattr(self.args, 'pretrained', None):
            ckpt = self.args.pretrained
            print(f"[INFO] Loading full-model weights from: {ckpt}")
            state = torch.load(ckpt, map_location=self.device)
            self.model.load_state_dict(state, strict=True)
        
        if args.finetune:
            inp_var = 'univar' if args.features == 'S' else 'multivar'
            model_dir = str([path for path in Path(f'/export/home/TS_SSL/ts2vec/training/ts2vec/{args.data}/')
                .rglob(f'forecast_{inp_var}_*')][args.finetune_model_seed])
            state_dict = torch.load(os.path.join(model_dir, 'model.pkl'))
            for name in list(state_dict.keys()):
                if name != 'n_averaged':
                    state_dict[name[len('module.'):]] = state_dict[name]
                del state_dict[name]
            self.model[0].encoder.load_state_dict(state_dict)

    def _get_data(self, flag):
        args = self.args

        data_dict_ = {
            'ETTh1': Dataset_ETT_hour,
            'ETTh2': Dataset_ETT_hour,
            'ETTm1': Dataset_ETT_minute,
            'ETTm2': Dataset_ETT_minute,
            'WTH': Dataset_Custom,
            'ECL': Dataset_Custom,
            'Solar': Dataset_Custom,
            'custom': Dataset_Custom,
        }
        data_dict = defaultdict(lambda: Dataset_Custom, data_dict_)
        Data = data_dict[self.args.data]
        timeenc = 2

        if flag  == 'test':
            shuffle_flag = False;
            drop_last = False;
            batch_size = args.test_bsz;
            freq = args.freq
        elif flag == 'val':
            shuffle_flag = False;
            drop_last = False;
            batch_size = args.batch_size;
            freq = args.detail_freq
        elif flag == 'pred':
            shuffle_flag = False;
            drop_last = False;
            batch_size = 1;
            freq = args.detail_freq
            Data = Dataset_Pred
        else:
            shuffle_flag = True;
            drop_last = True;
            batch_size = args.batch_size;
            freq = args.freq
        
        #modify for multi-dataset
        if getattr(self.args, 'tickers', None) and flag in ['train', 'val', 'test']:
            # make every data a Dataset_Custom, then concat them
            single_sets = [
                Data(root_path=self.args.root_path,
                    data_path=f'{code}.csv',
                    flag=flag,
                    size=[args.seq_len, args.label_len, args.pred_len],
                    features=args.features,
                    target=args.target,
                    inverse=args.inverse,
                    timeenc=timeenc,
                    freq=freq,
                    cols=args.cols,
                    date_from=self.args.date_from,
                    date_to=self.args.date_to)
                for code in self.args.tickers
            ]
            data_set = ConcatDataset(single_sets)
            print(flag, len(data_set))
            print('ConcatDataset:', [len(s) for s in single_sets])
            batch_sampler = SameCompanyBatchSampler(single_sets,
                                            batch_size=batch_size,
                                            drop_last=True)
            data_loader = DataLoader(data_set,
                            batch_sampler=batch_sampler,
                            num_workers=args.num_workers)
        else:
            data_set = Data(
                root_path=args.root_path,
                data_path=args.data_path,
                flag=flag,
                size=[args.seq_len, args.label_len, args.pred_len],
                features=args.features,
                target=args.target,
                inverse=args.inverse,
                timeenc=timeenc,
                freq=freq,
                cols=args.cols,
                date_from=self.args.date_from,
                date_to=self.args.date_to
            )
            print(flag, len(data_set))
            data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last)

        # === NEW: custom test stride (overrides non_overlap_test) ===
        if flag == 'test':
            step = 0
            if getattr(self.args, 'test_stride', 0) and self.args.test_stride > 0:
                step = int(self.args.test_stride)
            elif getattr(self.args, 'non_overlap_test', False):
                step = int(self.args.pred_len)
            if step > 0:
                if isinstance(data_set, ConcatDataset):
                    batch_sampler = StepByPredLenConcatSampler(data_set.datasets, step=step)
                else:
                    batch_sampler = StepByPredLenSampler(len(data_set), step=step)
                data_loader = DataLoader(data_set, batch_sampler=batch_sampler,
                                         num_workers=self.args.num_workers)
        
        return data_set, data_loader

    def _select_optimizer(self):
        self.opt = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        return self.opt

    def _select_criterion(self):
        """
        · loss_mode == 'diff' → 先看 Δ%，大於門檻(10%) 的項目把 MSE 乘上 weight
        · 其他 loss_mode       → 普通 MSE
        """
        th = self.args.diff_threshold          
        w  = self.args.diff_weight

        def weighted_mse(pred_price, true_price, prev_price):
            """
            pred_flat, true_flat, prev_flat 全是 [B, T*C]
            prev_flat 只有在 diff 模式才會傳進來；否則是 None
            """
            if self.args.loss_mode == 'diff':
                diff_pred = pred_price - prev_price
                diff_true = true_price - prev_price
                pct       = diff_true / (prev_price.abs() + 1e-8)

                weight = torch.where(pct.abs() > th,
                                     torch.tensor(w, device=pred_price.device),
                                     torch.tensor(1.0, device=pred_price.device))
                return torch.mean(weight * (diff_pred - diff_true) ** 2)
            else:                              # direct
                return torch.mean((pred_price - true_price) ** 2)

        return weighted_mse

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        self.opt = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1

                self.opt.zero_grad()
                pred, true, prev = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                loss = criterion(pred, true, prev)
                train_loss.append(loss.item())
                
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(self.opt)
                    scaler.update()
                else:
                    loss.backward()
                    self.opt.step()
                self.model.store_grad()
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            #test_loss = self.vali(test_data, test_loader, criterion)
            test_loss = 0.

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(self.opt, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(vali_loader):
            pred, true, prev = self._process_one_batch(
                vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark, mode='vali')
            loss = criterion(pred.detach().cpu(), true.detach().cpu(), prev.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')

        # === 準備主資料的 t0_list（每一步的目標起始日） ===
        main_csv = os.path.join(self.args.root_path, self.args.data_path)
        main_t0_list = _build_t0_list_for_custom(
            csv_path=main_csv,
            date_from=self.args.date_from,
            date_to=self.args.date_to,
            seq_len=self.args.seq_len,
            label_len=self.args.label_len,
            pred_len=self.args.pred_len
        )

        # 對齊 main_t0：若使用 test_stride，採用相同步進；否則回退舊邏輯
        def get_main_t0(step_i:int):
            if getattr(self.args, 'test_stride', 0) and self.args.test_stride > 0:
                idx = step_i * self.args.test_stride
            elif getattr(self.args, 'non_overlap_test', False):
                idx = step_i * self.args.pred_len
            else:
                idx = step_i
            if idx >= len(main_t0_list):
                return None
            return main_t0_list[idx]

        # === 準備每個輔助股票：dataset 物件 + 對照表 {t0: index} ===
        aux_sets = []
        aux_maps = []
        if getattr(self.args, 'aux_tickers', None):
            data_dict_ = {
                'ETTh1': Dataset_ETT_hour,
                'ETTh2': Dataset_ETT_hour,
                'ETTm1': Dataset_ETT_minute,
                'ETTm2': Dataset_ETT_minute,
                'WTH': Dataset_Custom,
                'ECL': Dataset_Custom,
                'Solar': Dataset_Custom,
                'custom': Dataset_Custom,
            }
            Data = defaultdict(lambda: Dataset_Custom, data_dict_)[self.args.data]
            timeenc = 2
            freq = self.args.freq

            for code in self.args.aux_tickers:
                # 1) 建 dataset（和主資料同參數，但 flag='test'）
                ds = Data(
                    root_path=self.args.root_path,
                    data_path=f'{code}.csv',
                    flag='test',
                    size=[self.args.seq_len, self.args.label_len, self.args.pred_len],
                    features=self.args.features,
                    target=self.args.target,
                    inverse=self.args.inverse,
                    timeenc=timeenc,
                    freq=freq,
                    cols=self.args.cols,
                    date_from=self.args.date_from,
                    date_to=self.args.date_to
                )
                aux_sets.append(ds)
                # 2) 建該輔助資料的 t0_list，並反向成 {t0: index}
                aux_csv = os.path.join(self.args.root_path, f'{code}.csv')
                t0s = _build_t0_list_for_custom(
                    csv_path=aux_csv,
                    date_from=self.args.date_from,
                    date_to=self.args.date_to,
                    seq_len=self.args.seq_len,
                    label_len=self.args.label_len,
                    pred_len=self.args.pred_len
                )
                m = {t0: i for i, t0 in enumerate(t0s)}
                aux_maps.append(m)

            print(f"[AUX ALIGN] loaded {len(aux_sets)} aux sets; lr×{self.args.aux_lr_scale}, steps={self.args.aux_inner}")
        # === 封裝：針對某一步的 t0 跑一次 aux 更新（低 LR，不輸出） ===
        def run_aux_updates_for_t0(t0):
            if not aux_sets or self.online == 'none' or t0 is None:
                return
            # 暫時調低 LR
            orig_lrs = [pg['lr'] for pg in self.opt.param_groups]
            for pg in self.opt.param_groups:
                pg['lr'] = pg['lr'] * self.args.aux_lr_scale

            f_dim = -1 if self.args.features == 'MS' else 0
            criterion = self._select_criterion()

            prev_mean = getattr(self.model, 'col_mean', None)
            prev_std  = getattr(self.model, 'col_std',  None)
            
            for ds, m in zip(aux_sets, aux_maps):
                i_aux = m.get(t0, None)
                if i_aux is None:
                    continue

                # ---- 取 dataset 的 sample ----
                ax, ay, axm, aym = ds[i_aux]     # ax:(S,Dx)  ay:(L+T,Dy)  axm:(S,7)
                ax_t  = torch.as_tensor(ax,  dtype=torch.float32, device=self.device)
                axm_t = torch.as_tensor(axm, dtype=torch.float32, device=self.device)
                ay_t  = torch.as_tensor(ay,  dtype=torch.float32, device=self.device)

                # ---- 組 x / y 形狀 ----
                x_aux = torch.cat([ax_t, axm_t], dim=-1)    # (S, D+7)
                if x_aux.dim() == 2:                        # -> (1, S, D+7)
                    x_aux = x_aux.unsqueeze(0)
                if ay_t.dim() == 2:                         # -> (1, L+T, D)
                    ay_t = ay_t.unsqueeze(0)

                gt_aux   = ay_t[:, -self.args.pred_len:, f_dim:]    # (1, T, C_sel)
                true_aux = rearrange(gt_aux, 'b t d -> b (t d)')     # (1, T*C_sel)

                last_in  = ax_t[-1:, f_dim:]                         # (1, C_sel)
                last_in  = last_in.unsqueeze(0)                      # (1, 1, C_sel)
                prev_aux = last_in.expand_as(gt_aux)                 # (1, T, C_sel)
                prev_aux_flat = rearrange(prev_aux, 'b t d -> b (t d)')  # (1, T*C_sel)

                # ---- ★ 取 aux 的 mean/std 並設到 model ----
                # 取出該欄位在 ds.scaler 的 mean/std；支援 utils.tools.StandardScaler / sklearn 兩種命名
                scaler = getattr(ds, 'scaler', None)
                D_out  = ay_t.size(-1)              # y 的欄位數，與 scaler 對齊
                col_idx = self.args.limit_col % D_out

                def _fetch_stats(sc, idx):
                    if hasattr(sc, 'mean_'):  # sklearn
                        mean_arr = sc.mean_
                        if hasattr(sc, 'scale_'):
                            std_arr = sc.scale_
                        elif hasattr(sc, 'std_'):
                            std_arr = sc.std_
                        else:
                            raise AttributeError("scaler has mean_ but no scale_/std_")
                    else:                      # utils.tools.StandardScaler
                        mean_arr = sc.mean
                        std_arr  = sc.std
                    return float(mean_arr[idx]), float(std_arr[idx] + 1e-12)

                if scaler is None:
                    raise RuntimeError("aux dataset has no scaler; cannot set mean/std for pct_limit")

                col_mean, col_std = _fetch_stats(scaler, col_idx)
                self.model.set_col_stats(col_mean, col_std)

                # ---- 低 LR 的輔助更新 ----
                for _ in range(getattr(self.args, 'aux_inner', 1)):
                    out_aux  = self.model(x_aux, prev_aux_flat if self.args.pct_limit > 0 else None)
                    loss_aux = criterion(out_aux, true_aux, prev_aux_flat)
                    loss_aux.backward()
                    self.opt.step()
                    self.model.store_grad()
                    self.opt.zero_grad()

            # 還原 LR
            for pg, lr in zip(self.opt.param_groups, orig_lrs):
                pg['lr'] = lr
            # 還原主資料原本的 mean/std，避免影響下一步
            # 注意 prev_* 可能是 Tensor 或 None
            if prev_mean is None or prev_std is None:
                self.model.set_col_stats(None, None)
            else:
                self.model.col_mean = prev_mean
                self.model.col_std  = prev_std
        # === AUX END ===
        
        self.model.eval()
        if self.online == 'regressor':
            for p in self.model.encoder.parameters():
                p.requires_grad = False 
        elif self.online == 'none':
            for p in self.model.parameters():
                p.requires_grad = False
        
        preds = []
        trues = []
        start = time.time()
        maes,mses,rmses,mapes,mspes = [],[],[],[],[]
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(test_loader)):
            
            t0 = get_main_t0(i)
            # aux fist: 先跑輔助資料的更新
            if getattr(self.args, 'aux_update_order', 'after') == 'before':
                run_aux_updates_for_t0(t0)
            
            pred, true, _ = self._process_one_batch(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark, mode='test')
            preds.append(pred.detach().cpu())
            trues.append(true.detach().cpu())
            mae, mse, rmse, mape, mspe = metric(pred.detach().cpu().numpy(), true.detach().cpu().numpy())
            maes.append(mae)
            mses.append(mse)
            rmses.append(rmse)
            mapes.append(mape)
            mspes.append(mspe)
            
            if getattr(self.args, 'aux_update_order', 'after') == 'after':
                run_aux_updates_for_t0(t0)

        preds = torch.cat(preds, dim=0).numpy()
        trues = torch.cat(trues, dim=0).numpy()
        print('test shape:', preds.shape, trues.shape)
        
        MAE, MSE, RMSE, MAPE, MSPE = cumavg(maes), cumavg(mses), cumavg(rmses), cumavg(mapes), cumavg(mspes)
        mae, mse, rmse, mape, mspe = MAE[-1], MSE[-1], RMSE[-1], MAPE[-1], MSPE[-1]

       
        
        end = time.time()
        exp_time = end - start
        #mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, time:{}'.format(mse, mae, exp_time))
        return [mae, mse, rmse, mape, mspe, exp_time], MAE, MSE, preds, trues

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, mode='train'):
        if mode =='test' and self.online != 'none':
            return self._ol_one_batch(dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark)

        x = torch.cat([batch_x.float(), batch_x_mark.float()], dim=-1).to(self.device)
        batch_y = batch_y.float()
        # --------- 新增：先算 prev_flat 給裁剪用 ---------
        f_dim   = -1 if self.args.features == 'MS' else 0
        last_in = batch_x[:, -1:, f_dim:].to(self.device)               # (B,1,C)
        prev_flat = last_in.expand(-1, self.args.pred_len, -1)     # (B,T,C)
        prev_flat = rearrange(prev_flat, 'b t c -> b (t c)')        # (B,T*C)
        # for limiting output
        if isinstance(dataset_object, torch.utils.data.ConcatDataset):
            scaler = dataset_object.datasets[0].scaler   # SameCompanyBatchSampler ⇒ 同公司
        else:
            scaler = dataset_object.scaler
        mean_val = scaler.mean_[self.col]  if hasattr(scaler, 'mean_')  else scaler.mean[self.col]
        std_val  = scaler.scale_[self.col] if hasattr(scaler, 'scale_') else scaler.std[self.col]

        mean = torch.tensor(mean_val, device=self.device, dtype=torch.float32)
        std  = torch.tensor(std_val,  device=self.device, dtype=torch.float32)
        
        
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self.model(x,
                                    prev_flat if self.args.pct_limit > 0 else None,
                                    mean     if self.args.pct_limit > 0 else None,
                                    std      if self.args.pct_limit > 0 else None)
        else:
            outputs = self.model(x,
                                prev_flat if self.args.pct_limit > 0 else None,
                                mean     if self.args.pct_limit > 0 else None,
                                std      if self.args.pct_limit > 0 else None)
        f_dim = -1 if self.args.features=='MS' else 0
        # 真值價格  (B,T,C) → (B,T*C)
        gt_seq  = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
        true    = rearrange(gt_seq, 'b t d -> b (t d)')
        
        # new loss calculation
        # 前一價格  (最後一個輸入點) 同樣攤平
        last_in   = batch_x[:, -1:, f_dim:].to(self.device)      # (B,1,C)
        prev_flat = rearrange(last_in.expand_as(gt_seq), 'b t d -> b (t d)')
        
        return outputs, true, prev_flat
    
    def _ol_one_batch(self,dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        f_dim = -1 if self.args.features == 'MS' else 0
        true = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
        true = rearrange(true, 'b t d -> b (t d)').float()
        last_in = batch_x[:, -1:, f_dim:].to(self.device)              # (B,1,C)
        prev_flat = rearrange(last_in.expand_as(batch_y[:, -self.args.pred_len:, f_dim:]), 'b t d -> b (t d)')
        criterion = self._select_criterion()
        
        x = torch.cat([batch_x.float(), batch_x_mark.float()], dim=-1).to(self.device)
        batch_y = batch_y.float()
        
        # for limiting output
        if isinstance(dataset_object, torch.utils.data.ConcatDataset):
            scaler = dataset_object.datasets[0].scaler   # SameCompanyBatchSampler ⇒ 同公司
        else:
            scaler = dataset_object.scaler
        mean_val = scaler.mean_[self.col]  if hasattr(scaler, 'mean_')  else scaler.mean[self.col]
        std_val  = scaler.scale_[self.col] if hasattr(scaler, 'scale_') else scaler.std[self.col]

        mean = torch.tensor(mean_val, device=self.device, dtype=torch.float32)
        std  = torch.tensor(std_val,  device=self.device, dtype=torch.float32)
        
        for _ in range(self.n_inner):
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(x,
                                        prev_flat if self.args.pct_limit > 0 else None,
                                        mean     if self.args.pct_limit > 0 else None,
                                        std      if self.args.pct_limit > 0 else None)
            else:
                outputs = self.model(x,
                                        prev_flat if self.args.pct_limit > 0 else None,
                                        mean     if self.args.pct_limit > 0 else None,
                                        std      if self.args.pct_limit > 0 else None)
            loss = criterion(outputs, true, prev_flat)
            loss.backward()
            self.opt.step()       
            self.model.store_grad()
            self.opt.zero_grad()

        f_dim = -1 if self.args.features=='MS' else 0
        # 真值價格  (B,T,C) → (B,T*C)
        gt_seq  = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
        true    = rearrange(gt_seq, 'b t d -> b (t d)')
        
        # new loss calculation
        # 前一價格  (最後一個輸入點) 同樣攤平
        last_in   = batch_x[:, -1:, f_dim:].to(self.device)      # (B,1,C)
        prev_flat = rearrange(last_in.expand_as(gt_seq), 'b t d -> b (t d)')
        
        return outputs, true, prev_flat

# new class for multi-dataset
class SameCompanyBatchSampler(Sampler):
    """
    Build for ConcatDataset : Every batch contains data only from the same company.
    datasets = data which send to ConcatDataset(single_sets list)
    """
    def __init__(self, datasets, batch_size, drop_last=True):
        self.batch_size, self.drop_last = batch_size, drop_last
        self.buckets = []                         # [range(start, end), ...]
        offset = 0
        for d in datasets:
            self.buckets.append(range(offset, offset + len(d)))
            offset += len(d)

    def __iter__(self):
        comps = self.buckets.copy()
        random.shuffle(comps)                     # randomly shuffle the order of companies
        for idx_range in comps:
            pool = list(idx_range)
            random.shuffle(pool)                  # shuffle through the data of the same company
            for i in range(0, len(pool), self.batch_size):
                batch = pool[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    yield batch                   # give to DataLoader collate

    def __len__(self):
        return sum(len(r) // self.batch_size for r in self.buckets)

