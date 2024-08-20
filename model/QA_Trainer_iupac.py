import os
from typing import Any, Dict

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

from model.QA_llama3_iupac import Align2llama
import pytorch_lightning as pl
from torch import optim
from lavis.common.optims import LinearWarmupCosineLRScheduler, LinearWarmupStepLRScheduler
import torch.distributed as dist
from utils import AttrDict


def load_ignore_unexpected(model, state_dict):
    keys = set(model.state_dict().keys())
    state_dict = {k: v for k, v in state_dict.items() if k in keys}

    # try to print keys that are not included
    model.load_state_dict(state_dict, strict=True)


def get_module_state_dict(state_dict, module_name):
    module_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(module_name):
            key = key[len(module_name) + 1:]
            if key == '':
                return value
            module_state_dict[key] = value
    return module_state_dict


class QA_Trainer(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        if isinstance(args, dict):
            args = AttrDict(**args)

        self.args = args
        self.caption_eval_epoch = args.caption_eval_epoch
        self.num_beams = args.num_beams
        self.llm_tune = args.llm_tune
        self.blip2opt = Align2llama(args.gin_num_layers,
                                    args.gin_hidden_dim,
                                    args.drop_ratio,
                                    args.tune_gnn,
                                    args.llm_tune,
                                    args.opt_model,
                                    args
                                    )

        self.save_hyperparameters(args)
        self.test_step_outputs = []
        self.list_targets = None
        self.list_predictions = None

    def configure_optimizers(self):
        self.trainer.fit_loop.setup_data()
        warmup_steps = min(len(self.trainer.train_dataloader), self.args.warmup_steps)

        optimizer = optim.AdamW(self.parameters(),
                                lr=self.args.init_lr,
                                weight_decay=self.args.weight_decay
                                )
        if self.args.scheduler == 'linear_warmup_cosine_lr':
            self.scheduler = LinearWarmupCosineLRScheduler(optimizer,
                                                           self.args.max_epochs,
                                                           self.args.min_lr,
                                                           self.args.init_lr,
                                                           warmup_steps,
                                                           self.args.warmup_lr)
        elif self.args.scheduler == 'linear_warmup_step_lr':
            self.scheduler = LinearWarmupStepLRScheduler(optimizer,
                                                         self.args.max_epochs,
                                                         self.args.min_lr,
                                                         self.args.init_lr,
                                                         self.args.lr_decay_rate,
                                                         self.args.warmup_lr,
                                                         warmup_steps)
        elif self.args.scheduler == 'None':
            self.scheduler = None
        else:
            raise NotImplementedError()
        return optimizer

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint.pop('optimizer_states')
        to_be_removed = []
        for key, value in checkpoint['state_dict'].items():
            try:
                if not self.get_parameter(key).requires_grad:
                    to_be_removed.append(key)
            except AttributeError:
                # to_be_removed.append(key)
                pass
        for key in to_be_removed:
            checkpoint['state_dict'].pop(key)

    @property
    def max_epochs(self):
        return self.fit_loop.max_epochs

    def save_predictions(self, mae):
        # assert len(predictions) == len(targets)
        # assert predictions.shape[0] == targets.shape[0]
        file_name = f"mae_{mae}.txt"
        with open(os.path.join(self.logger.log_dir, file_name), 'w', encoding='utf8') as f:
            pass
            # for p, t in zip(predictions, targets):
            #     line = {'prediction': p, 'target': t}
            #     f.write(json.dumps(line, ensure_ascii=True) + '\n')

    def save_predictions_rmse(self, rmse):
        # assert len(predictions) == len(targets)
        # assert predictions.shape[0] == targets.shape[0]
        file_name = f"rmse_{rmse}.txt"
        with open(os.path.join(self.logger.log_dir, file_name), 'w', encoding='utf8') as f:
            pass
            # for p, t in zip(predictions, targets):
            #     line = {'prediction': p, 'target': t}
            #     f.write(json.dumps(line, ensure_ascii=True) + '\n')

    def training_step(self, batch, batch_idx):
        if self.scheduler:
            self.scheduler.step(self.trainer.current_epoch, self.trainer.global_step)
        batch_size = batch[-1].size(0)
        # ============== Overall Loss ===================#
        loss = self.blip2opt(batch)
        self.log("molecule loss", float(loss['loss']), batch_size=batch_size, sync_dist=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], batch_size=batch_size, sync_dist=True)
        return loss['loss']

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx):

        if dataloader_idx == 0:
            batch_size = batch[-1].size(0)
            loss = self.blip2opt(batch)
            # ============== Overall Loss =================== #
            self.log("val molecule loss", float(loss['loss']), batch_size=batch_size, sync_dist=True)
            return loss['loss']

        elif dataloader_idx == 1:
            if (self.current_epoch + 1) % self.caption_eval_epoch != 0:
                return
            graphs, instruction_tokens, label_values = batch
            # ============== Captioning Results =================== #
            samples = {'graphs': graphs, 'instruction_tokens': instruction_tokens}
            predictions = self.blip2opt.generate(
                samples,
                num_beams=self.num_beams,
            )
            self.list_predictions.append(predictions)
            self.list_targets.append(label_values)
        else:
            raise NotImplementedError

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        graphs, instruction_tokens, label_values = batch
        # ============== Captioning Results =================== #
        samples = {'graphs': graphs, 'instruction_tokens': instruction_tokens}
        predictions = self.blip2opt.generate(
            samples,
            num_beams=self.num_beams,
        )
        self.test_step_outputs.append((predictions, label_values))
        return predictions, label_values

    def on_validation_epoch_start(self) -> None:
        self.list_predictions = []
        self.list_targets = []

    def on_validation_epoch_end(self) -> None:
        if (self.current_epoch + 1) % self.caption_eval_epoch != 0:
            return

        list_predictions = self.list_predictions
        list_targets = self.list_targets

        # predictions = []
        # for item in list_predictions:
        #     predictions.append(item.flatten())
        predictions = [tensor.float() for tensor in list_predictions]
        targets = [tensor.float() for tensor in list_targets]

        predictions = torch.cat(predictions, dim=0).cpu().numpy()
        targets = torch.cat(targets, dim=0).cpu().numpy()

        predictions = predictions.reshape(targets.shape)

        all_predictions = [None for _ in range(self.trainer.world_size)]
        all_targets = [None for _ in range(self.trainer.world_size)]
        try:
            dist.all_gather_object(all_predictions, predictions)
            dist.all_gather_object(all_targets, targets)
        except RuntimeError:
            all_predictions = [predictions]
            all_targets = [targets]

        if self.global_rank == 0:
            all_predictions = [i for ii in all_predictions for i in ii]
            all_targets = [i for ii in all_targets for i in ii]

            # rmse = mean_squared_error(np.array(all_targets), np.array(all_predictions), squared=False)

            # self.log("rmse", rmse, sync_dist=False)
            # self.save_predictions_rmse(all_predictions, all_targets, rmse)

            mae = mean_absolute_error(np.array(all_targets), np.array(all_predictions))
            # rmse = mean_squared_error(np.array(all_targets), np.array(all_predictions), squared=False)

            self.log("mae", mae, sync_dist=False)
            self.save_predictions(mae)
            # self.log("rmse", rmse, sync_dist=False)
            # self.save_predictions_rmse(rmse)

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        list_predictions, list_targets = zip(*outputs)

        predictions = [tensor.float() for tensor in list_predictions]
        targets = [tensor.float() for tensor in list_targets]

        predictions = torch.cat(predictions, dim=0).cpu().numpy()
        targets = torch.cat(targets, dim=0).cpu().numpy()

        all_predictions = [None for _ in range(self.trainer.world_size)]
        all_targets = [None for _ in range(self.trainer.world_size)]
        try:
            dist.all_gather_object(all_predictions, predictions)
            dist.all_gather_object(all_targets, targets)
        except RuntimeError:
            all_predictions = [predictions]
            all_targets = [targets]

        if self.global_rank == 0:
            all_predictions = [i for ii in all_predictions for i in ii]
            all_targets = [i for ii in all_targets for i in ii]

            # rmse = mean_squared_error(np.array(all_targets), np.array(all_predictions), squared=False)
            # self.log("rmse", rmse, sync_dist=False)
            # self.save_predictions_rmse(all_predictions, all_targets, rmse)

            mae = mean_absolute_error(np.array(all_targets), np.array(all_predictions))
            self.log("mae", mae, sync_dist=False)
            self.save_predictions(mae)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("GINSimclr")
        # train mode
        # GIN
        parser.add_argument('--gin_hidden_dim', type=int, default=300)
        parser.add_argument('--gin_num_layers', type=int, default=5)
        parser.add_argument('--drop_ratio', type=float, default=0.5)
        parser.add_argument('--tune_gnn', type=bool, default=False)

        # OPT
        parser.add_argument('--opt_model', type=str, default="LLama3-8B")
        parser.add_argument('--num_beams', type=int, default=5)
        parser.add_argument('--llm_tune', action='store_true', default=False)

        # optimization
        parser.add_argument('--weight_decay', type=float, default=0.05, help='optimizer weight decay')
        parser.add_argument('--init_lr', type=float, default=1e-4, help='optimizer init learning rate')
        parser.add_argument('--min_lr', type=float, default=1e-5, help='optimizer min learning rate')
        parser.add_argument('--warmup_lr', type=float, default=1e-6, help='optimizer warmup learning rate')
        parser.add_argument('--warmup_steps', type=int, default=1000, help='optimizer warmup steps')
        parser.add_argument('--lr_decay_rate', type=float, default=0.9, help='optimizer lr decay rate')
        parser.add_argument('--scheduler', type=str, default='linear_warmup_cosine_lr',
                            help='type of scheduler')  # or linear_warmup_step_lr
        parser.add_argument('--optimizer', type=str, default='adamw', help='type of scheduler')
        parser.add_argument('--finetune_path', type=str,
                            default='/home/wrz/LLM/llama3-8b')
        parser.add_argument('--stage2_path', type=str, default='all_checkpoints/synthesized_qm9_llama3_8B_final/lsat.ckpt')
        parser.add_argument('--init_checkpoint', type=str, default='')
        parser.add_argument('--caption_eval_epoch', type=int, default=1)
        parser.add_argument('--save_every_n_epochs', type=int, default=100)
        return parent_parser

