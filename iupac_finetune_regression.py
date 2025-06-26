import os
import torch
import argparse
import warnings
import pytorch_lightning as pl
from pytorch_lightning import Trainer, strategies
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import CSVLogger

from dataprocess.preprocess_dm_regression_llama3_iupac import ProcessDatasets
from model.iupac_prompt.QA_Trainer_iupac import QA_Trainer

from utils import MyDeepSpeedStrategy

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'False'
# for pyg bug
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')


def main(args):
    pl.seed_everything(args.seed)
    # model
    if args.init_checkpoint:
        model = QA_Trainer.load_from_checkpoint(args.init_checkpoint, strict=False, args=args)
        print(f"loaded init checkpoint from {args.init_checkpoint}")
    elif args.stage2_path:
        model = QA_Trainer(args)
        ckpt = torch.load(args.stage2_path, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'], strict=False)
        print(f"loaded stage2 model from {args.stage2_path}")
    else:
        model = QA_Trainer(args)

    print('total_trainable params', sum(p.numel() for p in model.parameters()))
    print('total_trainable params', sum(p.numel() for p in model.parameters() if p.requires_grad))
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    tokenizer = model.blip2opt.llm_tokenizer

    dm = ProcessDatasets(args.mode,
                         args.num_workers,
                         args.batch_size,
                         args.root,
                         args.text_max_len,
                         tokenizer,
                         args)
    callbacks = [plc.ModelCheckpoint(dirpath="all_checkpoints/" + args.filename + "/",
                                     filename='{epoch:02d}',
                                     every_n_epochs=args.save_every_n_epochs,
                                     save_last=True,
                                     save_top_k=-1,
                                     save_on_train_epoch_end=True)]
    # fixme save only used parameters
    if len(args.devices.split(',')) > 1:
        if args.strategy_name == 'deepspeed':
            strategy = MyDeepSpeedStrategy(stage=2)
        else:
            strategy = strategies.DDPStrategy(start_method='spawn')
    else:
        strategy = 'auto'
        args.devices = eval(args.devices)
    logger = CSVLogger(save_dir=f'./all_checkpoints/{args.filename}/')

    trainer = Trainer(accelerator=args.accelerator,
                      devices=args.devices,
                      precision=args.precision,
                      max_epochs=args.max_epochs,
                      accumulate_grad_batches=args.accumulate_grad_batches,
                      check_val_every_n_epoch=args.check_val_every_n_epoch,
                      callbacks=callbacks,
                      strategy=strategy,
                      logger=logger,
                      # num_sanity_val_steps=-1,
                      )
    if args.mode in {'pretrain', 'ft'}:
        trainer.fit(model, datamodule=dm)
        trainer.test(model, datamodule=dm)
    elif args.mode == 'eval':
        # trainer.fit_loop.epoch_progress.current.completed = args.caption_eval_epoch - 1
        trainer.test(model, datamodule=dm)
    else:
        raise NotImplementedError()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default="debug")
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--mode', type=str, default='ft')
    parser.add_argument('--strategy_name', type=str, default='deepspeed')
    parser = QA_Trainer.add_model_specific_args(parser)  # add model args
    parser = ProcessDatasets.add_model_specific_args(parser)  # add data args
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=str, default='0,1,2,3')
    parser.add_argument('--precision', type=str, default='bf16')
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--accumulate_grad_batches', type=int, default=2)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--enable_flash', action='store_true', default=False)
    args = parser.parse_args()

    print("=========================================")
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)
    print("=========================================")
    return args


if __name__ == '__main__':
    main(get_args())

