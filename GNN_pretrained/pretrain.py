import os
import time
import numpy as np
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader as torch_DataLoader

from torch_geometric.loader import DataLoader as pyg_DataLoader
from transformers import AutoModel, AutoTokenizer

from PubChemSTM import ChemBI_HMDB_Datasets_Graph
from model.molecule_gnn.molecule_gnn_model import GNN, GNN_graphpred
from utils import prepare_text_tokens, get_molecule_repr, freeze_network

os.environ['TOKENIZERS_PARALLELISM'] = 'False'


def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr


def do_CL(X, Y, args):
    if args.normalize:
        X = F.normalize(X, dim=-1)
        Y = F.normalize(Y, dim=-1)

    if args.SSL_loss == 'EBM_NCE':
        criterion = nn.BCEWithLogitsLoss()
        neg_Y = torch.cat([Y[cycle_index(len(Y), i + 1)] for i in range(args.CL_neg_samples)], dim=0)
        neg_X = X.repeat((args.CL_neg_samples, 1))

        pred_pos = torch.sum(X * Y, dim=1) / args.T
        pred_neg = torch.sum(neg_X * neg_Y, dim=1) / args.T

        loss_pos = criterion(pred_pos, torch.ones(len(pred_pos)).to(pred_pos.device))
        loss_neg = criterion(pred_neg, torch.zeros(len(pred_neg)).to(pred_neg.device))
        CL_loss = (loss_pos + args.CL_neg_samples * loss_neg) / (1 + args.CL_neg_samples)

        CL_acc = (torch.sum(pred_pos > 0).float() + torch.sum(pred_neg < 0).float()) / \
                 (len(pred_pos) + len(pred_neg))
        CL_acc = CL_acc.detach().cpu().item()

    elif args.SSL_loss == 'InfoNCE':
        criterion = nn.CrossEntropyLoss()
        B = X.size()[0]
        logits = torch.mm(X, Y.transpose(1, 0))  # B*B
        logits = torch.div(logits, args.T)
        labels = torch.arange(B).long().to(logits.device)  # B*1

        CL_loss = criterion(logits, labels)
        pred = logits.argmax(dim=1, keepdim=False)
        CL_acc = pred.eq(labels).sum().detach().cpu().item() * 1. / B

    else:
        raise Exception

    return CL_loss, CL_acc


def save_model(save_best, epoch=None):
    if args.output_model_dir is not None:
        if save_best:
            global optimal_loss
            print("save model with loss: {:.5f}".format(optimal_loss))
            # model_file = "model.pth"
            model_file = "model_{}.pth".format(epoch)

        elif epoch is None:
            model_file = "model_final.pth"

        else:
            model_file = "model_{}.pth".format(epoch)

        saved_file_path = os.path.join(args.output_model_dir, "text_{}".format(model_file))
        torch.save(text_model.state_dict(), saved_file_path)

        saved_file_path = os.path.join(args.output_model_dir, "molecule_{}".format(model_file))
        torch.save(molecule_model.state_dict(), saved_file_path)

        saved_file_path = os.path.join(args.output_model_dir, "text2latent_{}".format(model_file))
        torch.save(text2latent.state_dict(), saved_file_path)

        saved_file_path = os.path.join(args.output_model_dir, "mol2latent_{}".format(model_file))
        torch.save(mol2latent.state_dict(), saved_file_path)
    return


def train(
        epoch,
        dataloader,
        text_model,
        text_tokenizer,
        molecule_model,
        MegaMolBART_wrapper=None):
    if args.representation_frozen:
        text_model.eval()
        molecule_model.eval()
    else:
        text_model.train()
        molecule_model.train()
    text2latent.train()
    mol2latent.train()

    if args.verbose:
        L = tqdm(dataloader)
    else:
        L = dataloader

    start_time = time.time()
    accum_loss, accum_acc = 0, 0
    for step, batch in enumerate(L):
        description = batch[0]
        molecule_data = batch[1]

        description_tokens_ids, description_masks = prepare_text_tokens(
            device=device,
            description=description,
            tokenizer=text_tokenizer,
            max_seq_len=args.max_seq_len)
        description_output = text_model(input_ids=description_tokens_ids, attention_mask=description_masks)
        description_repr = description_output["pooler_output"]
        description_repr = text2latent(description_repr)

        if molecule_type == "SMILES":
            molecule_data = list(molecule_data)  # for SMILES_list
            molecule_repr = get_molecule_repr(
                molecule_data, mol2latent=mol2latent,
                molecule_type=molecule_type, MegaMolBART_wrapper=MegaMolBART_wrapper)
        else:
            molecule_data = molecule_data.to(device)
            molecule_repr = get_molecule_repr(
                molecule_data, mol2latent=mol2latent,
                molecule_type=molecule_type, molecule_model=molecule_model)

        loss_01, acc_01 = do_CL(description_repr, molecule_repr, args)
        loss_02, acc_02 = do_CL(molecule_repr, description_repr, args)
        loss = (loss_01 + loss_02) / 2
        acc = (acc_01 + acc_02) / 2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accum_loss += loss.item()
        accum_acc += acc

    accum_loss /= len(L)
    accum_acc /= len(L)

    global optimal_loss
    temp_loss = accum_loss
    if temp_loss < optimal_loss:
        optimal_loss = temp_loss
        save_model(save_best=True, epoch=epoch)
    print("CL Loss: {:.5f}\tCL Acc: {:.5f}\tTime: {:.5f}".format(accum_loss, accum_acc, time.time() - start_time))
    with open('pretrain_result.log', 'a+') as f:
        f.write("CL Loss: {:.5f}\tCL Acc: {:.5f}\tTime: {:.5f}".format(accum_loss, accum_acc, time.time() - start_time))
        f.write('\n')
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=1)

    parser.add_argument("--dataspace_path", type=str, default="")
    parser.add_argument("--dataset", type=str, default="ChemBI_HMDB", choices=["PubChemSTM, ChemBI_HMDB"])
    parser.add_argument("--text_type", type=str, default="SciBERT", choices=["SciBERT"])
    parser.add_argument("--molecule_type", type=str, default="Graph", choices=["SMILES", "Graph"])
    parser.add_argument("--representation_frozen", dest='representation_frozen', action='store_true')
    parser.add_argument('--no_representation_frozen', dest='representation_frozen', action='store_false')
    parser.set_defaults(representation_frozen=False)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--text_lr", type=float, default=1e-4)
    parser.add_argument("--mol_lr", type=float, default=1e-5)
    parser.add_argument("--text_lr_scale", type=float, default=1)
    parser.add_argument("--mol_lr_scale", type=float, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--decay", type=float, default=0)

    parser.add_argument("--verbose", type=bool, default=True)
    # parser.add_argument('--verbose', dest='verbose', action='store_true')
    # parser.set_defaults(verbose=False)

    parser.add_argument("--output_model_dir", type=str, default='output_model')

    # for SciBERT #
    parser.add_argument("--max_seq_len", type=int, default=512)

    # for 2D GNN #
    parser.add_argument("--pretrain_gnn_mode", type=str, default="GraphMVP_G", choices=["GraphMVP_G"])
    parser.add_argument("--gnn_emb_dim", type=int, default=300)
    parser.add_argument("--num_layer", type=int, default=5)
    parser.add_argument('--JK', type=str, default='last')
    parser.add_argument("--dropout_ratio", type=float, default=0.5)
    parser.add_argument("--gnn_type", type=str, default="gin")
    parser.add_argument('--graph_pooling', type=str, default='mean')

    # for contrastive SSL #
    parser.add_argument("--SSL_loss", type=str, default="EBM_NCE", choices=["EBM_NCE", "InfoNCE"])
    parser.add_argument("--SSL_emb_dim", type=int, default=768)
    parser.add_argument("--CL_neg_samples", type=int, default=1)
    parser.add_argument("--T", type=float, default=0.1)
    parser.add_argument('--normalize', dest='normalize', action='store_true')
    parser.add_argument('--no_normalize', dest='normalize', action='store_false')
    parser.set_defaults(normalize=True)

    args = parser.parse_args()
    print("arguments\t", args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:" + str(args.device)) \
        if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if "PubChemSTM" in args.dataset:
        dataset_root = os.path.join(args.dataspace_path, "")
    elif "ChemBI_HMDB" in args.dataset:
        dataset_root = os.path.join(args.dataspace_path, "")
    else:
        raise Exception

    kwargs = {}

    # prepare text model #
    if args.text_type == "SciBERT":
        pretrained_SciBERT_folder = os.path.join(args.dataspace_path, 'pretrained_SciBERT')
        bert_name = 'pretrained_SciBERT/allenai/scibert_scivocab_uncased'

        text_tokenizer = AutoTokenizer.from_pretrained(bert_name, cache_dir=pretrained_SciBERT_folder)
        text_model = AutoModel.from_pretrained(bert_name, cache_dir=pretrained_SciBERT_folder).to(device)

        kwargs["text_tokenizer"] = text_tokenizer
        kwargs["text_model"] = text_model
        text_dim = 768
    else:
        raise Exception

    # prepare molecule model #
    molecule_type = args.molecule_type

    if molecule_type == "Graph":
        if args.dataset == "ChemBI_HMDB":
            dataset = ChemBI_HMDB_Datasets_Graph(dataset_root)

        dataloader_class = pyg_DataLoader
        molecule_node_model = GNN(
            num_layer=args.num_layer,
            emb_dim=args.gnn_emb_dim,
            JK=args.JK,
            drop_ratio=args.dropout_ratio,
            gnn_type=args.gnn_type)
        molecule_model = GNN_graphpred(
            num_layer=args.num_layer,
            emb_dim=args.gnn_emb_dim,
            JK=args.JK,
            graph_pooling=args.graph_pooling,
            num_tasks=1,
            molecule_node_model=molecule_node_model)
        pretrained_model_path = os.path.join(args.dataspace_path, "pretrained_GraphMVP", args.pretrain_gnn_mode,
                                             "model.pth")
        molecule_model.from_pretrained(pretrained_model_path)

        molecule_model = molecule_model.to(device)

        kwargs["molecule_model"] = molecule_model
        molecule_dim = args.gnn_emb_dim

    else:
        raise Exception
    dataloader = dataloader_class(dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers)

    text2latent = nn.Linear(text_dim, args.SSL_emb_dim).to(device)
    mol2latent = nn.Linear(molecule_dim, args.SSL_emb_dim).to(device)

    if args.representation_frozen:
        print("Representation is fronzen during pretraining.")
        freeze_network(text_model)
        freeze_network(molecule_model)
        model_param_group = [
            {"params": text2latent.parameters(), "lr": args.text_lr * args.text_lr_scale},
            {"params": mol2latent.parameters(), "lr": args.mol_lr * args.mol_lr_scale},
        ]
    else:
        model_param_group = [
            {"params": text_model.parameters(), "lr": args.text_lr},
            {"params": molecule_model.parameters(), "lr": args.mol_lr},
            {"params": text2latent.parameters(), "lr": args.text_lr * args.text_lr_scale},
            {"params": mol2latent.parameters(), "lr": args.mol_lr * args.mol_lr_scale},
        ]
    print('molecule_model_params', sum(p.numel() for p in molecule_model.parameters() if p.requires_grad))
    print('text_model_params', sum(p.numel() for p in text_model.parameters() if p.requires_grad))
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)

    optimizer = optim.Adam(model_param_group, weight_decay=args.decay)
    optimal_loss = 1e10

    for e in range(1, args.epochs + 1):
        print("Epoch {}".format(e))
        train(e, dataloader, **kwargs)

    save_model(save_best=False)
