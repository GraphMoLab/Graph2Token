import torch
from pytorch_lightning import LightningDataModule
from torch_geometric.data import Data
from torch.utils.data import DataLoader, Dataset
from torch_geometric.loader.dataloader import Collater
from dataprocess.smiles2graph_regression import smiles2graph


class TrainCollater:
    def __init__(self, tokenizer, text_max_len, mol_token_id, reg_token_id):
        self.text_max_len = text_max_len
        self.tokenizer = tokenizer
        self.collater = Collater([], [])
        self.mol_token_id = mol_token_id
        self.reg_token_id = reg_token_id

    def __call__(self, batch):
        graph, instruction, text_values = zip(*batch)
        graphs = self.collater(graph)

        self.tokenizer.padding_side = 'right'
        self.tokenizer.truncation_side = 'left'
        instruction_tokens = self.tokenizer(text=instruction,
                                            truncation=True,
                                            max_length=self.text_max_len,
                                            padding='longest',
                                            add_special_tokens=True,
                                            return_tensors='pt',
                                            return_attention_mask=True)

        is_mol_token = instruction_tokens.input_ids == self.mol_token_id
        instruction_tokens['is_mol_token'] = is_mol_token
        is_reg_token = instruction_tokens.input_ids == self.reg_token_id
        instruction_tokens['is_reg_token'] = is_reg_token

        text_values = torch.tensor(text_values).to(torch.int64)
        return graphs, instruction_tokens, text_values


class InferenceCollater:
    def __init__(self, tokenizer, text_max_len, mol_token_id, reg_token_id):
        self.text_max_len = text_max_len
        self.tokenizer = tokenizer
        self.collater = Collater([], [])
        self.mol_token_id = mol_token_id
        self.reg_token_id = reg_token_id

    def __call__(self, batch):
        graph, instruction, text_values = zip(*batch)
        graphs = self.collater(graph)
        # deal with prompt
        self.tokenizer.padding_side = 'right'
        self.tokenizer.truncation_side = 'left'
        instruction_tokens = self.tokenizer(instruction,
                                            return_tensors='pt',
                                            max_length=self.text_max_len,
                                            add_special_tokens=True,
                                            padding='longest',
                                            truncation=True,
                                            return_attention_mask=True)

        is_mol_token = instruction_tokens.input_ids == self.mol_token_id
        instruction_tokens['is_mol_token'] = is_mol_token
        is_reg_token = instruction_tokens.input_ids == self.reg_token_id
        instruction_tokens['is_reg_token'] = is_reg_token

        text_values = torch.tensor(text_values).to(torch.int64)
        return graphs, instruction_tokens, text_values


def smiles2data(smiles):
    graph = smiles2graph(smiles)
    x = torch.from_numpy(graph['node_feat'])
    edge_index = torch.from_numpy(graph['edge_index'], )
    edge_attr = torch.from_numpy(graph['edge_feat'])
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data


class CheBIDataset(Dataset):
    def __init__(self, path, text_max_len):
        self.path = path
        self.text_max_len = text_max_len
        self.mol_g = '<|mol_g|>'
        self.reg_g = '<|reg_g|>'
        self.prompt_1 = 'This molecule with IUPAC name is '
        self.prompt_2 = 'And given the graph representation ' + self.mol_g + '. '
        self.prompt_3 = '\n Please take into account the IUPAC name and its graph representation' \
                        ' to analysis this molecule, and then generate the predictions ' + \
                        self.reg_g + '.'

        with open(self.path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines][1:]

        self.smiles_list = []
        self.instruction_list = []
        self.text_list = []
        for line in lines:
            argv = line.split('\t')
            smiles, instruction, text, iupac = argv[0], argv[1], argv[2], argv[-2],
            # smiles, instruction, text, _, _, _, _, iupac, _ = line.split('\t')
            # instruction, smiles, text, iupac = line.split('\t')  # qm9
            self.smiles_list.append(smiles)
            self.instruction_list.append(
                self.prompt_1 + iupac + '. ' + self.prompt_2 + instruction + '. ' + self.prompt_3)
            # self.instruction_list.append(
            #     self.prompt_2 + ' ' + self.prompt_3)

            # self.instruction_list.append(self.prompt + iupac + '.' + ' ' + instruction + '<|eot_id|><|mol_g|>')
            # self.instruction_list.append(instruction + '<|eot_id|><|mol_g|>')
            # self.instruction_list.append(self.prompt + iupac + '.' + '<|eot_id|><|mol_g|>')
            # self.instruction_list.append('<|mol_g|>')
            self.text_list.append(text)

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, index):
        smiles = self.smiles_list[index]
        instruction = self.instruction_list[index]
        if self.text_list[index] == 'Yes':
            text_values = 1
        if self.text_list[index] == 'No':
            text_values = 0
        graph = smiles2data(smiles)
        return graph, instruction, text_values


class ProcessDatasets(LightningDataModule):
    def __init__(
            self,
            mode: str = 'pretrain',
            num_workers: int = 0,
            batch_size: int = 256,
            root: str = 'data/',
            text_max_len: int = 128,
            tokenizer=None,
            args=None,
    ):
        super().__init__()
        self.args = args
        self.mode = mode
        self.batch_size = batch_size
        self.inference_batch_size = args.inference_batch_size
        self.num_workers = num_workers
        self.text_max_len = text_max_len
        self.train_dataset = CheBIDataset(root + f'/train.txt', text_max_len)
        self.val_dataset = CheBIDataset(root + '/valid.txt', text_max_len)
        self.test_dataset = CheBIDataset(root + '/test.txt', text_max_len)
        self.init_tokenizer(tokenizer)

    def init_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        self.train_dataset.tokenizer = tokenizer
        self.val_dataset.tokenizer = tokenizer
        self.test_dataset.tokenizer = tokenizer
        self.mol_token_id = self.tokenizer.mol_token_id
        self.reg_token_id = self.tokenizer.reg_token_id

    def train_dataloader(self):
        # assert self.mode == 'pretrain'
        # assert self.mode == 'ft'
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            persistent_workers=True,
            collate_fn=TrainCollater(self.tokenizer, self.text_max_len, self.mol_token_id, self.reg_token_id),
        )
        return loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=TrainCollater(self.tokenizer, self.text_max_len, self.mol_token_id, self.reg_token_id),
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=InferenceCollater(self.tokenizer, self.text_max_len, self.mol_token_id, self.reg_token_id),
        )

        return [val_loader, test_loader]

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=InferenceCollater(self.tokenizer, self.text_max_len, self.mol_token_id, self.reg_token_id),
        )
        return loader

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=2)
        parser.add_argument('--batch_size', type=int, default=16)
        parser.add_argument('--inference_batch_size', type=int, default=8)
        parser.add_argument('--root', type=str, default='data/finetune/construct_cls/bace_bbbp_hiv')
        # parser.add_argument('--root', type=str, default='data')
        parser.add_argument('--text_max_len', type=int, default=256)

        return parent_parser



