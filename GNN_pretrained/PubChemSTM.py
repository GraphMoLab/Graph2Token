import os
from itertools import repeat
import pandas as pd
import json
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset

from rdkit import Chem
from rdkit import RDLogger

from GNN_pretrained.process_hmdb_chebi_dataset import smi_to_graph_data_obj_simple

RDLogger.DisableLog('rdApp.*')

class ChemBI_HMDB_Datasets_Graph(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        # only for `process` function
        self.CID2text_file = os.path.join(self.root, "graph2text.csv")

        super(ChemBI_HMDB_Datasets_Graph, self).__init__(root, transform, pre_transform, pre_filter)

        self.load_Graph_CID_and_text()
        return

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def process(self):
        with open(self.CID2text_file, 'r') as f:
            reader = pd.read_csv(f, iterator=True, chunksize=1, delimiter='\t')

            CID_list, graph_list, text_list = [], [], []
            for index, chunk in enumerate(reader):
                for idx, row in chunk.iterrows():
                    CID = int(row['CID'])
                    smiles = row['smiles']
                    text = row['text']
                    graph = smi_to_graph_data_obj_simple(smiles)

                    CID_list.append(CID)
                    graph_list.append(graph)
                    text_list.append(text)

        if self.pre_filter is not None:
            graph_list = [graph for graph in graph_list if self.pre_filter(graph)]

        if self.pre_transform is not None:
            graph_list = [self.pre_transform(graph) for graph in graph_list]

        graphs, slices = self.collate(graph_list)
        torch.save((graphs, slices), self.processed_paths[0])
        return

    def load_Graph_CID_and_text(self):
        self.graphs, self.slices = torch.load(self.processed_paths[0])

        CID_text_df = pd.read_csv(self.CID2text_file, delimiter='\t')
        self.text_list = CID_text_df["text"].tolist()
        return

    def get(self, idx):
        text = self.text_list[idx]

        data = Data()
        for key in self.graphs.keys():
            item, slices = self.graphs[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]
        return text, data

    def __len__(self):
        return len(self.text_list)
