import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data
from tqdm import tqdm

from model.molecule_gnn.features import atom_to_feature_vector, bond_to_feature_vector


def smi_to_graph_data_obj_simple(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)

        # atoms
        atom_features_list = []
        for atom in mol.GetAtoms():
            atom_feature = atom_to_feature_vector(atom)
            atom_features_list.append(atom_feature)
        x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

        # bonds
        if len(mol.GetBonds()) <= 0:  # mol has no bonds
            num_bond_features = 3  # bond type & direction
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)
        else:  # mol has bonds
            edges_list = []
            edge_features_list = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                edge_feature = bond_to_feature_vector(bond)

                edges_list.append((i, j))
                edge_features_list.append(edge_feature)
                edges_list.append((j, i))
                edge_features_list.append(edge_feature)

            # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
            edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

            # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
            edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        return data
    except:
        print(smiles)
        return None


if __name__ == '__main__':
    df = pd.read_csv('merged_chebi_hmdb.csv', sep='\t')

    df['text'] = df[['chebi_description', 'hmdb_description']].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)

    df = df[['CID', 'smiles', 'text']]
    for smi in tqdm(df['smiles'], total=len(df)):
        graph = smi_to_graph_data_obj_simple(smi)
        if graph is None:
            df = df[df['smiles'] != smi].dropna()
            continue
    print(len(df))

    df.to_csv('graph2text.csv', index=False, sep='\t')
    print("save successful：graph2text.csv")


