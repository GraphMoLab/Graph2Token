import os
import json
import pandas as pd
from random import sample
from rdkit import Chem
from tqdm import tqdm


def match_smiles_annotations(SDF_file, annotation_file, output_path):
    df_annotation = pd.read_csv(annotation_file, delimiter='\t')
    df_annotation_cid = df_annotation['CID'].tolist()

    suppl = Chem.SDMolSupplier(SDF_file)
    CID_list, SMILES_list = [], []
    for idx, mol in tqdm(enumerate(suppl), total=len(suppl)):
        if mol is None:
            continue
        CID = mol.GetProp("PUBCHEM_COMPOUND_CID")
        CAN_SMILES = mol.GetProp("PUBCHEM_OPENEYE_CAN_SMILES")
        ISO_SMILES = mol.GetProp("PUBCHEM_OPENEYE_ISO_SMILES")

        RDKit_mol = Chem.MolFromSmiles(CAN_SMILES)
        if RDKit_mol is None:
            continue
        RDKit_CAN_SMILES = Chem.MolToSmiles(RDKit_mol)

        try:
            if int(CID) not in df_annotation_cid:
                print(CID, df_annotation_cid[idx])
                print(idx)

            assert int(CID) in df_annotation_cid
        except:
            continue
            # break

        CID_list.append(int(CID))
        SMILES_list.append(RDKit_CAN_SMILES)

    df_cid_to_smiles = pd.DataFrame({"CID": CID_list, "SMILES": SMILES_list})

    selected_columns = df_annotation[['CID', 'name', 'description']]
    rows_with_selected_columns = selected_columns[selected_columns.isna().any(axis=1)]
    print(rows_with_selected_columns)

    result = pd.merge(
        selected_columns,
        df_cid_to_smiles,
        on='CID',
        how='left'
    )

    rows_with_nulls = result[result.isna().any(axis=1)]
    print(rows_with_nulls)

    result = result.dropna()

    print(len(result))
    result.to_csv(output_path, index=False, sep='\t')


if __name__ == "__main__":

    match_smiles_annotations('chebi_sdfs.sdf',
                             'chebi_output.csv',
                             'chebi_smiles_description.csv')

    match_smiles_annotations('hmdb_sdfs.sdf',
                             'hmdb_annotation.csv',
                             'hmdb_smiles_description.csv')

'''
    # SDF_file = '/home/wrz/Graph2Token/chebi_sdfs.sdf'
    SDF_file = '/home/wrz/Graph2Token/HMDB_sdfs.sdf'
    # annotation_file = '/home/wrz/Graph2Token/chebi_output.csv'
    annotation_file = '/home/wrz/Graph2Token/HMDB_annotations.csv'

    df_annotation = pd.read_csv(annotation_file, delimiter='\t')
    df_annotation_cid = df_annotation['CID'].tolist()

    description_list = []
    suppl = Chem.SDMolSupplier(SDF_file)
    CID_list, SMILES_list = [], []
    for idx, mol in tqdm(enumerate(suppl), total=len(suppl)):
        if mol is None:
            continue
        CID = mol.GetProp("PUBCHEM_COMPOUND_CID")
        CAN_SMILES = mol.GetProp("PUBCHEM_OPENEYE_CAN_SMILES")
        ISO_SMILES = mol.GetProp("PUBCHEM_OPENEYE_ISO_SMILES")

        RDKit_mol = Chem.MolFromSmiles(CAN_SMILES)
        if RDKit_mol is None:
            continue
        RDKit_CAN_SMILES = Chem.MolToSmiles(RDKit_mol)

        try:
            if int(CID) not in df_annotation_cid:
                print(CID, df_annotation_cid[idx])
                print(idx)
            # if int(CID) != df_annotation_cid[idx]:
            #     print(CID, df_annotation_cid[idx])

            assert int(CID) in df_annotation_cid
            # assert int(CID) == df_annotation_cid[idx]
        except:
            continue
            # break

        CID_list.append(int(CID))
        SMILES_list.append(RDKit_CAN_SMILES)
        # description_list.append(df_annotation['description'][idx])
        # if chebi
        # description_list.append(df_annotation['annotation'].tolist()[idx])

    df_cid_to_smiles = pd.DataFrame({"CID": CID_list, "SMILES": SMILES_list})

    selected_columns = df_annotation[['CID', 'name', 'description']]
    rows_with_selected_columns = selected_columns[selected_columns.isna().any(axis=1)]
    print(rows_with_selected_columns)

    result = pd.merge(
        selected_columns,
        df_cid_to_smiles,  # 只选择需要的列
        on='CID',
        how='left'
    )

    rows_with_nulls = result[result.isna().any(axis=1)]
    print(rows_with_nulls)

    result = result.dropna()

    print(len(result))
    result.to_csv('HMDB_smiles_description.csv', index=False, sep='\t')
    # result.to_csv('chebi_smiles_description.csv', index=False, sep='\t')
'''
