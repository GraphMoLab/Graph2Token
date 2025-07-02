import re

import pandas as pd
import requests
from tqdm import tqdm
from collections import defaultdict
import json


def clean_up_description(description):
    description = description + " "

    ##### extra adj Pure #####
    if description.startswith("Pure "):
        description = description.replace("Pure ", "")
    ##### fix typo #####
    if description.startswith("Mercurycombines"):
        description = description.replace("Mercurycombines", "Mercury combines")

    name_special_case_list = [
        '17-Hydroxy-6-methylpregna-3,6-diene-3,20-dione. ',
        '5-Thymidylic acid. ',
        "5'-S-(3-Amino-3-carboxypropyl)-5'-thioadenosine. ",
        "Guanosine 5'-(trihydrogen diphosphate), monoanhydride with phosphorothioic acid. ",
        "5'-Uridylic acid. ",
        "5'-Adenylic acid, ",
        "Uridine 5'-(tetrahydrogen triphosphate). ",
        "Inosine 5'-Monophosphate. ",
        "Pivaloyloxymethyl butyrate (AN-9), ",
        "4-Amino-5-cyano-7-(D-ribofuranosyl)-7H- pyrrolo(2,3-d)pyrimidine. ",
        "Cardamonin (also known as Dihydroxymethoxychalcone), ",
    ]

    ##### a special case #####
    description = description.replace("17-Hydroxy-6-methylpregna-3,6-diene-3,20-dione. ",
                                      "17-Hydroxy-6-methylpregna-3,6-diene-3,20-dione is ")

    ##### a special case #####
    description = description.replace("5-Thymidylic acid. ", "5-Thymidylic acid. is ")

    ##### a special case #####
    description = description.replace("5'-S-(3-Amino-3-carboxypropyl)-5'-thioadenosine. ",
                                      "5'-S-(3-Amino-3-carboxypropyl)-5'-thioadenosine. is ")

    ##### a special case #####
    description = description.replace(
        "Guanosine 5'-(trihydrogen diphosphate), monoanhydride with phosphorothioic acid. ",
        "Guanosine 5'-(trihydrogen diphosphate), monoanhydride with phosphorothioic acid is ")

    ##### a special case #####
    description = description.replace("5'-Uridylic acid. ", "5'-Uridylic acid is ")

    ##### a special case #####
    description = description.replace("5'-Adenylic acid, ", "5'-Adenylic acid is ")

    ##### a special case #####
    description = description.replace("Uridine 5'-(tetrahydrogen triphosphate). ",
                                      "Uridine 5'-(tetrahydrogen triphosphate). is ")

    ##### a special case #####
    description = description.replace("Inosine 5'-Monophosphate. ", "Inosine 5'-Monophosphate. is ")

    ##### a special case #####
    description = description.replace("Pivaloyloxymethyl butyrate (AN-9), ", "Pivaloyloxymethyl butyrate (AN-9) is ")

    ##### a special case #####
    description = description.replace("4-Amino-5-cyano-7-(D-ribofuranosyl)-7H- pyrrolo(2,3-d)pyrimidine. ",
                                      "4-Amino-5-cyano-7-(D-ribofuranosyl)-7H- pyrrolo(2,3-d)pyrimidine is ")

    ##### a special case #####
    description = description.replace("Cardamonin (also known as Dihydroxymethoxychalcone), ",
                                      "Cardamonin (also known as Dihydroxymethoxychalcone) is ")

    ##### a special case #####
    description = description.replace("Lithium has been used to treat ", "Lithium is ")

    ##### a special case #####
    description = description.replace("4,4'-Methylenebis ", "4,4'-Methylenebis is ")

    ##### a special case #####
    description = description.replace("2,3,7,8-Tetrachlorodibenzo-p-dioxin", "2,3,7,8-Tetrachlorodibenzo-p-dioxin is ")

    ##### a special case #####
    description = description.replace("Exposure to 2,4,5-trichlorophenol ", "2,4,5-Trichlorophenol exposure ")

    index = 0
    L = len(description)
    if description.startswith('C.I. '):
        start_index = len('C.I. ')
    elif description.startswith('Nectriapyrone. D '):
        start_index = len('Nectriapyrone. D ')
    elif description.startswith('Salmonella enterica sv. Minnesota LPS core oligosaccharide'):
        start_index = len('Salmonella enterica sv. Minnesota LPS core oligosaccharide')
    else:
        start_index = 0
    for index in range(start_index, L - 1):
        if index < L - 2:
            if description[index] == '.' and description[index + 1] == ' ' and 'A' <= description[index + 2] <= 'Z':
                break
        elif index == L - 2:
            break

    first_sentence = description[:index + 1]
    return first_sentence


def detect_and_replace(sentence):
    target_words = {
        'is': 'This molecule',
        'was': 'This molecule',
        'appears': 'This molecule',
        'occurs': 'This molecule',
        'stands for': 'This molecule',
        'belongs to': 'This molecule',
        'exists': 'This molecule',
        'has been used in trials': 'This molecule',
        'has been investigated': 'This molecule',
        'has many uses': 'This molecule',
        'are': 'These molecules',
        'were': 'These molecules'
    }

    match = re.search(r'\b(is|was|appears|occurs|stands for|belongs to|exists|has been used in trials|has been investigated|has many uses|are|were)\b', sentence, flags=re.IGNORECASE)
    if not match:
        return None

    first_word = match.group(1).lower()
    replace_word = target_words.get(first_word, None)

    return replace_word


def extract_name(name_raw, description):
    first_sentence = clean_up_description(description)

    splitter = '  --  --  '

    replaced_words = detect_and_replace(first_sentence)

    if replaced_words is None:
        return None, None, None

    # if ' are ' in first_sentence or ' were ' in first_sentence:
    #     replaced_words = 'These molecules'
    # else:
    #     replaced_words = 'This molecule'

    first_sentence = first_sentence.replace(' is ', splitter)
    first_sentence = first_sentence.replace(' are ', splitter)
    first_sentence = first_sentence.replace(' was ', splitter)
    first_sentence = first_sentence.replace(' were ', splitter)
    first_sentence = first_sentence.replace(' appears ', splitter)
    first_sentence = first_sentence.replace(' occurs ', splitter)
    first_sentence = first_sentence.replace(' stands for ', splitter)
    first_sentence = first_sentence.replace(' belongs to ', splitter)
    first_sentence = first_sentence.replace(' exists ', splitter)  # only for CID=11443
    first_sentence = first_sentence.replace(' has been used in trials ', splitter)
    first_sentence = first_sentence.replace(' has been investigated ', splitter)
    first_sentence = first_sentence.replace(' has many uses ', splitter)

    if splitter in first_sentence:
        extracted_name = first_sentence.split(splitter, 1)[0]
    elif first_sentence.startswith(name_raw):
        extracted_name = name_raw
    elif name_raw in first_sentence:
        extracted_name = name_raw
        extracted_name = None
        print("=====", name_raw)
        print("first sentence: ", first_sentence)
        # print()
    else:
        extracted_name = None

    if extracted_name is not None:
        extracted_description = description.replace(extracted_name, replaced_words)
    else:
        extracted_description = description

    return extracted_name, extracted_description, first_sentence


def process_dataframe(df):
    processed_rows = []
    for index, row in tqdm(df.iterrows(), total=len(df)):
        name_raw = row['name'].strip()
        description = row['description'].strip()

        extracted_name, extracted_description, first_sentence = extract_name(name_raw, description)

        if extracted_name is None or extracted_description is None:
            continue

        assert extracted_description.startswith('Th')

        # Create a new row with the extracted information
        new_row = {
            'CID': row['CID'],
            'name': extracted_name,
            'smiles': row['SMILES'],
            'description': extracted_description,
        }
        processed_rows.append(new_row)

    # Create new dataframe from processed rows
    return pd.DataFrame(processed_rows)


if __name__ == "__main__":

    df_chebi = pd.read_csv('chebi_smiles_description.csv', delimiter='\t')
    df_hmdb = pd.read_csv('hmdb_smiles_description.csv', delimiter='\t')
    print(len(df_chebi), len(df_hmdb))

    new_df_chebi = process_dataframe(df_chebi)
    new_df_hmdb = process_dataframe(df_hmdb)

    new_df_chebi = new_df_chebi.rename(columns={'description': 'chebi_description'})
    new_df_hmdb = new_df_hmdb.rename(columns={'description': 'hmdb_description'})

    merged_df = pd.merge(
        new_df_chebi,
        new_df_hmdb,
        on=['CID', 'name', 'smiles'],
        how='outer'
    )

    print(f"Processed ChEBI entries: {len(new_df_chebi)}")
    print(f"Processed HMDB entries: {len(new_df_hmdb)}")
    print(f"Merged dataframe entries: {len(merged_df)}")

    merged_df.to_csv('merged_chebi_hmdb.csv', index=False, sep='\t')

