import pandas as pd
import requests


def get_PUG_REST_POST_body(*cids):
    """Generate the POST request body for PubChem PUG REST API.

    Args:
        *cids: Variable-length list of PubChem Compound IDs (CIDs).

    Returns:
        str: Formatted string for POST data (e.g., 'cid=1,2,3').
    """
    s = 'cid='
    for i in range(len(cids)):
        s += f"{cids[i]},"
    return s.rstrip(',')


def prepare_SDFs(cids, file='SDFs.sdf', batch_size=1000):
    """Obtain PubChem data in SDF format using PUG REST interface via POST requests and save it to the given file.
    Retrieve batch_size of SDFs every POST request.

    Args:
        cids: PubChem Compound IDs (CIDs) as variable arguments.
        file: Output file path (default: 'data/SDFs.sdf').
        batch_size: Number of CIDs per request (default: 1000).
    """
    for i in range(len(cids) // batch_size):
        r = requests.post(
            'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/SDF',
            data=get_PUG_REST_POST_body(*cids[i * batch_size:(i + 1) * batch_size]),
            headers={'Content-Type': 'application/x-www-form-urlencoded'}
        )
        with open(file, 'a') as f:
            f.write(r.text)
        print(f"Batch {i} was saved successfully!")

    if len(cids) % batch_size:
        r = requests.post(
            'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/SDF',
            data=get_PUG_REST_POST_body(*cids[len(cids) // batch_size * batch_size:]),
            headers={'Content-Type': 'application/x-www-form-urlencoded'}
        )
        with open(file, 'a') as f:
            f.write(r.text)
        print("The last batch was saved successfully!")


if __name__ == '__main__':

    # from chebi_output_path
    df_chebi = pd.read_csv('chebi_output.csv', delimiter='\t')
    chebi_cids = df_chebi['CID'].tolist()
    prepare_SDFs(chebi_cids, 'chebi_sdfs.sdf')

    # from hmdb_annotation_path
    df_hmdb = pd.read_csv('hmdb_annotation.csv', delimiter='\t')
    hmdb_cids = df_hmdb['CID'].tolist()
    prepare_SDFs(hmdb_cids, 'HMDB_sdfs.sdf')

