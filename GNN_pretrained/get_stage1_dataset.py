import csv
import requests
import xml.etree.ElementTree as ET


chebi_output_path = 'chebi_output.csv'
hmdb_output_path = 'hmdb_id_from_pubchem.txt'
hmdb_annotation_path = 'hmdb_annotation.csv'
# download the hmdb xml file path .xml
hmdb_xml_path = '../hmdb_metabolites.xml'


def get_PubChem_PUG_View_URL(annotation, record, out, options=None):
    """Generate a URL based on PubChem PUG-View URL syntax and return it as a string.
    A typical PUG-View request URL encodes three pieces of information.

    Arguments are
    * annotation: the type of annotations to retrieve.
    * record: the specification of the record of interest.
    * out: the desired output format.
    * options: additional information.
    """
    return f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/{annotation}/{record}/{out}?{options}" if options else f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/{annotation}/{record}/{out}"


CHEBI_annotation_URLs = [
    get_PubChem_PUG_View_URL(
        'annotations',
        'heading',
        'JSON',
        f"source=ChEBI&heading_type=Compound&heading=Record+Description&page={i + 1}"
    )
    for i in range(175)
]


def prepare_PubChem_annotations(records, *, file='../data/annotations.txt', mode='w', header=True, CID=False):
    """Extract PubChem annotations from given records and save them to the given file.
    The extraction structure is defined based on PubChemAnnotations_ChEBI_heading=Record Description.json downloaded from PubChem.
    The URL is https://pubchem.ncbi.nlm.nih.gov/source/ChEBI#data=Annotations
    """
    with open(file, mode) as f:
        if CID:
            dir = file.rsplit('.', 1)[0]
            with open(dir + '.log.txt', mode) as f1, open(dir + '__count.csv', mode) as f2:
                count_list = []
                if header:
                    f.write('CID\tname\tdescription\n')
                for record in records:
                    name = record['Name']
                    annotation = record['Data'][0]['Value']['StringWithMarkup'][0]['String']
                    if 'LinkedRecords' in record:
                        CIDs = record['LinkedRecords']['CID']
                        f.write(f"{CIDs[0]}\t{name}\t{annotation}\n") if len(CIDs) == 1 else f1.write(
                            f"{len(CIDs)} linked records are found when processing {annotation}\n")
                        if len(count_list) < len(CIDs) + 1:
                            for _ in range(len(CIDs) - len(count_list) + 1):
                                count_list.append(0)
                        count_list[len(CIDs)] += 1
                    else:
                        f1.write(f"No linked records are found when processing \"{annotation}\".\n")
                        if count_list:
                            count_list[0] += 1
                        else:
                            count_list.append(1)
                print(*count_list, sep=',', file=f2)
                f1.write('Done!\n')
        else:
            if header:
                f.write('description\n')
            for record in records:
                f.write(record['Data'][0]['Value']['StringWithMarkup'][0]['String'] + '\n')


def prepare_annotations_from_URLs(URLs, file, *, CID=False):
    """Fetch annotations from PubChem PUG-View URLs and save them to a file.

    Args:
        URLs: List of PubChem PUG-View URLs.
        file: Output file path.
        CID: If True, include Compound IDs in output.
    """
    with open(file, 'w') as f:
        f.write('CID\tname\tdescription\n') if CID else f.write('description\n')

    for URL in URLs:
        for attempt in range(11):  # 1 initial attempt + 10 retries
            r = requests.get(URL)
            if r.status_code == 200:
                prepare_PubChem_annotations(
                    r.json()['Annotations']['Annotation'],
                    file=file,
                    mode='a',
                    header=False,
                    CID=CID
                )
                print(f"Successfully processed the URL: \"{URL}\"")
                break
            else:
                if attempt == 10:  # Final attempt failed
                    print(f"ERROR: Failed to process \"{URL}\". Status code: {r.status_code}")
                elif attempt > 0:
                    print(f"Retry {attempt}/10 for URL: \"{URL}\"")


# prepare_annotations_from_URLs(CHEBI_annotation_URLs, chebi_output_path, CID=True)

HMDB_ID_URLs = [
    get_PubChem_PUG_View_URL(
        'annotations',
        'heading',
        'JSON',
        f"source=Human+MetaboLome+Database+(HMDB)&heading_type=Compound&heading=HMDB+ID&page={i + 1}"
    )
    for i in range(218)
]

# prepare_annotations_from_URLs(HMDB_ID_URLs, hmdb_output_path, CID=True)

with open(hmdb_output_path) as f:
    HMDB_IDs = {HMDB_ID: CID for line in f.readlines()[1:] for CID, name, HMDB_ID in [line.strip().split('\t')]}

tree = ET.parse(hmdb_xml_path)
root = tree.getroot()

namespace = {'hmdb': 'http://www.hmdb.ca'}

with open(hmdb_annotation_path, 'w', newline='') as f, open('error.txt', 'w') as f_error:
    writer = csv.writer(f, delimiter='\t')
    writer.writerow(['index', 'accession', 'CID', 'name', 'description'])

    for index, metabolite in enumerate(root.findall('hmdb:metabolite', namespace)):
        accession = metabolite.find('hmdb:accession', namespace)
        CID = metabolite.find('hmdb:pubchem_compound_id', namespace)

        if CID is not None and CID.text:
            if accession.text in HMDB_IDs:
                if CID.text != HMDB_IDs[accession.text]:
                    f_error.write(
                        f"Different CIDs: {CID.text} from HMDB, {HMDB_IDs[accession.text]} from PubChem when processing \"{accession.text}\".\n")
                    continue
            else:
                f_error.write(f"Losing CID from PubChem when processing \"{accession.text}\" which index is {index}.\n")
                continue
        elif accession.text in HMDB_IDs:
            f_error.write(f"Losing CID from HMDB when processing \"{accession.text}\" which index is {index}.\n")
            continue
        else:
            f_error.write(
                f"Losing CIDs from both HMDB and PubChem when processing \"{accession.text}\" which index is {index}.\n")
            continue

        name = metabolite.find('hmdb:name', namespace)
        description = metabolite.find('hmdb:description', namespace)

        writer.writerow((index, accession.text, CID.text, name.text, description.text))


