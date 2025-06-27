## Dataset
### step 1  Download the text description
Download all information containing all HMDB IDs from [HMDB xml file](https://hmdb.ca/downloads). Then run:
```bash
python get_stage1_dataset.py
```
### step 2  Download all the SDF files
```bash
python get_molecule_sdf.py
```
### step 3  
```bash
python match_smiles_annotation.py
```
### step 4 
```bash
python extract_description.py
```
### step 5
```bash
python process_hmdb_chebi_dataset.py
```
* Note: all above should set the your file path.

## Train GNN
Download SciBERT model from Huggingface. This can be done by simplying calling the following for SciBERT:
```bash
bert_name = 'Your SciBERT model path'
text_tokenizer = AutoTokenizer.from_pretrained(bert_name, cache_dir=pretrained_SciBERT_folder)
text_model = AutoModel.from_pretrained(bert_name, cache_dir=pretrained_SciBERT_folder).to(device)
```
Download GraphMVP model, check this [repo](https://github.com/chao1224/GraphMVP), and the checkpoints on this [link](https://drive.google.com/drive/folders/1uPsBiQF3bfeCAXSDd4JfyXiTh-qxYfu6).
Put it at the your file path and then run:
```bash
python pretrain.py
```
