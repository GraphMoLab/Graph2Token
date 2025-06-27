# Installation
* conda create -n graph2token python=3.8
* conda activate graph2token
* conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
* conda install pyg -c pyg
* pip install rouge_score nltk ogb peft rdkit salesforce-lavis
* pip install -U transformers pytorch-lightning
* pip install deepspeed
* Download nltk corpus:
```bash
import nltk

nltk.download('wordnet')
```
## Stage 1
Run the code in \GNN_pretrained to get the graph-text data and pretrain the GNN encoder.
## Stage 2
* Download the LLama3-8B checkpoint from Huggingface.
* Set your LLM model, pretrained gnn and mol2latent model paths.
* For classification task, run iupac_finetune_classification.py.
* For regression task, run iupac_finetune_regression.py.
* For few-shot finetuning and evaluation, set --stage2_path and the data path, then rerun.

