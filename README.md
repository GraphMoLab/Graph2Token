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
## Run the code in \GNN_pretrained to get the graph-text data and pretrain the GNN encoder.
## Run
Coming soon !
