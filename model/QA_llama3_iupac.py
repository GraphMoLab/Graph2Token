import logging
import os

import torch
import torch.nn as nn
from math import sqrt

from torch.nn import L1Loss

from transformers import AutoTokenizer

# from model.gin_model import GNN
from model.modeling_llama3 import LlamaForSequenceClassification
from model.molecule_gnn.molecule_gnn_model import GNN, GNN_graphpred

local_rank = int(os.environ.get('LOCAL_RANK', '0'))
device_map = {'': local_rank}


def init_graph_encoder(gin_num_layers, gin_hidden_dim, gin_drop_ratio):
    # graph_encoder = GNN(
    #     num_layer=gin_num_layers,
    #     emb_dim=gin_hidden_dim,
    #     gnn_type='gin',
    #     drop_ratio=gin_drop_ratio,
    #     JK='last',
    #     # JK='sum',
    # )

    molecule_node_model = GNN(
        num_layer=gin_num_layers,
        emb_dim=gin_hidden_dim,
        JK='last',
        drop_ratio=gin_drop_ratio,
        gnn_type='gin'
    )
    molecule_model = GNN_graphpred(
        num_layer=gin_num_layers,
        emb_dim=gin_hidden_dim,
        JK='last',
        graph_pooling='mean',
        num_tasks=1,
        molecule_node_model=molecule_node_model
    )
    input_model_path = 'molecule_model_final.pth'
    print("Loading from {}...".format(input_model_path))
    state_dict = torch.load(input_model_path, map_location='cpu')
    molecule_model.load_state_dict(state_dict)

    # ckpt = torch.load('pretrained_model/graphcl_80.pth', map_location=torch.device('cpu'))
    # missing_keys, unexpected_keys = graph_encoder.load_state_dict(ckpt, strict=False)
    # if len(missing_keys) or len(unexpected_keys):
    #     print(missing_keys)
    #     print(unexpected_keys)

    # ln_graph = LayerNorm(graph_encoder.num_features)
    # ln_graph = nn.LayerNorm(4096)

    return molecule_model


def init_mol2latent(gin_hidden_dim):
    mol2latent = nn.Linear(gin_hidden_dim, 512)
    input_model_path = 'mol2latent_model_final.pth'
    print("Loading from {}...".format(input_model_path))
    state_dict = torch.load(input_model_path, map_location='cpu')
    mol2latent.load_state_dict(state_dict)
    return mol2latent


class ReprogrammingLayer(nn.Module):
    def __init__(self, attention_dim, graph_dim, n_heads, d_keys=None, llm_dim=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (attention_dim // n_heads)

        self.query_projection = nn.Linear(graph_dim, d_keys * n_heads)
        # self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.key_projection = nn.Linear(llm_dim, d_keys * n_heads)
        # self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(llm_dim, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, llm_dim)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding


class Align2llama(nn.Module):

    def __init__(
            self,
            gin_num_layers,
            gin_hidden_dim,
            gin_drop_ratio,
            tune_gnn=False,
            lora_tuning=False,
            llm_model="decapoda-research/llama-7b-hf",
            args=None,
    ):
        super().__init__()
        self.graph_encoder = init_graph_encoder(
            gin_num_layers,
            gin_hidden_dim,
            gin_drop_ratio
        )
        self.tune_gnn = tune_gnn

        # for name, param in self.graph_encoder.named_parameters():
        #     param.requires_grad = False

        self.mol2latent = init_mol2latent(gin_hidden_dim)

        if not self.tune_gnn:
            for name, param in self.graph_encoder.named_parameters():
                param.requires_grad = False
            for name, param in self.mol2latent.named_parameters():
                param.requires_grad = False
            # self.graph_encoder = self.graph_encoder.eval()
            # self.graph_encoder.train = disabled_train
            logging.info("freeze graph encoder")

        # initialize opt model
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model,
                                                           use_fast=False,
                                                           padding_side='right')
        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.pad_token_id = self.llm_tokenizer.pad_token_id  # 128256
        # self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})

        self.llm_tokenizer.add_special_tokens({'additional_special_tokens': ['<|mol_g|>']})  # 128257
        self.llm_tokenizer.mol_token_id = self.llm_tokenizer("<|mol_g|>", add_special_tokens=False).input_ids[0]
        self.llm_tokenizer.add_special_tokens({'additional_special_tokens': ['<|reg_g|>']})  # 128258
        self.llm_tokenizer.reg_token_id = self.llm_tokenizer("<|reg_g|>", add_special_tokens=False).input_ids[0]

        # self.llm_model = LlamaForCausalLM.from_pretrained(llm_model, torch_dtype=torch.bfloat16)
        self.llm_model = LlamaForSequenceClassification.from_pretrained(llm_model, torch_dtype=torch.bfloat16)
        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

        for name, param in self.llm_model.named_parameters():
            param.requires_grad = False
        # for name, param in self.llm_model.named_parameters():
        #     if name == 'score.weight':
        #         param.requires_grad = True

        # self.bos_token_id = self.llm_tokenizer.bos_token_id  # 128000 <|begin_of_text|>
        # self.eos_token_id = self.llm_tokenizer.eos_token_id  # 128009 <|eot_id|>

        self.new_add_token = 3
        self.num_tokens = 100
        self.graph_dim = 512  # 300
        self.attention_dim = 512
        self.n_head = 8
        self.key = self.attention_dim // self.n_head
        self.llm_dim = 4096
        # self.output_hidden_dim = 256

        # self.word_embeddings = self.llm_model.get_input_embeddings().weight[:-self.new_add_token]
        # self.vocab_size = self.word_embeddings.shape[0]
        self.vocab_size = 128256
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)
        self.reprogramming_layer = ReprogrammingLayer(self.attention_dim, self.graph_dim, self.n_head, self.key, self.llm_dim)

        # self.down_graph_token = nn.Linear(self.llm_dim, self.output_hidden_dim)
        # self.activate = nn.ReLU()
        # self.score = nn.Linear(self.output_hidden_dim, 1, bias=False)
        self.score = nn.Linear(self.llm_dim, 1, bias=False)

        for name, param in self.mapping_layer.named_parameters():
            param.requires_grad = False


    def forward(self, batch):
        graphs, instruction_tokens, text_values = batch
        h_graph_feature, _ = self.graph_encoder(graphs)
        h_graph = self.mol2latent(h_graph_feature)

        # graph_atts_mask = torch.ones(h_graph.size()[0], dtype=torch.long).to(device)
        # graph_atts_mask = graph_atts_mask.unsqueeze(1)

        word_embeddings = self.llm_model.get_input_embeddings().weight[:-self.new_add_token]
        source_embeddings = self.mapping_layer(word_embeddings.permute(1, 0)).permute(1, 0)

        graph_inputs_llm = self.reprogramming_layer(h_graph.unsqueeze(1), source_embeddings, source_embeddings)
        instruction_embeds = self.llm_model.get_input_embeddings()(instruction_tokens.input_ids)

        word_embeddings_avg = word_embeddings.mean(dim=0, keepdim=True)

        instruction_embeds[instruction_tokens.is_mol_token] = graph_inputs_llm.flatten(0, 1)
        instruction_embeds[instruction_tokens.is_reg_token] = word_embeddings_avg.repeat(graph_inputs_llm.size()[0], 1)

        # inputs_embeds = torch.cat([instruction_embeds, graph_inputs_llm], dim=1)
        # attention_mask = torch.cat([instruction_tokens.attention_mask, graph_atts_mask], dim=1)

        outputs = self.llm_model(
            # inputs_embeds=inputs_embeds,
            # attention_mask=attention_mask,
            inputs_embeds=instruction_embeds,
            attention_mask=instruction_tokens.attention_mask,
            return_dict=True,
            # labels=targets,
            # use_cache=False,
        )

        output = outputs.hidden_states

        sequence_lengths = torch.eq(instruction_tokens.input_ids, self.pad_token_id).int().argmax(-1) - 1
        sequence_lengths = sequence_lengths % instruction_tokens.input_ids.shape[-1]
        sequence_lengths = sequence_lengths.to(output.device)

        output_token_embedding = output[torch.arange(output.shape[0], device=output.device), sequence_lengths]
        # graph_token_embedding = output[:, sequence_lengths, :]

        logits = self.score(output_token_embedding)
        # down_graph_token_embedding = self.activate(self.down_graph_token(graph_token_embedding))
        # logits = self.score(down_graph_token_embedding)
        # loss_fct = MSELoss()
        loss_fct = L1Loss()
        loss_mae = loss_fct(logits.squeeze(), text_values.squeeze())
        return {"loss": loss_mae}

    @torch.no_grad()
    def generate(
            self,
            samples,
            num_beams=None,
    ):
        graphs = samples['graphs']
        instruction_tokens = samples['instruction_tokens']
        # with self.maybe_autocast():
        h_graph_feature, _ = self.graph_encoder(graphs)
        h_graph = self.mol2latent(h_graph_feature)

        # graph_atts_mask = torch.ones(h_graph.size()[0], dtype=torch.long).to(h_graph.device)
        # graph_atts_mask = graph_atts_mask.unsqueeze(1)

        word_embeddings = self.llm_model.get_input_embeddings().weight[:-self.new_add_token]
        source_embeddings = self.mapping_layer(word_embeddings.permute(1, 0)).permute(1, 0)

        graph_inputs_llm = self.reprogramming_layer(h_graph.unsqueeze(1), source_embeddings, source_embeddings)
        instruction_embeds = self.llm_model.get_input_embeddings()(instruction_tokens.input_ids)

        word_embeddings_avg = word_embeddings.mean(dim=0, keepdim=True)

        instruction_embeds[instruction_tokens.is_mol_token] = graph_inputs_llm.flatten(0, 1)
        instruction_embeds[instruction_tokens.is_reg_token] = word_embeddings_avg.repeat(graph_inputs_llm.size()[0], 1)

        # inputs_embeds = torch.cat([instruction_embeds, graph_inputs_llm], dim=1)
        # attention_mask = torch.cat([instruction_tokens.attention_mask, graph_atts_mask], dim=1)

        outputs = self.llm_model(
            # inputs_embeds=inputs_embeds,
            # attention_mask=attention_mask,
            inputs_embeds=instruction_embeds,
            attention_mask=instruction_tokens.attention_mask,
            return_dict=True,
            # labels=targets,
            # use_cache=False,
        )

        output = outputs.hidden_states

        sequence_lengths = torch.eq(instruction_tokens.input_ids, self.pad_token_id).int().argmax(-1) - 1
        sequence_lengths = sequence_lengths % instruction_tokens.input_ids.shape[-1]
        sequence_lengths = sequence_lengths.to(output.device)

        output_token_embedding = output[torch.arange(output.shape[0], device=output.device), sequence_lengths]
        # graph_token_embedding = output[:, sequence_lengths, :]

        logits = self.score(output_token_embedding)
        # down_graph_token_embedding = self.activate(self.down_graph_token(graph_token_embedding))
        # logits = self.score(down_graph_token_embedding)

        return logits


