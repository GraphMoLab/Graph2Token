from typing import Any, Optional, Dict

from lightning_fabric.utilities.types import _PATH
from pytorch_lightning import strategies
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from tqdm import tqdm
import numpy as np
import torch


class MyDeepSpeedStrategy(strategies.DeepSpeedStrategy):
    def save_checkpoint(
            self, checkpoint: Dict[str, Any], filepath: _PATH, storage_options: Optional[Any] = None
    ):
        # use this method when no need to save the optimizer states
        """Save model/training states as a checkpoint file through state-dump and file-write.

        Args:
            checkpoint: dict containing model and trainer state
            filepath: write-target file's path
            storage_options: parameter for how to save to st
            orage, passed to ``CheckpointIO`` plugin
        """
        if self.is_global_zero:
            self.checkpoint_io.save_checkpoint(checkpoint, filepath, storage_options=storage_options)

    def save_checkpoint_v2(self, checkpoint: Dict, filepath: _PATH, storage_options: Optional[Any] = None) -> None:
        # use this method when need to save the optimizer states
        """Save model/training states as a checkpoint file through state-dump and file-write.

        Args:
            checkpoint: The checkpoint state dictionary
            filepath: write-target file's path
            storage_options: not used for ``DeepSpeedStrategy`` as ``CheckpointIO`` is not used

        Raises:
            TypeError:
                If ``storage_options`` arg is passed in
        """
        # broadcast the filepath from rank 0 to ensure all the states are saved in a common filepath
        filepath = self.broadcast(filepath)
        if storage_options is not None:
            raise TypeError(
                "`Trainer.save_checkpoint(..., storage_options=...)` with `storage_options` arg"
                f" is not supported for `{self.__class__.__name__}` as `CheckpointIO` is not used."
            )

        if self.zero_stage_3 and self._multi_device and self.is_global_zero:
            print(
                "Warning: When saving the DeepSpeed Stage 3 checkpoint, "
                "each worker will save a shard of the checkpoint within a directory. "
                "If a single file is required after training, "
                "see https://lightning.ai/docs/pytorch/stable/advanced/model_parallel.html#"
                "deepspeed-zero-stage-3-single-file for instructions."
            )
        # Use deepspeed's internal checkpointing function to handle partitioned weights across processes
        # dump states as a checkpoint dictionary object
        _exclude_keys = ["state_dict", "optimizer_states"]
        checkpoint = {k: v for k, v in checkpoint.items() if k not in _exclude_keys}
        self.deepspeed_engine.save_checkpoint(filepath, client_state=checkpoint, tag="checkpoint",
                                              exclude_frozen_parameters=True)


def caption_evaluate(predictions, targets, tokenizer, text_trunc_length):
    meteor_scores = []
    references = []
    hypotheses = []
    for gt, out in tqdm(zip(targets, predictions)):
        gt_tokens = tokenizer.tokenize(gt, truncation=True, max_length=text_trunc_length,
                                       padding='max_length')
        gt_tokens = list(filter(('[PAD]').__ne__, gt_tokens))
        gt_tokens = list(filter(('[CLS]').__ne__, gt_tokens))
        gt_tokens = list(filter(('[SEP]').__ne__, gt_tokens))

        out_tokens = tokenizer.tokenize(out, truncation=True, max_length=text_trunc_length,
                                        padding='max_length')
        out_tokens = list(filter(('[PAD]').__ne__, out_tokens))
        out_tokens = list(filter(('[CLS]').__ne__, out_tokens))
        out_tokens = list(filter(('[SEP]').__ne__, out_tokens))

        references.append([gt_tokens])
        hypotheses.append(out_tokens)

        mscore = meteor_score([gt_tokens], out_tokens)
        meteor_scores.append(mscore)

    bleu2 = corpus_bleu(references, hypotheses, weights=(.5, .5))
    bleu4 = corpus_bleu(references, hypotheses, weights=(.25, .25, .25, .25))
    bleu2 *= 100
    bleu4 *= 100

    print('BLEU-2 score:', bleu2)
    print('BLEU-4 score:', bleu4)
    _meteor_score = np.mean(meteor_scores)
    _meteor_score *= 100
    print('Average Meteor score:', _meteor_score)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])

    rouge_scores = []

    references = []
    hypotheses = []

    for gt, out in tqdm(zip(targets, predictions)):
        rs = scorer.score(out, gt)
        rouge_scores.append(rs)

    print('ROUGE score:')
    rouge_1 = np.mean([rs['rouge1'].fmeasure for rs in rouge_scores]) * 100
    rouge_2 = np.mean([rs['rouge2'].fmeasure for rs in rouge_scores]) * 100
    rouge_l = np.mean([rs['rougeL'].fmeasure for rs in rouge_scores]) * 100
    print('rouge1:', rouge_1)
    print('rouge2:', rouge_2)
    print('rougeL:', rouge_l)
    return bleu2, bleu4, rouge_1, rouge_2, rouge_l, _meteor_score


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def pad_and_concat(tensor_list, fill_value=0):
    '''
    concat the first dimension and pad the second dimension
    tensor_list: [[B (diff), N_num, *], ...]
    '''
    device = tensor_list[0].device
    dtype = tensor_list[0].dtype
    max_dim1 = max(t.shape[1] for t in tensor_list)
    sum_dim0 = sum(t.shape[0] for t in tensor_list)
    if len(tensor_list[0].shape) == 3:
        out = torch.full((sum_dim0, max_dim1, tensor_list[0].shape[-1]), fill_value=fill_value, device=device,
                         dtype=dtype)
        i = 0
        for t in tensor_list:
            out[i:i + t.shape[0], :t.shape[1]] = t
            i += t.shape[0]
        return out
    elif len(tensor_list[0].shape) == 2:
        out = torch.full((sum_dim0, max_dim1), fill_value=fill_value, device=device, dtype=dtype)
        i = 0
        for t in tensor_list:
            out[i:i + t.shape[0], :t.shape[1]] = t
            i += t.shape[0]
        return out
    raise NotImplementedError()


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self