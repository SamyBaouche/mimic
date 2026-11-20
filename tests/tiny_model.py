"""A tiny BERT + word-level tokenizer built offline, to test the matcher pipeline in CI.

It is randomly initialised and only knows words from the training data, so it cannot
generalise like the real pre-trained model; it proves the plumbing works.
"""

from mimic.matcher_data import CONCEPTS


def build_tiny(seed: int = 0):
    import torch
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors, trainers
    from transformers import BertConfig, BertForSequenceClassification, PreTrainedTokenizerFast

    corpus = [
        f"{noun} '{label}' ({typ})"
        for _, (kind, typ, labels) in CONCEPTS.items()
        for label in labels
        for noun in ["button", "input field", "dropdown"]
    ]
    tok = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tok.normalizer = normalizers.Lowercase()
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.train_from_iterator(
        corpus, trainers.WordLevelTrainer(special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"])
    )
    cls_id, sep_id = tok.token_to_id("[CLS]"), tok.token_to_id("[SEP]")
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", cls_id), ("[SEP]", sep_id)],
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    )
    torch.manual_seed(seed)
    config = BertConfig(
        vocab_size=tok.get_vocab_size(),
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        max_position_embeddings=128,
        num_labels=1,
    )
    return BertForSequenceClassification(config), tokenizer
