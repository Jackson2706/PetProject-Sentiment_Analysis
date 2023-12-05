from torchtext.vocab import build_vocab_from_iterator


def yield_tokens(sentences, tokenizer):
    for sentence in sentences:
        yield tokenizer(sentence)


def build_vocabulary(sentences, tokenizer):
    vocabulary = build_vocab_from_iterator(
        yield_tokens(sentences, tokenizer),
        specials=["<unk>"],
    )
    vocabulary.set_default_index(vocabulary["<unk>"])
    return vocabulary
