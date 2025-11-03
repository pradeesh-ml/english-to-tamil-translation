import torch
import sacrebleu
from app.translator import translate
from tqdm import tqdm
from app.data_loader import SequenceLoader
import youtokentome
import codecs
import os

sacrebleu_in_python = True
MAX_LEN=200

test_loader = SequenceLoader(path=r"E:\Learning Zone\DL\Translation\dataset",
                             source_suffix="en",
                             target_suffix="ta",
                             split="test",
                             tokens_in_batch=None)
test_loader.create_batches()

with torch.no_grad():
    hypotheses = list()
    references = list()
    for i, (source_sequence, target_sequence, source_sequence_length, target_sequence_length) in enumerate(
            tqdm(test_loader, total=test_loader.n_batches)):
        
        seq_len=source_sequence.size(1)
        if seq_len>MAX_LEN:
            continue
        hypotheses.append(translate(source_sequence=source_sequence,
                                    beam_size=4,
                                    length_norm_coefficient=0.6)[0])
        references.extend(test_loader.bpe_model.decode(target_sequence.tolist(), ignore_ids=[0, 2, 3]))
    
    print("\n13a tokenization, cased:\n")
    print(sacrebleu.corpus_bleu(hypotheses, [references]))
    print("\n13a tokenization, caseless:\n")
    print(sacrebleu.corpus_bleu(hypotheses, [references], lowercase=True))
    print("\nInternational tokenization, cased:\n")
    print(sacrebleu.corpus_bleu(hypotheses, [references], tokenize='intl'))
    print("\nInternational tokenization, caseless:\n")
    print(sacrebleu.corpus_bleu(hypotheses, [references], tokenize='intl', lowercase=True))