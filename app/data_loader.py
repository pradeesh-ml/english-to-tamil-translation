import os
import codecs
import youtokentome
import torch
from torch.nn.utils.rnn import pad_sequence
from itertools import groupby
from random import shuffle


class SequenceLoader(object):
    def __init__(self,path,source_suffix,target_suffix,split,tokens_in_batch):
        self.source_suffix=source_suffix
        self.target_suffix=target_suffix
        self.split=split.lower()
        self.tokens_in_batch=tokens_in_batch
        self.bpe_model=youtokentome.BPE(os.path.join(path,'bpe.model'))
        self.for_training=self.split=='train'

        with codecs.open(os.path.join(path,'.'.join([self.split,self.source_suffix])),'r',encoding='utf-8') as f:
            source_data=f.read().split('\n')[:-1]
        with codecs.open(os.path.join(path,'.'.join([self.split,self.target_suffix])),'r',encoding='utf-8') as f:
            target_data=f.read().split('\n')[:-1]
        
        assert len(source_data)==len(target_data)
        src_len=[len(s) for s in self.bpe_model.encode(source_data,bos=False,eos=False)]
        tgt_len=[len(t) for t in self.bpe_model.encode(target_data,bos=True,eos=True)]
        
        self.data=list(zip(source_data,target_data,src_len,tgt_len))
        if self.for_training:
            self.data.sort(key=lambda x: x[3])

        self.create_batches()

    def create_batches(self):
        if self.for_training:

            chunks=[list(g) for _,g in groupby(self.data,key=lambda x:x[3])]
            self.all_batches=list()

            for chunk in chunks:
                chunk.sort(key=lambda x:x[2])
                seq_per_batch=self.tokens_in_batch//chunk[0][3]
                self.all_batches.extend([chunk[i : i+seq_per_batch] for i in range(0,len(chunk),seq_per_batch)])
            
            shuffle(self.all_batches)
            self.n_batches=len(self.all_batches)
            self.current_batch=-1
        else:
            self.all_batches=[[d] for d in self.data]
            self.n_batches=len(self.all_batches)
            self.current_batch=-1
    def __iter__(self):
        return self

    def __next__(self):
        self.current_batch+=1
        try:
            source_data,target_data,source_len,target_len=zip(*self.all_batches[self.current_batch])
        except IndexError:
            raise StopIteration
        
        source_data=self.bpe_model.encode(source_data,bos=False,eos=False,output_type=youtokentome.OutputType.ID)
        target_data=self.bpe_model.encode(target_data,bos=True,eos=True,output_type=youtokentome.OutputType.ID)

        source_data=pad_sequence(sequences=[torch.LongTensor(s) for s in source_data] ,batch_first=True,padding_value=self.bpe_model.subword_to_id('<PAD>'))
        target_data=pad_sequence(sequences=[torch.LongTensor(t) for t in target_data],batch_first=True,padding_value=self.bpe_model.subword_to_id('<PAD>'))

        source_len=torch.LongTensor(source_len)
        target_len=torch.LongTensor(target_len)

        return source_data,target_data,source_len,target_len