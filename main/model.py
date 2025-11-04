import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


device=('cuda' if torch.cuda.is_available() else 'cpu')

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,n_heads,d_queries,d_values,dropout,in_decoder=False):
        super(MultiHeadAttention,self).__init__()
        self.d_model=d_model
        self.n_heads=n_heads
        self.d_queries=d_queries
        self.d_keys=d_queries
        self.d_values=d_values
        self.dropout=dropout
        self.in_decoder=in_decoder

        self.cast_queires=nn.Linear(d_model,n_heads*d_queries)
        self.cast_key_values=nn.Linear(d_model,n_heads*(d_queries+d_values))

        self.cast_output=nn.Linear(n_heads*d_values,d_model)

        self.softmax=nn.Softmax(dim=-1)
        self.layer_norm=nn.LayerNorm(d_model)
        self.apply_dropout=nn.Dropout(dropout)
    
    def forward(self,query_sentences,key_value_sequence,key_value_sequence_length):
        batch_size=query_sentences.size(0)
        query_pad_len=query_sentences.size(1)
        key_value_pad_len=key_value_sequence.size(1)

        self_attention=torch.equal(query_sentences,key_value_sequence)

        input_to_add=query_sentences.clone()

        query_sentences=self.layer_norm(query_sentences)
        if self_attention:
            key_value_sequence=self.layer_norm(key_value_sequence)
        
        queries=self.cast_queires(query_sentences)
        keys,values=self.cast_key_values(key_value_sequence).split(split_size=self.n_heads*self.d_keys,dim=-1)

        queries=queries.contiguous().view(batch_size,query_pad_len,self.n_heads,self.d_queries)
        keys=keys.contiguous().view(batch_size,key_value_pad_len,self.n_heads,self.d_keys)
        values=values.contiguous().view(batch_size,key_value_pad_len,self.n_heads,self.d_values)

        queries=queries.permute(0,2,1,3).contiguous().view(-1,query_pad_len,self.d_queries)
        keys=keys.permute(0,2,1,3).contiguous().view(-1,key_value_pad_len,self.d_keys)
        values=values.permute(0,2,1,3).contiguous().view(-1,key_value_pad_len,self.d_values)

        attention_weights=torch.bmm(queries,keys.permute(0,2,1))

        attention_weights=attention_weights/math.sqrt(self.d_keys)

        not_in_pad=torch.LongTensor(range(key_value_pad_len)).unsqueeze(0).unsqueeze(0).expand_as(attention_weights).to(device)
        not_in_pad=not_in_pad < key_value_sequence_length.repeat_interleave(self.n_heads).unsqueeze(1).unsqueeze(2).expand_as(attention_weights)

        attention_weights=attention_weights.masked_fill(~not_in_pad,value=-float('inf'))

        if self.in_decoder and self_attention:
            not_future_mask=torch.ones_like(attention_weights).tril().bool().to(device)
            attention_weights=attention_weights.masked_fill(~not_future_mask,value=-float('inf'))
        attention_weights=self.softmax(attention_weights)
        attention_weights=self.apply_dropout(attention_weights)

        sequence=torch.bmm(attention_weights,values)

        sequence=sequence.contiguous().view(batch_size,self.n_heads,query_pad_len,self.d_values).permute(0,2,1,3)
        sequence=sequence.contiguous().view(batch_size,query_pad_len,-1)

        sequence=self.cast_output(sequence)

        sequence=self.apply_dropout(sequence) + input_to_add

        return sequence

class PostionWiseFCNetwork(nn.Module):
    def __init__(self,d_model,d_inner,dropout):
        super(PostionWiseFCNetwork,self).__init__()
        self.d_model=d_model
        self.d_inner=d_inner
        self.dropout=dropout

        self.layer_norm=nn.LayerNorm(d_model)
        self.fc1=nn.Linear(d_model,d_inner)
        self.relu=nn.ReLU()
        self.fc2=nn.Linear(d_inner,d_model)
        self.apply_dropout=nn.Dropout(dropout)
    
    def forward(self,sequences):
        input_to_add=sequences.clone()

        sequences=self.layer_norm(sequences)
        sequences=self.fc1(sequences)
        sequences=self.relu(sequences)
        sequences=self.apply_dropout(sequences)
        sequences=self.fc2(sequences)
        sequences=self.apply_dropout(sequences)

        sequences=sequences+input_to_add
        return sequences
    
class Encoder(nn.Module):
    def __init__(self,vocab_size,positional_encoding,d_model,n_heads,d_queries,d_values,d_inner,n_layers,dropout):
        super(Encoder,self).__init__()
        self.vocan_size=vocab_size
        self.positional_encoding=positional_encoding
        self.d_model=d_model
        self.d_queries=d_queries
        self.n_heads=n_heads
        self.d_values=d_values
        self.d_inner=d_inner
        self.n_layers=n_layers
        self.dropout=dropout

        self.embedding=nn.Embedding(vocab_size,d_model)
        self.positional_encoding.requires_grad=False
        self.encoder_layers=nn.ModuleList([self.make_encoder_layer() for  i in range(self.n_layers)])
        self.apply_dropout=nn.Dropout(dropout)
        self.layer_norm=nn.LayerNorm(d_model)
    
    def make_encoder_layer(self):
        encoder_layer=nn.ModuleList([MultiHeadAttention(self.d_model,self.n_heads,self.d_queries,self.d_values,self.dropout,in_decoder=False,),
                                     PostionWiseFCNetwork(self.d_model,self.d_inner,self.dropout)])
        return encoder_layer
    def forward(self,encoder_sequences,encoder_sequences_len):
        pad_len=encoder_sequences.size(1)
        encoder_sequences = self.embedding(encoder_sequences) * math.sqrt(self.d_model) + self.positional_encoding[:,:pad_len, :].to(device)

        encoder_sequences=self.apply_dropout(encoder_sequences)

        for encoder_layer in self.encoder_layers:
            encoder_sequences = encoder_layer[0](query_sentences=encoder_sequences,
                                                 key_value_sequence=encoder_sequences,
                                                 key_value_sequence_length=encoder_sequences_len)  
            encoder_sequences = encoder_layer[1](sequences=encoder_sequences)
        encoder_sequences=self.layer_norm(encoder_sequences)
        return encoder_sequences

class Decoder(nn.Module):
    def __init__(self,vocab_size,positional_encoding,d_model,n_heads,d_queries,d_values,d_inner,n_layers,dropout):
        super(Decoder,self).__init__()

        self.vocan_size=vocab_size
        self.positional_encoding=positional_encoding
        self.d_model=d_model
        self.d_queries=d_queries
        self.n_heads=n_heads
        self.d_values=d_values
        self.d_inner=d_inner
        self.n_layers=n_layers
        self.dropout=dropout

        self.embedding=nn.Embedding(vocab_size,d_model)
        self.positional_encoding.requires_grad=False
        self.decoder_layers=nn.ModuleList([self.make_decoder_layers() for _ in range(n_layers)])
        self.apply_dropout=nn.Dropout(dropout)
        self.layer_norm=nn.LayerNorm(d_model)
        self.fc=nn.Linear(d_model,vocab_size)
    
    def make_decoder_layers(self):
        decoder_layer=nn.ModuleList([
            MultiHeadAttention(d_model=self.d_model,
                               n_heads=self.n_heads,
                               d_queries=self.d_queries,
                               d_values=self.d_values,
                               dropout=self.dropout,
                               in_decoder=True),
            MultiHeadAttention(d_model=self.d_model,
                               n_heads=self.n_heads,
                               d_queries=self.d_queries,
                               d_values=self.d_values,
                               dropout=self.dropout,
                               in_decoder=True),
            PostionWiseFCNetwork(d_model=self.d_model,
                                 d_inner=self.d_inner,
                                 dropout=self.dropout)
        ])
        return decoder_layer
    
    def forward(self,decoder_sequences,decoder_sequences_len,encoder_sequences,encoder_sequences_len):
        pad_len=decoder_sequences.size(1)

        decoder_sequences=self.embedding(decoder_sequences)*math.sqrt(self.d_model) + self.positional_encoding[:,:pad_len,:].to(device)
        decoder_sequences=self.apply_dropout(decoder_sequences)
        #query_sequences,key_value_sequences,key_value_sequence_lengths
        for decoder_layer in self.decoder_layers:
            decoder_sequences=decoder_layer[0](decoder_sequences,decoder_sequences,decoder_sequences_len)
            decoder_sequences=decoder_layer[1](decoder_sequences,encoder_sequences,encoder_sequences_len)
            decoder_sequences=decoder_layer[2](decoder_sequences)
        
        decoder_sequences=self.layer_norm(decoder_sequences)
        decoder_sequences=self.fc(decoder_sequences)
        return decoder_sequences

class Transformer(nn.Module):
    def __init__(self, vocab_size, positional_encoding, d_model=512, n_heads=8, d_queries=64, d_values=64,d_inner=2048, n_layers=6, dropout=0.1):
        super(Transformer,self).__init__()
        
        self.vocab_size = vocab_size
        self.positional_encoding = positional_encoding
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_queries = d_queries
        self.d_values = d_values
        self.d_inner = d_inner
        self.n_layers = n_layers
        self.dropout = dropout

        self.encoder=Encoder(vocab_size=vocab_size,
                             positional_encoding=positional_encoding,
                             d_model=d_model,
                             n_heads=n_heads,
                             d_queries=d_queries,
                             d_values=d_values,
                             d_inner=d_inner,
                             n_layers=n_layers,
                             dropout=dropout)
        self.decoder=Decoder(vocab_size=vocab_size,
                             positional_encoding=positional_encoding,
                             d_model=d_model,
                             n_heads=n_heads,
                             d_queries=d_queries,
                             d_values=d_values,
                             d_inner=d_inner,
                             n_layers=n_layers,
                             dropout=dropout)
        self.init_weights()
    def init_weights(self):
        for p in self.parameters():
            if p.dim()>1:
                nn.init.xavier_normal_(p,gain=0.1)

        nn.init.normal_(self.encoder.embedding.weight,mean=0,std=math.pow(self.d_model,-0.5))
        self.decoder.embedding.weight=self.encoder.embedding.weight
        self.decoder.fc.weight=self.decoder.embedding.weight
    
    def forward(self,encoder_sequences,decoder_sequences,encoder_sequences_len,decoder_sequences_len):
        encoder_sequences=self.encoder(encoder_sequences,encoder_sequences_len)
        decoder_sequences=self.decoder(decoder_sequences,decoder_sequences_len,encoder_sequences,encoder_sequences_len)
        return decoder_sequences

class LabelSmoothCE(nn.Module):
    def __init__(self,eps=0.1):
        super(LabelSmoothCE,self).__init__()
        self.eps=eps
    
    def forward(self,inputs,targets,lengths):
        inputs,_,_,_= pack_padded_sequence(input=inputs,
                                           lengths=lengths.cpu(),
                                           enforce_sorted=True,
                                           batch_first=True)
        target,_,_,_=pack_padded_sequence(input=targets,
                                         lengths=lengths.cpu(),
                                         enforce_sorted=True,
                                         batch_first=True)
        target_vector=torch.zeros_like(inputs).scatter(dim=1,index=target.unsqueeze(1),value=1.0).to(device)
        target_vector=target_vector * (1-self.eps) +self.eps/target_vector.size(1)

        loss=(-1*target_vector*F.log_softmax(inputs,dim=1)).sum(dim=1)

        loss=torch.mean(loss)

        return loss

