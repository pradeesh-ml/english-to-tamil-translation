from datasets import load_dataset
import os
import codecs
import youtokentome
from tqdm import tqdm


def load_data(path,train_subset_size=500000,val_subset_size=20000,test_subset_size=15000):
    print("Loading Dataset...")
    dataset=load_dataset("ai4bharat/samanantar", "ta")
    print("Original Dataset size:",len(dataset['train']))

    train_test_split=dataset['train'].train_test_split(test_size=0.01,seed=101)
    test_data=train_test_split['test']

    train_val_split=train_test_split['train'].train_test_split(test_size=0.0123,seed=101)
    val_data=train_val_split['test']
    train_data=train_val_split['train']

    print("Train Size:",len(train_data))
    print("Val Size:",len(val_data))
    print("Test Size:",len(test_data))

    train_data=train_data.shuffle(seed=101).select(range(train_subset_size))
    val_data=val_data.shuffle(seed=101).select(range(val_subset_size))
    test_data=test_data.shuffle(seed=101).select(range(test_subset_size))

    print("Subset Train Size:",len(train_data))
    print("Subset Val Size:",len(val_data)) 
    print("Subset Test Size:",len(test_data))

    with codecs.open(os.path.join(path,'train.en'),'w',encoding='utf-8') as f:
        for line in train_data['src']:
            f.write(line.strip()+'\n')
    with codecs.open(os.path.join(path,'train.ta'),'w',encoding='utf-8') as f:
        for line in train_data['tgt']:
            f.write(line.strip()+'\n')
    
    with codecs.open(os.path.join(path,'val.en'),'w',encoding='utf-8') as f:
        for line in val_data['src']:
            f.write(line.strip()+'\n')
    with codecs.open(os.path.join(path,'val.ta'),'w',encoding='utf-8') as f:
        for line in val_data['tgt']:
            f.write(line.strip()+'\n')
    
    with codecs.open(os.path.join(path,'test.en'),'w',encoding='utf-8') as f:
        for line in test_data['src']:
            f.write(line.strip()+'\n')
    with codecs.open(os.path.join(path,'test.ta'),'w',encoding='utf-8') as f:
        for line in test_data['tgt']:
            f.write(line.strip()+'\n')
    print("Data loaded and saved successfully.")

def prepare_data(path,retain_case=True,min_len=3,max_len=150,max_len_ratio=2.):
    english=list()
    tamil=list()
    
    with codecs.open(os.path.join(path,'train.en'),'r',encoding='utf-8') as f:
        if retain_case:
            english.extend(f.read().split('\n'))
        else:
            english.extend(f.read().lower().split('\n'))
    with codecs.open(os.path.join(path,'train.ta'),'r',encoding='utf-8') as f:
        if retain_case:
            tamil.extend(f.read().split('\n'))
        else:
            tamil.extend(f.read().lower().split('\n'))

    assert len(english)==len(tamil)

    with codecs.open(os.path.join(path,'train.en'),'w',encoding='utf-8') as f:
        f.write('\n'.join(english))
    with codecs.open(os.path.join(path,'train.ta'),'w',encoding='utf-8') as f:
        f.write('\n'.join(tamil))
    with codecs.open(os.path.join(path,'train.enta'),'w',encoding='utf-8') as f:
        f.write('\n'.join(english+tamil))
    del english,tamil

    print('\nLearning BPE Model...')
    youtokentome.BPE.train(data=os.path.join(path,'train.enta'),vocab_size=37000,model=os.path.join(path,'bpe.model'))
    print('\nBPE Model Trained and Saved Successfully')

    print('\nLoading BPE Model...')
    bpe_model=youtokentome.BPE(os.path.join(path,'bpe.model'))

    with codecs.open(os.path.join(path,'train.en'),'r',encoding='utf-8') as f:
        english=f.read().split('\n')
    with codecs.open(os.path.join(path,'train.ta'),'r',encoding='utf-8') as f:
        tamil=f.read().split('\n')
    
    pairs=list()
    for en,ta in tqdm(zip(english,tamil),total= len(english)):
        en_tok=bpe_model.encode(en,output_type=youtokentome.OutputType.ID)
        ta_tok=bpe_model.encode(ta,output_type=youtokentome.OutputType.ID)
        len_en_tok=len(en_tok)
        len_ta_tok=len(ta_tok)  

        if min_len < len_en_tok < max_len and min_len < len_ta_tok < max_len and 1./max_len_ratio <= len_en_tok/len_ta_tok <=max_len_ratio:
            pairs.append((en,ta))
        else:
            continue
    print(f'{(100.*(len(english)-len(pairs))/len(english))} % of en-ta pairs were filtered out based on sub word sequence length limit')
    print("Filtered Pairs:",len(pairs))
    english,tamil=zip(*pairs)
    print(english[:5])
    print(tamil[:5])
    os.remove(os.path.join(path,'train.en'))
    os.remove(os.path.join(path,'train.ta'))
    os.remove(os.path.join(path,'train.enta'))

    with codecs.open(os.path.join(path,'train.en'),'w',encoding='utf-8') as f:
        f.write('\n'.join(english))
    with codecs.open(os.path.join(path,'train.ta'),'w',encoding='utf-8') as f:
        f.write('\n'.join(tamil))
    
    del english,tamil,pairs,bpe_model

if __name__=="__main__":
    data_path=r'E:\Learning Zone\DL\Translation\dataset'
    load_data(path=data_path,
              train_subset_size=500000,
              val_subset_size=20000,
              test_subset_size=15000)
    prepare_data(path=data_path,
                 retain_case=True,
                 min_len=3,
                 max_len=150,
                 max_len_ratio=2.)



