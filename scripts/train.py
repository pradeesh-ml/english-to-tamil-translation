import time
import torch
import torch.backends.cudnn as cudnn
from model import Transformer,LabelSmoothCE
from app.data_loader import SequenceLoader
from app.utils import *
from tqdm import tqdm


path=r'E:\Learning Zone\DL\Translation\dataset'
d_model=256
n_heads=4
d_queries=64
d_values=64
d_inner=1024
n_layers=3
dropout=0.2

positional_embedding=get_positional_embedding(d_model,max_len=200)

checkpoint=r'E:\Learning Zone\DL\Translation\epoch348_transformer_checkpoint.pth.tar'
tokens_in_batch=2000
batch_per_step=25000 // tokens_in_batch
print_frequency=20
n_steps=100000
warmup_steps=4000
step=1
lr=get_lr(step,d_model,warmup_steps)
start_epoch=0
betas=(0.9,0.98)
epsilon=1e-9
label_smooting=0.1

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cudnn.benchmar=False

def main():
    global checkpoint,step,start_epoch,epoch,epochs

    train_loader=SequenceLoader(path=path,
                                source_suffix='en',
                                target_suffix='ta',
                                split='train',
                                tokens_in_batch=tokens_in_batch)
    val_loader=SequenceLoader(path=path,
                              source_suffix='en',
                              target_suffix='ta',
                              split='val',
                              tokens_in_batch=tokens_in_batch)
    if checkpoint is None:
        model=Transformer(vocab_size=train_loader.bpe_model.vocab_size(),
                          positional_encoding=positional_embedding,
                          d_model=d_model,
                          n_heads=n_heads,
                          d_queries=d_queries,
                          d_values=d_values,
                          d_inner=d_inner,
                          n_layers=n_layers,
                          dropout=dropout)
        optimizer=torch.optim.Adam(params=[p for p in model.parameters() if p.requires_grad],
                                   lr=lr,
                                   betas=betas,
                                   eps=epsilon)
    else:
        checkpoint=torch.load(checkpoint,weights_only=False)
        start_epoch=checkpoint['epoch'] + 1
        model=checkpoint['model']
        optimizer=checkpoint['optimizer']

    criterion=LabelSmoothCE(eps=label_smooting)
    model=model.to(device)
    criterion=criterion.to(device)
    epochs=(n_steps // (train_loader.n_batches // batch_per_step)) + 1

    for epoch in range(start_epoch,epochs):
        step=epoch*train_loader.n_batches // batch_per_step
        train_loader.create_batches()
        train(train_loader,model,criterion,optimizer,epoch,step)
        val_loader.create_batches()
        val(val_loader,model,criterion)
        save_checkpoint(epoch, model, optimizer, prefix='epoch'+str(epoch+1)+'_')

def train(train_loader,model,criterion,optimizer,epoch,step):
    model.train()

    data_time=AverageMeter()
    step_time=AverageMeter()
    losses=AverageMeter()

    start_data_time=time.time()
    start_step_time=time.time()

    for i,(source_sequence,target_sequence,source_sequence_len,target_sequence_len) in enumerate(train_loader):
        source_sequence=source_sequence.to(device)
        target_sequence=target_sequence.to(device)
        source_sequence_len=source_sequence_len.to(device)
        target_sequence_len=target_sequence_len.to(device)

        data_time.update(time.time()-start_data_time)

        predicted_sequence=model(source_sequence,target_sequence,source_sequence_len,target_sequence_len)

        loss=criterion(inputs=predicted_sequence,
                       targets=target_sequence[:,1:],
                       lengths=target_sequence_len-1)
        
        (loss/batch_per_step).backward()
        losses.update(loss.item(),(target_sequence_len-1).sum().item())

        if (i+1)%batch_per_step==0:
            optimizer.step()
            optimizer.zero_grad()

            step+=1
            change_lr(optimizer,lr=get_lr(step,d_model,warmup_steps))
            step_time.update(time.time()-start_step_time)

            if step%print_frequency==0:
                print(f'Epoch: [{epoch+1}/{epochs}][{i+1}/{train_loader.n_batches}] '
                      f'Step {step}/{n_steps} '
                      f'Loss {losses.val:.4f} ({losses.avg:.4f}) '
                      f'Step Time {step_time.val:.3f} ({step_time.avg:.3f}) '
                      f'Data Time {data_time.val:.3f} ({data_time.avg:.3f}) '
                      f'LR {optimizer.param_groups[0]["lr"]:.6f}')
            start_step_time=time.time()
            if epoch in [epochs-1,epochs-2] and step%1500==0:
                save_checkpoint(epoch,model,prefix='step'+str(step)+'_')
        start_data_time=time.time()


def val(val_loader,model,criterion):
    model.eval()
    with torch.no_grad():
        losses=AverageMeter()
        for i,(source_sequence,target_sequence,source_sequence_len,target_sequence_len) in enumerate(tqdm(val_loader,total=val_loader.n_batches)):
            source_sequence=source_sequence.to(device)
            target_sequence=target_sequence.to(device)
            source_sequence_len=source_sequence_len.to(device)
            target_sequence_len=target_sequence_len.to(device)

            predicted_sequence=model(source_sequence,target_sequence,source_sequence_len,target_sequence_len)

            loss=criterion(inputs=predicted_sequence,
                           targets=target_sequence[:,1:],
                           lengths=target_sequence_len-1)
            losses.update(loss.item(),(target_sequence_len-1).sum().item())
        print(f'Validation Loss {losses.avg:.4f}')



if __name__=="__main__":
    main()