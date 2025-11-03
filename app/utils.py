import torch
import math

def get_positional_embedding(d_model,max_len=200):
    positional_encoding=torch.zeros((max_len,d_model))
    for i in range(max_len):
        for j in range(d_model):
            if j%2==0:
                positional_encoding[i][j]=math.sin(i/math.pow(1000,j/d_model))
            else:
                positional_encoding[i][j]=math.cos(i/math.pow(1000,((j-1)/d_model)))
    positional_encoding=positional_encoding.unsqueeze(0)
    return positional_encoding

def get_lr(step,d_model,warmup_steps):
    lr=2*math.pow(d_model,-0.5) * min(math.pow(step,-0.5), step*math.pow(warmup_steps,-1.5))
    return lr
def change_lr(optimizer,lr):
    for param_group in optimizer.param_groups:
        param_group['lr']=lr
def save_checkpoint(epoch,model,optimizer,prefix=''):
    state={'epoch':epoch,
           'model':model,
           'optimizer':optimizer}
    filename=prefix +'transformer_checkpoint.pth.tar'
    torch.save(state,filename)
class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val=0
        self.avg=0
        self.sum=0
        self.count=0
    def update(self,val,n=1):
        self.val=val
        self.sum+=val*n
        self.count+=n
        self.avg=self.sum/self.count