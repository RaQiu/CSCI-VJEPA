import torch 
import torch.nn as nn 

from torch.nn import functional as F
from tools.utils import rearrange
import torch.distributed as dist 



class Motion_loss(nn.Module):
    
    def __init__(self, debug=None):
        super(Motion_loss, self).__init__()
        self.debug = debug

    def forward(self, x):
        # x --> B , T, C
        # for i in range(x.shape[1]):x[:,i] = x[:,i] * 0 + i
        # x[:,0] = x[:,0] * 0 + 0
        # x[:,1] = x[:,1] * 0 + 1 
        delta = 1
        shifted_left = torch.cat([x[:, :1], x[:, :-1]], dim=1)
        # shifted_left[0,:,0] tensor([0., 0., 1., 2., 3., 4., 5., 6.], device='cuda:0', grad_fn=<SelectBackward0>)

        shifted_right = torch.cat([x[:, 1:], x[:, -1:]], dim=1)
        # tensor([1., 2., 3., 4., 5., 6., 7., 7.], device='cuda:0', grad_fn=<SelectBackward0>)

        # shifted_left = torch.cat([x[:, -1][:,None,:], x[:, :-1]], dim=1) # t=3 | t = 0,1,2   ==> 3,0,1,2
        # shifted_left[0,:,0] tensor([3., 0., 1., 2.], device='cuda:0', grad_fn=<SelectBackward0>)
        
        # shifted_right = torch.cat([x[:, 1:], x[:, 0][:,None,:]], dim=1) # t=1,2,3 | t = 3  ==> 1 2 3 3
        # shifted_right[0,:,0] tensor([1., 2., 3., 0.], device='cuda:0', grad_fn=<SelectBackward0>)
        
        # Compute central difference derivative
        central_diff = torch.abs(shifted_right - shifted_left) / 2.0
        # C = torch.mean(torch.mean(torch.mean(central_diff, 1),1))
        C = central_diff.mean(1).mean(1).mean()
        
        # V = torch.mean(torch.mean(b, 1))
        V = x.var(1).mean(1).mean()
        
        L = C+V
        loss = 1/(L+delta)
        if self.debug:print(f" Loss : {loss.item()} C: {C.item()} V: {V.item()}")
        return loss


class KL_Loss():
    def __init__(self,tau=0.8, lambda1=0.64):
        self.tau = tau
        self.lambda1 = lambda1
        
    def single_element_loss(self, teacher, student, reduction=None):
        distillation_loss = F.kl_div(
            F.log_softmax(student / self.tau, dim=-1),
            F.log_softmax(teacher / self.tau, dim=-1),
            reduction=reduction, log_target=True ) * (self.tau * self.tau) / student.numel()
        return self.lambda1 * distillation_loss

    def __call__(self, teacher, student, reduction="sum", **misc):
        return self.single_element_loss(teacher, student, reduction=reduction)
        
class MSE():
    def __init__(self, mode="l2", use_square=True, mean=True):
        self.use_square = use_square
        self.mode = mode
        if mode == None:
            self.mode = "l2"
        self.mean = mean

    def __call__(self, x, y):
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)
        if self.mode == "l2":
            diff = x - y 
            diff = (diff ** 2).sum(-1)
            if self.mean:
                if self.use_square:
                    return diff.mean()
                else:
                    return (diff ** 0.5).mean()    
            else:
                if self.use_square:
                    return diff
                else:
                    return (diff ** 0.5)
        elif self.mode == "cosine":
            cosine = x * y 
            cosine = (cosine).sum(-1)
            diff = 1 - cosine
            if self.mean:
                return diff.mean()    
            else:
                return diff

class Cosine_Disentangle():
    def __init__(self, ):
        self.cosine_product = nn.CosineSimilarity(-1)
        self.rectifier = nn.ReLU() 

    def __call__(self, x, y):
        # x.y --> 0 ==> x || y  90, perpendicular
        loss = self.cosine_product(x, y)
        loss = loss.abs().mean()
        # loss = self.rectifier(loss).mean()
        return loss 
        

class Cosine_Similarity():
    def __init__(self, ):
        self.cosine_product = nn.CosineSimilarity(-1)
        self.rectifier = nn.ReLU() 

    def __call__(self, x, y):
        # x.y --> 0 ==> x || y  90, perpendicular
        loss = self.cosine_product(x, y)
        loss = (1 - loss.abs()).mean()
        # loss = self.rectifier(loss).mean()
        return loss 
        

            