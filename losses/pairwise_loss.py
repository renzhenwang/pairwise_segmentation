# -*- coding: utf-8 -*-

import torch
from torch import nn
import torch.nn.functional as F


class CELoss(nn.Module):
    def __init__(self, weight=None, batch_average=False):
        super(CELoss, self).__init__()
        self.batch_average = batch_average
        self.ce_loss = nn.CrossEntropyLoss(weight)

    def forward(self, output, target):
        if not isinstance(target, torch.LongTensor):
            target = target.long()
        
        n, c, h, w = output.size()
        loss = self.ce_loss(output, target)

        if self.batch_average:
            loss /= n

        return loss


class DiceLoss(nn.Module):
    def __init__(self, **kwargs):
        super(DiceLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, output, target):
        if not isinstance(target, torch.FloatTensor):
            target = target.float()
        if len(output.size())>1:
            output = output.permute(0, 2, 3, 1).contiguous()
            output = F.softmax(output, dim=-1)
            pred = output[..., 1]
            pred = pred.view(-1)
        else:
            output = output.view(-1)
            pred = self.sigmoid(output)

        target = target.view(-1)

        smooth = 1.
        intersection = (pred * target).sum()

        return 1 - ((2. * intersection + smooth) /
                    (pred.sum() + target.sum() + smooth))
        

class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, logits, target):
        if not isinstance(target, torch.FloatTensor):
            target = target.float()
        num = target.size(0)
        pred = self.sigmoid(logits)
        m1 = pred.view(num, -1)
        m2 = target.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + 1) / (m1.sum(1) + m2.sum(1) + 1)
        score = 1 - score.sum() / num
        return score


 
class MultiClassDiceLoss(nn.Module):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
    batch size and C is number of classes
    """
    def __init__(self, weights=None, **kwargs):
        super(MultiClassDiceLoss, self).__init__()
        self.weights = weights
 
    def forward(self, output, target):
        if not isinstance(target, torch.FloatTensor):
            target = target.float()
        # output = torch.Tensor(output)
        # target = torch.LongTensor(target)
        if output.dim()>2:
            output = output.permute(0, 2, 3, 1).contiguous()
            output = output.view(-1, output.size(3))
        
        N = output.shape[0]
        C = output.shape[1]
        # predict = F.softmax(output, dim=1)
        predict = output

        class_mask = output.new(N, C).fill_(0)
        # class_mask = torch.Tensor(class_mask)
        ids = target.view(-1, 1).long()
        class_mask.scatter_(1, ids, 1.)
        # if weights is None:
        # 	weights = torch.ones(C) #uniform weights for all classes
        dice = DiceLoss()
        totalLoss = 0
        
        for i in range(C):
            diceLoss = dice(predict[:,i], class_mask[:,i])
            if self.weights is not None:
                diceLoss *= self.weights[i]
            totalLoss += diceLoss
        
        return totalLoss


class MultiLabelBCELoss(nn.Module):
    def __init__(self, weights=None, **kwargs):
        super(MultiLabelBCELoss, self).__init__()
        self.weights = weights
        self.bce = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, output, target):
        if not isinstance(target, torch.FloatTensor):
            target = target.float()
        target = target.permute(0, 2, 3, 1).contiguous()
        target = target.view(-1,target.size(3))

        output = output.permute(0, 2, 3, 1).contiguous()
        output = output.view(-1, output.size(3))
        
        total_loss = 0
        N,C = output.shape
        for i in range(C):
            loss = self.bce(self.sigmoid(output[:,i]), target[:,i])
            if self.weights is not None:
                loss *= self.weights[i]
            total_loss += loss
        return total_loss 


class MultiLabelDiceLoss(nn.Module):
    def __init__(self, weights=None, **kwargs):
        super(MultiLabelDiceLoss, self).__init__()
        self.weights = weights
        self.dice = DiceLoss()
        
    def forward(self, output, target):
        if not isinstance(target, torch.FloatTensor):
            target = target.float()
        target = target.permute(0, 2, 3, 1).contiguous()
        target = target.view(-1,target.size(3))

        output = output.permute(0, 2, 3, 1).contiguous()
        output = output.view(-1, output.size(3))
        
        total_loss = 0
        N,C = output.shape
        for i in range(C):
            loss = self.dice(output[:,i], target[:,i])
            if self.weights is not None:
                loss *= self.weights[i]
            total_loss += loss
        return total_loss        


# focal loss, borrowed from https://raw.githubusercontent.com/clcarwin/focal_loss_pytorch  
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, output, target):
        if not isinstance(target, torch.FloatTensor):
            target = target.float()
        if output.dim()>2:
            '''
            output = output.view(output.size(0),output.size(1),-1)  # N,C,H,W => N,C,H*W
            output = output.transpose(1,2)    # N,C,H*W => N,H*W,C
            output = output.contiguous().view(-1,output.size(2))   # N,H*W,C => N*H*W,C
            '''
            output = output.permute(0, 2, 3, 1).contiguous()
            output = output.view(-1, output.size(3))
        target = target.view(-1,1)

        logpt = F.log_softmax(output, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        if self.alpha is not None:
            if self.alpha.type()!=output.data.type():
                self.alpha = self.alpha.type_as(output.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()       
  
      
class PairwiseCompose(nn.Module):
    def __init__(self, base_loss, fusion_loss, loss_weights = [1.,1.,1.], 
                 fusion_weights=None, fusion_name = None, is_soft=False, **kwargs):
        super(PairwiseCompose, self).__init__()
        
        print('base_loss is: '+base_loss)
        print('fusion_loss is: '+fusion_loss)
        print('pairwise loss weights: ', loss_weights)
        self.br1_weight = loss_weights[0]
        self.br2_weight = loss_weights[1]
        self.f_weight = loss_weights[2]
        
        if base_loss == 'binary_dice':
            self.base_loss = DiceLoss() if not is_soft else SoftDiceLoss()
            if is_soft:
                print('bass_loss is soft dice')
        elif base_loss == 'multi_class_dice':
            self.base_loss = MultiClassDiceLoss()
        elif base_loss == 'multi_label_dice':
            self.base_loss = MultiLabelDiceLoss()
        elif base_loss == 'cross_entropy':
            self.base_loss= CELoss()
        elif base_loss == 'focal_loss':
            self.base_loss = FocalLoss()
        else:
            raise NotImplementedError
            
        if fusion_loss == 'binary_dice':
            self.fusion_loss = DiceLoss() if not is_soft else SoftDiceLoss()
            if is_soft:
                print('fusion_loss is soft dice')
        elif fusion_loss == 'multi_class_dice':
            self.fusion_loss = MultiClassDiceLoss(fusion_weights)
        elif fusion_loss == 'multi_label_dice':
            self.fusion_loss = MultiLabelDiceLoss()
        elif fusion_loss == 'cross_entropy':
            self.fusion_loss= CELoss()
        elif fusion_loss == 'focal_loss':
            self.fusion_loss = FocalLoss()
        else:
            raise NotImplementedError
            
        
    def forward(self, outs, targets):
        br1_out, br2_out, f_out = outs
        br1_target, br2_target, f_target = targets
        br1_loss = self._br1_loss(br1_out, br1_target) * self.br1_weight
        br2_loss = self._br2_loss(br2_out, br2_target) * self.br2_weight
        f_loss = self._fusion_loss(f_out, f_target) * self.f_weight
        loss = br1_loss + br2_loss + f_loss
        return loss
    
        
    def _br1_loss(self, br1_out, br1_target):
        loss = self.base_loss(br1_out, br1_target)
        return loss
        
    
    def _br2_loss(self, br2_out, br2_target):
        loss = self.base_loss(br2_out, br2_target)
        return loss
     
        
    def _fusion_loss(self, f_out, f_target):
        loss = self.fusion_loss(f_out, f_target)
        return loss

''' 
import time
start = time.time()
if __name__ == "__main__":
    
    output = torch.rand(1,256,256,3)
    
    masks1 = torch.rand(1,256,256,1)
    masks1[masks1<0.5]=0
    masks1[masks1>=0.5]=1
    
    masks2 = torch.rand(1,256,256,1)
    masks2[masks2<0.5]=0
    masks2[masks2>=0.5]=1
    
    masks = masks1+masks2
    
    loss_fun1 = PairwiseCompose(fusion_name='dice')
    loss1 = loss_fun1._fusion_loss(output, masks)
    loss_fun2 = PairwiseCompose(fusion_name='multi_task_dice')
    loss2 = loss_fun2._fusion_loss(output, masks)
    
    print(loss1, loss2)
    
end = time.time()
print(end-start)
'''