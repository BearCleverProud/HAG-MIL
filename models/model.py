
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.model_utils import Attn_Net, Attn_Net_Gated, TransLayer

def create_attention_net(feat_in, sz, dropout, gate):
    fc = [nn.Linear(feat_in, sz[1]), nn.GELU()]
    if dropout:
        fc.append(nn.Dropout(dropout))
    if gate:
        attention_net = Attn_Net_Gated(L=sz[1], D=sz[2], dropout=dropout, n_classes=1)
    else:
        attention_net = Attn_Net(L=sz[1], D=sz[2], dropout=dropout, n_classes=1)
    fc.append(attention_net)
    return nn.Sequential(*fc)

class IAMBlock(nn.Module):
    def __init__(self, in_dim, out_dim, final_dim, size, dropout, gate):
        super(IAMBlock, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU())
        self.attention_net = create_attention_net(out_dim, size, dropout, gate)
        self.layer = TransLayer(dim=out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.layer_map_to_final = nn.Sequential(nn.Linear(out_dim, final_dim), nn.LayerNorm(final_dim), nn.GELU())
        
    def aggregate(self, h, norm_func, attn_net):
        _h = norm_func(h).squeeze()
        A, _h = attn_net(_h)  # NxK    
        A = torch.transpose(A, 1, 0)  # KxN
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        result = (torch.mul(h.squeeze().T, A.squeeze())).sum(dim=1).unsqueeze(0)
        slide_level = norm_func(result)
        return A, A_raw, slide_level, _h
    
    def forward(self, h):
        h = self.fc(h)
        h = self.layer(h)
        A, A_raw, slide_level, _h = self.aggregate(h, self.norm, self.attention_net)
        slide_level = self.layer_map_to_final(slide_level)
        return h, A, A_raw, slide_level, _h


class IATModel(nn.Module):
    def __init__(self, gate=True, dropout=0.25, k_sample=8, n_classes=2, instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, feat_in=1024, hidden_dims=[1024,1536,512,1024]):
        super(IATModel, self).__init__()
        self.n_classes = n_classes
        size = [feat_in, 512, 256]
        self.instance_classifiers = nn.ModuleList([nn.Linear(size[1], 2) for _ in range(n_classes)])
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.subtyping = subtyping
        self.weights = nn.Linear(len(hidden_dims), 1)

        self.layers = []
        self.layers.append(IAMBlock(feat_in, hidden_dims[0], hidden_dims[-1], size, dropout, gate))
        for i in range(len(hidden_dims)-1):
            self.layers.append(IAMBlock(hidden_dims[i], hidden_dims[i+1], hidden_dims[-1], size, dropout, gate))
        self.layers = nn.ModuleList(self.layers)

        self.fc5 = nn.Linear(hidden_dims[-1], self.n_classes)
    
    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length, ), 1, device=device, dtype=torch.long)
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length, ), 0, device=device, dtype=torch.long)
    
    #instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier): 
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits.cpu(), all_targets.cpu())
        return instance_loss.to(device), all_preds, all_targets
    
    #instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits.cpu(), p_targets.cpu())
        return instance_loss.to(device), p_preds, p_targets

    def forward(self, h, label=None, instance_eval=False):

        h = h.unsqueeze(0)
        slide_reps = []
        for i in range(len(self.layers)):
            h, A, A_raw, slide_level, _h = self.layers[i](h)
            slide_reps.append(slide_level)

        h = torch.cat(slide_reps, dim=0)
        h = self.weights(h.T).T

        logits = self.fc5(h) #[B, n_classes]
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim=1)

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1: #in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A, _h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else: #out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A, _h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)
                
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        return logits, Y_prob, Y_hat, A_raw, results_dict