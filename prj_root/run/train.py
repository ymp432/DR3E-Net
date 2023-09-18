# ================================================================================
from tqdm import tqdm
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from datetime import datetime
import gc

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F

from super_classes import param_class as param_class_module

# ================================================================================
# Source from : https://github.com/cvqluu/Angular-Penalty-Softmax-Losses-Pytorch
class AngularPenaltySMLoss(nn.Module):

    def __init__(self, in_features, out_features, loss_type='cosface', eps=1e-7, s=None, m=None):
        '''
        Angular Penalty Softmax Loss
        Three 'loss_types' available: ['arcface', 'sphereface', 'cosface']
        These losses are described in the following papers: 
        
        ArcFace: https://arxiv.org/abs/1801.07698
        SphereFace: https://arxiv.org/abs/1704.08063
        CosFace/Ad Margin: https://arxiv.org/abs/1801.05599
        '''
        super(AngularPenaltySMLoss, self).__init__()
        loss_type = loss_type.lower()
        assert loss_type in  ['arcface', 'sphereface', 'cosface']
        if loss_type == 'arcface':
            self.s = 64.0 if not s else s
            self.m = 0.5 if not m else m
        if loss_type == 'sphereface':
            self.s = 64.0 if not s else s
            self.m = 1.35 if not m else m
        if loss_type == 'cosface':
            # Scaling factor will be used as trainable parameter in below code
            self.s = 1.5 if not s else s
            # I adjusted these values
            self.m = 0.2 if not m else m

        self.loss_type = loss_type
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.eps = eps

    def forward(self, x, labels, class_weight_to_be_applied, positive_parameter):
        '''
        input shape (N, in_features)
        '''
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features
        
        self.fc.weight.data = F.normalize(self.fc.weight.data, p=2, dim=1)

        x = F.normalize(x, p=2, dim=1)

        wf = self.fc(x)

        if self.loss_type == 'cosface':
            # positive_parameter : Trainable scaling parameter
            numerator = positive_parameter * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        if self.loss_type == 'arcface':
            numerator = self.s * torch.cos(torch.acos(torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1.+self.eps, 1-self.eps)) + self.m)
        if self.loss_type == 'sphereface':
            numerator = self.s * torch.cos(self.m * torch.acos(torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1.+self.eps, 1-self.eps)))

        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(positive_parameter * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L*class_weight_to_be_applied),positive_parameter

# ================================================================================
class LincsDataset(Dataset):
    def __init__(self, processed_data):
        self.img_labels = processed_data

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        return self.img_labels[idx,:]

# ================================================================================
class PositiveParameter(nn.Module):
    def __init__(self):
        super(PositiveParameter, self).__init__()
        self.parameter = nn.Parameter(torch.tensor(1.5))
    
    def forward(self):
        return torch.abs(self.parameter)

# ================================================================================
class DenseLayer(nn.Module):
    def __init__(self, in_features, growth_rate):
        super(DenseLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features, growth_rate),
            nn.ReLU(inplace=True),
            nn.Linear(growth_rate, growth_rate),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        out = self.layer(x)
        out = torch.cat((x, out), dim=1)
        return out

class DenseNet(nn.Module):
    def __init__(self, in_features, out_features, num_class, growth_rate=16, num_layers=30):

        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate
        self.num_layers = num_layers
        
        # Initial transition layer
        self.transition = nn.Sequential(
            nn.Linear(in_features, growth_rate*2),
            nn.ReLU(inplace=True)
        )
        
        # Dense blocks
        self.layers = nn.ModuleList()
        in_features = growth_rate*2
        for i in range(num_layers):
            layer = DenseLayer(in_features, growth_rate)
            self.layers.append(layer)
            in_features += growth_rate
        
        # Final dense layer and output layer
        self.final_dense = nn.Sequential(
            nn.Linear(in_features, growth_rate*2),
            nn.ReLU(inplace=True)
        )
        self.output = nn.Linear(growth_rate*2, out_features)


        self.adm_softmax_loss=AngularPenaltySMLoss(in_features=32, out_features=num_class, loss_type='cosface')

        self.positive_parameter=PositiveParameter()
    
    def forward(self, x):
        # Initial transition layer
        out = self.transition(x)
        
        # Dense blocks
        for i in range(self.num_layers):
            out = self.layers[i](out)
        
        # Final dense layer and output layer
        out = self.final_dense(out)
        out = self.output(out)
        return out

# ================================================================================
class Train_Class(param_class_module.Param_Super_Class):

  def __init__(self,configurable_parameters_dict):
    super().__init__(configurable_parameters_dict)

  def train_start(self,expression_pert_df_copied):

      gc.collect()
      torch.cuda.empty_cache()

      expression_dataset=LincsDataset(expression_pert_df_copied)
      train_dataloader = DataLoader(expression_dataset, batch_size=700, shuffle=True)

      model = DenseNet(in_features=977, out_features=32, num_class=self.number_of_perturbagens, growth_rate=16, num_layers=70).cuda()
      # Number of trainable parameters: 753505  layers 60
      # Number of trainable parameters: 931745  layers 70

      # define learning rates and weight decays for each parameter group
      lr1 = 0.001
      lr2 = 1.0
      wd1 = 0.0
      wd2 = 0.0
      gamma=0.5
      adjust_lr_per_epoch=10
      minimal_lr=0.000001

      # Create parameter groups
      params_to_optimize = []
      params_to_optimize.append({"params": [p for name, p in model.named_parameters() if "positive_parameter" not in name],
                                 "lr": lr1,
                                 "weight_decay": wd1})
      params_to_optimize.append({"params": [p for name, p in model.named_parameters() if "positive_parameter" in name],
                                 "lr": lr2,
                                 "weight_decay": wd2})
      
      optimizer = optim.Adam(params_to_optimize)

      scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10, 10000], gamma=gamma)
      
      loss_vals_after_epochs=[]
      
      all_epoch_loss=[]
      all_epoch_scaling_param=[]
      for one_epoch in tqdm(range(10000)):

        all_batch_loss=[]
        all_batch_scaling_param=[]
        for i,batch_data in enumerate(train_dataloader):
          
          # ================================================================================
          x_data=batch_data[:,:-1].float().cuda()

          # ================================================================================
          y_data_np=np.array(batch_data[:,-1]).astype(int)
          lookup = np.vectorize(lambda x: self.class_weight_dict[x])
          class_weight_to_be_applied=torch.tensor(lookup(y_data_np)).cuda()
          y_data=torch.tensor(batch_data[:,-1]).long().cuda()

          # ================================================================================
          # If data augmentation by adding noise from Normal distribution is needed
          # stddev=0.3
          # noise=torch.from_numpy(np.random.normal(loc=0.0,scale=stddev,size=x_data.size())).float().cuda()
          # # print('noise',noise.shape)
          # # noise torch.Size([5000, 977])
          # noisy_x_data=x_data+noise
          # # print('noisy_x_data',noisy_x_data.shape)
          # # noisy_x_data torch.Size([5000, 977])

          # ================================================================================
          optimizer.zero_grad() 
          
          # ================================================================================
          # Forward to get output
          results = model(x_data)

          # ================================================================================
          err_v,scaling_param=model.adm_softmax_loss(results,y_data,class_weight_to_be_applied,model.positive_parameter())

          all_batch_loss.append(err_v.item())
          all_batch_scaling_param.append(scaling_param.item())

          err_v.backward()

          optimizer.step()


          all_epoch_loss.append(np.array(all_batch_loss).mean())
          all_epoch_scaling_param.append(np.array(all_batch_scaling_param).mean())

        for lr_idx,g in enumerate(optimizer.param_groups):
          if g['initial_lr']==lr1:
            if (one_epoch+1)%adjust_lr_per_epoch==0:
              current_lr=g['lr']
              if current_lr<=minimal_lr:
                pass
              next_lr=current_lr*gamma
              g['lr']=next_lr

        print('one_epoch',one_epoch)
        print('learning_rate g1',optimizer.param_groups[0]['lr'])
        print('learning_rate g2',optimizer.param_groups[1]['lr'])
        print('all_epoch_scaling_param',all_epoch_scaling_param[-1])

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.plot(range(len(all_epoch_loss)), all_epoch_loss)
        plt.savefig('./temp_imgs/{}.png'.format(str(one_epoch).zfill(5)),dpi=100)
        
        torch.save(model.state_dict(), "./trained_model/ckpt_{}.pt".format(str(one_epoch).zfill(5)))
