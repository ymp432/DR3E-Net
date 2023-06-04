# ================================================================================
from tqdm import tqdm
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import gc

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

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
            # self.s = 1.5 if not s else s
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
class Test_Class(param_class_module.Param_Super_Class):

  def __init__(self,configurable_parameters_dict):
    super().__init__(configurable_parameters_dict)

  def test_start(self,expression_pert_df_copied):
      gc.collect()
      torch.cuda.empty_cache()
      
      expression_dataset=LincsDataset(expression_pert_df_copied)
      train_dataloader = DataLoader(expression_dataset, batch_size=5000, shuffle=False)

      model = DenseNet(in_features=977, out_features=32, num_class=self.number_of_perturbagens, growth_rate=16, num_layers=70).cuda()

      # Load trained model checkpoint
      model.load_state_dict(torch.load("./trained_model/ckpt_00078.pt"))
      model.eval()

      with torch.no_grad():

        loss_vals_after_epochs=[]

        predictions=[]
        for i,batch_data in enumerate(train_dataloader):
            x_data=batch_data[:,:-1].float().cuda()
            y_data=np.array(batch_data[:,-1],dtype=int)

            results = model(x_data)

            predictions.append(pd.DataFrame(results.detach().cpu().numpy()))
        predictions_df=pd.concat(predictions)    
        return predictions_df
