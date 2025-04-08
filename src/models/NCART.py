import math
import torch
import torch.nn as nn
from entmax import sparsemax, entmax15
import torch.nn.functional as F
import numpy as np

class CART(nn.Module):
    def __init__(self, n_features, n_selected, n_out, n_trees, mask_type):
        super(CART, self).__init__()
        self.n_features = n_features
        self.n_selected = n_selected
        self.n_trees = n_trees
        self.n_out = n_out
        self.mask_type = mask_type
        self.bn = nn.BatchNorm1d(n_features)

        if n_features > n_selected and n_features > n_out:
            self.projection = True
            self.feature_selection_matrix = nn.Parameter(torch.randn((n_trees,
                                                                      n_features,
                                                                      n_selected),
                                                                     requires_grad=True))
            self.cut_points = nn.Parameter(torch.randn(
                (n_trees, 1, n_selected), requires_grad=True))
            self.linear_layer1 = nn.Parameter(torch.randn(
                (n_trees, n_selected, n_selected), requires_grad=True))
            self.bias1 = nn.Parameter(torch.randn(
                (n_trees, 1, n_selected), requires_grad=True))
            self.linear_layer2 = nn.Parameter(torch.randn(
                (n_trees, n_selected, n_out), requires_grad=True))
            self.bias2 = nn.Parameter(torch.randn(
                (n_trees, 1, n_out), requires_grad=True))
            self.tree_weight = nn.Parameter(torch.randn(
                (n_trees, 1, n_out), requires_grad=True))

        else:
            self.projection = False
            self.cut_points = nn.Parameter(torch.randn(
                (n_trees, 1, n_features), requires_grad=True))
            self.linear_layer1 = nn.Parameter(torch.randn(
                (n_trees, n_features, n_features), requires_grad=True))
            self.bias1 = nn.Parameter(torch.randn(
                (n_trees, 1, n_features), requires_grad=True))
            self.linear_layer2 = nn.Parameter(torch.randn(
                (n_trees, n_features, n_out), requires_grad=True))
            self.bias2 = nn.Parameter(torch.randn(
                (n_trees, 1, n_out), requires_grad=True))
            self.tree_weight = nn.Parameter(torch.randn(
                (n_trees, 1, n_out), requires_grad=True))
            

    def forward(self, x):
        # N:number of trees, B:batch size, D:dim of features, S:dim of selected features
        x = self.bn(x)  # (B, D) -> (B, D)
        
        if self.projection:  # (B, D)*(N, D, S) -> (N, B, S)
            if self.mask_type == 'sparsemax':
                self.projection_matrix = sparsemax(
                    self.feature_selection_matrix, dim=1)
                x = torch.matmul(x, self.projection_matrix)
            elif self.mask_type == 'entmax':
                self.projection_matrix = entmax15(
                    self.feature_selection_matrix, dim=1)
                x = torch.matmul(x, self.projection_matrix)
            else:
                raise NotImplementedError("Method not yet implemented")

        diff = x-self.cut_points
        score = torch.sigmoid(diff)

        # first linear layer for n trees, (N, B, D)*(N, D, D) -> (N, B, D)
        # Z_nbo = sum_k(X_nbk*Y_nko)
        o1 = F.relu(torch.einsum('nbk, nko->nbo', score,
                    self.linear_layer1) + self.bias1)

        # second linear layer for n trees, (N, B, D)*(N, D, O) -> (N, B, O)
        # Z_nbo = sum_k(X_nbk*Y_nko)
        o2 = torch.einsum('nbk, nko->nbo', o1, self.linear_layer2) + self.bias2 

        # merge all trees, (N, B, O)*(N, 1, O) -> (B, O)
        o3 = (o2*self.tree_weight).mean(0)

        return o3
    
    def get_index(self, x):  
        x = self.bn(x)
        if self.projection:
            # previous version, it raises error for multi gpu
            # x = torch.matmul(x, self.projection_matrix)  
            
            if self.mask_type == 'sparsemax':
                projection_matrix = sparsemax(
                    self.feature_selection_matrix, dim=1)
            elif self.mask_type == 'entmax':
                projection_matrix = entmax15(
                    self.feature_selection_matrix, dim=1)
            else:
                raise NotImplementedError("Method not yet implemented")   
            x = torch.matmul(x, projection_matrix)
            
        # shape = (N, A, D) or (N, A, S), where A is the input data number 
        return torch.heaviside((x-self.cut_points).detach().cpu(), torch.tensor([0.0]))
        
    
    def get_gini(self, x):
        num_data = x.shape[0]
        index = self.get_index(x)  # (N, A, D) or (N, A, S)
        if self.projection:
            pos = index.sum(-2, keepdim=True)/num_data   # (N, 1, S)
            neg = 1-pos
            # previous version, it raises error for multi gpu
            # return ((1.-pos**2-neg**2)*self.projection_matrix.detach().cpu()).sum(-1).sum(0) 
             
            if self.mask_type == 'sparsemax':
                projection_matrix = sparsemax(
                self.feature_selection_matrix, dim=1)
            elif self.mask_type == 'entmax':
                projection_matrix = entmax15(
                self.feature_selection_matrix, dim=1)
            else:
                raise NotImplementedError("Method not yet implemented")  
            
            # (N, 1, S)*(N, D, S) -> (N, D, S), -> (N, D) -> (N)
            return ((1.-pos**2-neg**2)*projection_matrix.detach().cpu()).sum(-1).sum(0)  
            
        else:
            pos = index.sum(-2)/num_data   # (N, D)
            neg = 1-pos
            return (1.-pos**2-neg**2).sum(0)  # (D)      


class NCART(nn.Module):
    def __init__(self, configs,):
        super(NCART, self).__init__()
        self.n_layers = configs.n_layers
        self.n_trees = configs.n_trees
        self.n_features = configs.dim_in
        self.n_selected = configs.n_selected
        self.mask_type = configs.mask_type
        self.n_out = configs.n_out

        if self.n_layers > 1:
            self.model_list = nn.ModuleList([CART(self.n_features,
                                                  self.n_selected,
                                                  self.n_features,
                                                  self.n_trees,
                                                  self.mask_type) for _ in range(self.n_layers-1)])
        self.last_layer = CART(self.n_features,
                               self.n_selected,
                               self.n_out,
                               self.n_trees,
                               self.mask_type)

    def forward(self, x):
        x = x.squeeze(1)
        if self.n_layers > 1:
            for m in self.model_list:
                x = F.relu(m(x) + x)
        out = self.last_layer(x)
        return out
    
    def feature_importance(self, x):
        importance = torch.zeros(x.shape[1])
        if self.n_layers > 1:
            for m in self.model_list:
                importance += m.get_gini(x)
        importance += self.last_layer.get_gini(x)
        return self.normalize(importance.numpy())
    
    def normalize(self, x):
        return x/np.sum(x)