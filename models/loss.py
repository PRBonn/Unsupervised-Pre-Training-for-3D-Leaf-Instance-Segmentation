import torch
import torch.nn as nn
import numpy as np 
import random
from utils.graph import Graph

class BarlowTwinsLoss(nn.Module):

    def __init__(self, lambdap, weakness, n_points, graph_knn):
        super(BarlowTwinsLoss, self).__init__()
        self.n_points_max = n_points
        self.lambda_param = lambdap
        self.weakness = weakness
        self.graph_knn = graph_knn

    def forward(self, pnt1, z: torch.Tensor) -> torch.Tensor:
        # normalize repr. along the batch dimension
        torch.cuda.empty_cache()
        weakly_correlated_loss = torch.tensor([0.0]).cuda()

        z_a_norm = torch.nn.functional.normalize(z.F, p=2, dim=1)
                #(z - torch.mean(z.F,0)) / torch.std(z.F,0) # NxD
        #z_b_norm = (z2 - torch.mean(z2.F,0)) / torch.std(z2.F,0) # NxD
        D = z.shape[1]
    
        # enforicing that all points in the same plant should be weakly correlated
        # this is indirectly also the normal barlow twins loss for corresponding points
        # 1. sampling points, to reduce complexity 
        N = min(self.n_points_max, z.F.shape[0]) #, z2.F.shape[0])
        
        idxs_a = torch.zeros(z_a_norm.shape[0], dtype=torch.bool)
        idxs_a[:N] = 1
        idxs_a = idxs_a[torch.randperm(len(idxs_a))]
        
        choice_a = z_a_norm[ idxs_a ] 
        points_1 = pnt1.F[ idxs_a, :3 ]
        
        graph = Graph(points_1, self.graph_knn)
        graph.laplacian_matrix = 1./(graph.laplacian_matrix + 1e-4)
        graph.laplacian_matrix /= np.max(graph.laplacian_matrix)
 
        # 2. compute cross correlation between all pairs of points  
        mat = choice_a @ choice_a.T # N x N 
        # 3. Build the identity matrix for the c_diff computation, extended for the N correlations
        identity_stacked_matrix = torch.ones((N,N)).cuda()
        # we need to put the distances on this shit
        identity_stacked_matrix *= (torch.tensor(graph.laplacian_matrix).cuda())
        
        # 4. As in the normal bt loss, we compute c_diff, with a weakness parameter for weaker correlation
        mat = (mat - identity_stacked_matrix).pow(2)
        mat[(identity_stacked_matrix == 0.)] *= self.lambda_param
    
        # 5. dividing by the number of pairs 
        return mat.sum() / (N**2)
