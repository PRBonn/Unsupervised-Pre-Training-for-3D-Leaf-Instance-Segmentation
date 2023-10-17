import torch
import torch.nn as nn
from models.minkunet import MinkUNet
from models.loss import BarlowTwinsLoss 
from pytorch_lightning.core.lightning import LightningModule
import MinkowskiEngine as ME
import datasets.datasets as data
from utils.eval_pq import compute_pq

class BarlowTwinsModel(LightningModule):

    def __init__(self, hparams: dict): 
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = MinkUNet()
        self.optimizer = self.configure_optimizers()
        self.unsupervised_loss = BarlowTwinsLoss(hparams['train']['lambd'], hparams['train']['weakness'], hparams['train']['loss_sampling_points'], 7) # hparams['train']['graph_knn'])
        self.pq_db = 0.0
        self.pq_j = 0.0
        self.pq_c = 0.0


    def getLoss(self, pnt1, z1):
        loss = self.unsupervised_loss(pnt1, z1)
        return loss

    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        loss = 0.0
        for k in range(len(batch)):
            sparse_tensor = data.numpy_to_sparse_tensor( batch[k][0][:,:3][None,:,:] , batch[k][0][None,:,:])  
            #sparse_tensor2 = data.numpy_to_sparse_tensor( batch[k][1][:,:3][None,:,:] , batch[k][1][None,:,:])  
            y = self.forward(sparse_tensor)
            embs = y.F.detach().cpu()
            embs = torch.nn.functional.normalize(embs,1).numpy()
            res = compute_pq(batch[k][0][:,:3], embs, batch[k][1])
            print(res)
            self.pq_db += res['dbscan']['pq']
            self.pq_j += res['jens']['pq']
            #self.pq_c += res['graphcut']['pq']
            loss += self.getLoss(sparse_tensor, y) #sparse_tensor2, y1, y2)
        if k != 0:
            loss /= k

        self.log('train:loss', loss, prog_bar = True)
        return loss
        
    def training_epoch_end(self, training_step_output):
        total_loss = torch.tensor([ x.get('loss') for x in training_step_output]).mean()
        self.log('epoch:loss', total_loss)
        print(self.pq_db/19)
        print(self.pq_j/19)
        print(self.pq_c/19)

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = self.hparams['train']['lr-weights'])
        return [ self.optimizer ]

    def forward(self, x1):
        return self.model(x1) #, self.model(x2)
