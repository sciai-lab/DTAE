import torch
import torchvision
from speedrun import BaseExperiment, TensorboardMixin, IOMixin
import os

import numpy as np
import pandas as pd

import models
from csv_dataset import CSVDataset
from initialization import Groundtruth
from loss import finetune_loss


class Trainer(BaseExperiment, TensorboardMixin, IOMixin):
    # With this, we tell speedrun that the default function to dispatch (via `run`) is `train`.
    DEFAULT_DISPATCH = 'train'

    def __init__(self):
        super(Trainer, self).__init__()
        # The magic happens here.
        self.auto_setup()
        # Build the module
        self._build()

    def _build(self):
        # Build the data loaders
        self._build_loaders()
        # Build model, optimizer, scheduler and criterion.
                
        self.K = self.get("K",100)
        self.weight_push_pull = self.get("weight_push_pull",1)
        print("Push pull loss weight : ",self.weight_push_pull)
        self.weight_cosine = self.get("weight_cosine",10)
        print("Cosine loss weight : ",self.weight_cosine)
 
        self.groundtruth = Groundtruth(self.dataset[:],n_jobs=16,K=self.K,device=self.device)

        checkpoint_experiment = str(self.get("checkpoint/experiment","template"))
        checkpoint_iteration = str(self.get("checkpoint/iteration","0"))
        checkpoint_path = "./experiments/" +checkpoint_experiment+"/Weights/ckpt_iter_"+checkpoint_iteration+".pt"
        print(f"Loading checkpoint file '{checkpoint_path}' ....")
        self.load_checkpoint(checkpoint_path)

        self.optimizer = getattr(torch.optim, self.get('optimizer/name', 'Adam')) \
            (self.model.parameters(),
             **self.get('optimizer/kwargs', {'lr': 1e-3}))
        self.criterion = torch.nn.MSELoss()

    def _build_loaders(self):
        # Build the dataloaders
        self.dataset_root = self.get("data/root","~/data/export_umap_top5000")
        self.dataset_name = self.dataset_root + "/X.csv"
        self.dataset = CSVDataset(self.dataset_name)
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=len(self.dataset), shuffle=False,num_workers=16)
    
    @property
    def device(self):
        return self.get('device', 'cuda:0')

    def train_epoch(self):
        self.model.train()
        # The progressbar (`self.progress`) is provided courtesy of IOMixin, and is based on tqdm.
        for input in self.progress(self.dataloader, desc='Training',position=1,leave=False):
            # Load tensors to device
            input = input.view(input.size(0), -1)
            input = input.to(self.device)
            # Evaluate loss, backprop and step.
            embedding = self.model.encode(input)
            reconstruction = self.model.decode(embedding)

            push_pull_loss,crispness_loss, cosine_loss = finetune_loss(embedding,self.groundtruth,device=self.device)
            rec_loss = self.criterion(reconstruction, input)
            loss = self.weight_push_pull*(push_pull_loss+crispness_loss)  + rec_loss + self.weight_cosine*cosine_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # Log if required to. `log_scalars_now` and `log_scalar` is brought to you
            # by `TensorboardMixin`.
            if self.log_scalars_now:
                self.log_scalar('training/loss', loss.item())
                self.log_scalar('training/reconstruction_loss', rec_loss.item())
                self.log_scalar('training/push_pull_loss', push_pull_loss.item())
                self.log_scalar('training/crispness_loss', crispness_loss.item())
                self.log_scalar('training/cosine_loss', cosine_loss.item())
            self.next_step()

    def checkpoint(self, force=True):
        save = force or (self.epoch % self.get('training/checkpoint_every', 5) == 0)
        if save:
            info = {
                'epoch': self.epoch,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'config/model/name': self.ckpt["config/model/name"],
                'config/model/kwargs': self.ckpt["config/model/kwargs"]
            }
            # `checkpoint_path` is provided by speedrun, and contains the step count. If you do use
            # it, be sure to increment your setp counter with `self.next_step`.
            torch.save(info, self.checkpoint_path)
    
    def load_checkpoint(self,path):
        self.ckpt = torch.load(path)
        self.model = getattr(models, self.ckpt["config/model/name"])(**self.ckpt["config/model/kwargs"]).to(self.device)
        self.model.load_state_dict(self.ckpt['model'])
        self.predict_embedding("pretrained")
        print("Loaded pretrained model")

    def train(self):
        # The progress bar is provided courtesy of `IOMixin`.
        for epoch_num in self.progress(range(self.get('training/num_epochs', 200)), desc='Epochs',position=0,leave=False):
            self.train_epoch()
            self.checkpoint(False)
            if epoch_num %30 ==0:
                self.predict_embedding(name=str(epoch_num))
            self.next_epoch()
            # The function below is provided by `TensorboardMixin`. It will backup your
            # tensorboard log files as json in the log directory of your experiment.
            self.dump_logs_to_json()
        self.checkpoint()

    def predict_embedding(self, name=""):
        pred_dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=1024, shuffle=False,num_workers=4)
        embeddings = []

        self.model.eval()
        preds = None
        preds_exist = False
        for input in self.progress(pred_dataloader, desc='Inference',position=2,leave=False):
            # Load tensors to device
            input = input.view(input.size(0), -1)
            input = input.to(self.device)
            # Evaluate loss, backprop and step.
            prediction = self.model.encode(input)
            if not preds_exist:
                preds = prediction.cpu().detach().numpy()
                preds_exist = True
            else:
                preds = np.concatenate((preds,prediction.cpu().detach().numpy()))
        
        self.model.train()
        df = pd.DataFrame(preds)
        filename = os.path.join(self.experiment_directory, 'embedding'+name+'.csv')
        df.to_csv(filename,header=False,index=False)    

    def predict_reconstruction(self, name=""):
        pred_dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=1024, shuffle=False,num_workers=4)
        reconstructions = []

        self.model.eval()
        preds = None
        preds_exist = False
        for input in self.progress(pred_dataloader,position=2, desc='Inference'):
            # Load tensors to device
            input = input.view(input.size(0), -1)
            input = input.to(self.device)
            # Evaluate loss, backprop and step.
            prediction = self.model(input)
            if not preds_exist:
                preds = prediction.cpu().detach().numpy()
                preds_exist = True
            else:
                preds = np.concatenate((preds,prediction.cpu().detach().numpy()))
            
        df = pd.DataFrame(preds)
        filename = os.path.join(self.experiment_directory, 'reconstruction'+name+'.csv')
        df.to_csv(filename,header=False,index=False)    


if __name__ == '__main__':
    # Be sure to call `run` and not `train`. Speedrun knows to map a `run` call to `train` via the
    # `DEFAULT_DISPATCH` attribute.
    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)
    
    trainer = Trainer()
    trainer.run()
    trainer.predict_embedding()
