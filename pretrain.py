import torch
from torchsummary import summary
from speedrun import BaseExperiment, TensorboardMixin, IOMixin
import os

import numpy as np
import pandas as pd

import models
from csv_dataset import CSVDataset

class TCellTrainer(BaseExperiment, TensorboardMixin, IOMixin):
    # With this, we tell speedrun that the default function to dispatch (via `run`) is `train`.
    DEFAULT_DISPATCH = 'train'

    def __init__(self):
        super(TCellTrainer, self).__init__()
        # The magic happens here.
        self.auto_setup()
        # Build the module
        self._build()

    def _build(self):
        # Build the data loaders
        self._build_loaders()
        # Build model, optimizer, scheduler and criterion.
        
        self.random_seed=self.get('random_seed',0)
        torch.random.manual_seed(self.random_seed)
        torch.cuda.manual_seed(self.random_seed)   
        
        self.model = getattr(models, self.get('model/name', 'AE')) \
            (**self.get('model/kwargs', {})).to(self.device)
        self.optimizer = getattr(torch.optim, self.get('optimizer/name', 'Adam')) \
            (self.model.parameters(),
             **self.get('optimizer/kwargs', {'lr': 1e-3}))
        self.criterion = torch.nn.MSELoss()

        self.loss = []

    def _build_loaders(self):
        # Build the dataloaders

        self.dataset_root = self.get("data/root","~/data/export_umap_top5000")
        self.dataset_name = self.dataset_root + "/X.csv"
        print("Loading dataset from "+ self.dataset_name)
        self.dataset = CSVDataset(self.dataset_name)
        
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.get('training/batch_size'), shuffle=True,num_workers=8)
    
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

            loss = self.criterion(reconstruction, input)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # Log if required to. `log_scalars_now` and `log_scalar` is brought to you
            # by `TensorboardMixin`.
            if self.log_scalars_now:
                self.log_scalar('training/loss', loss.item())
                self.loss.append(loss.item())
            self.next_step()

    def checkpoint(self, force=True):
        save = force or (self.epoch % self.get('training/checkpoint_every', 5) == 0)
        if save:
            info = {
                'epoch': self.epoch,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'config/model/name': self.get('model/name', 'AE'),
                'config/model/kwargs': self.get('model/kwargs', {})
            }
            # `checkpoint_path` is provided by speedrun, and contains the step count. If you do use
            # it, be sure to increment your setp counter with `self.next_step`.
            torch.save(info, self.checkpoint_path)
   #         self.predict_embedding(str(self.epoch))

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
    
    trainer = TCellTrainer()
    trainer.run()
    trainer.predict_embedding()
    #trainer.predict_reconstruction()
