###data
from torch.utils.data import random_split,  DataLoader, Dataset
import torch
import pytorch_lightning as pl
import numpy as np
import glob

import pandas as pd

class TimeDataModule(pl.LightningDataModule):
    def __init__(self, form: str, batch_size: int = 64, path: str = 'Data/', k_range: list = [1e-4, .1], #[1e-6, .1]
                 alpha_range: list = [.1, .85], num_steps: int = 64, num_samples: int = 2048): #0.85
        """
        Arguments:
            form (str): Gas-Solid Modell ['D1', 'D2', 'D3', 'D4', 'AE08', 'AE09', 'AE1', 'AE15', 'AE2', 'AE3', 'R1', 'R2', 'R3']
            batch_size (int): Size of Batch
            path (str): path to data folder
            k_range (list): [k_min, k_max] range of the material constant
            alpha_range (list): [alpha_min, alpha_max] range of the relative alpha
            num_steps (int): Number of steps to define the alpha interval
            num_samples (int): Number of samples to generate in k_range
        """
        super().__init__()
        self.form = form
        self.batch_size = batch_size
        self.path = path
        self.k_range = k_range
        self.alpha_range = alpha_range
        self.num_steps = num_steps
        self.num_sample = num_samples

    def setup(self, stage = None):
        if stage == "fit" or stage is None:
            train_set_full =  TrainValDataset(form = self.form, k_range= self.k_range, alpha_range = self.alpha_range, 
                                              num_steps = self.num_steps, num_samples = self.num_sample)
            train_set_size = int(len(train_set_full) * 0.8)
            valid_set_size = len(train_set_full) - train_set_size
            self.train, self.validate = random_split(train_set_full, [train_set_size, valid_set_size])
            #transform
            #self.train = PreproDataset(self.train)

        if stage == "test" or stage  == "predict" or stage is None:
            self.test = TestDataset(path = self.path, num_steps=self.num_steps, alpha_range=self.alpha_range)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=7, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=self.batch_size, num_workers=7, persistent_workers=True)

    #def test_dataloader(self):
    #    return DataLoader(self.test, batch_size=1, num_workers=7)

class TrainValDataset(Dataset):
    """Calculate Data on the Fly"""

    def __init__(self, form: str, k_range: list = [.1e-4, .1], alpha_range: list = [.1, .85], #1e-4
                 num_steps: int = 64, num_samples: int = 2048) -> Dataset:
        """
        Arguments:
            form (str): Gas-Solid Modell ['D1', 'D2', 'D3', 'D4', 'AE0.8', 'AE0.9', 'AE1', 'AE1.5', 'AE2', 'AE3', 'R1', 'R2', 'R3']
            k_range (list): [k_min, k_max] range of the material constant
            alpha_range (list): [alpha_min, alpha_max] range of the relative alpha
            num_steps (int): Number of steps to define the alpha interval #512
            num_samples (int): Number of samples to generate in k_range
        """
        self.form = form
        self.alpha = np.linspace(alpha_range[0], alpha_range[1], num_steps)
        self.ks = np.linspace(k_range[0], k_range[1], num_samples)
        self.num_samples = num_samples

        self.one = np.ones(self.alpha.shape[0])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        t_alpha = self.mod(self.ks[idx])
        #sample = {'t': t_alpha, 'label': self.form}
        return t_alpha, self.form

    def mod(self, k):
        """
        Returns t(alpha), regarding to the Gas-Solid-Model.
        """
        #add noise:
        #self.alpha = self.alpha + np.random.normal(0, 0.001, self.alpha.shape[0])
        if self.form == 'D1':
            t_alpha = (self.alpha**2) / k
        
        elif self.form == 'D2':
            t_alpha = (((self.one - self.alpha) * np.log(self.one - self.alpha)) + self.alpha) / k

        elif self.form == 'D3':
            t_alpha = (self.one - (self.one * 2/3 * self.alpha) - (self.one-self.alpha)**(2/3)) / k

        elif self.form == 'D4':
            t_alpha = ((self.one - (self.one-self.alpha)**(1/3))**2) / k

        elif self.form == 'AE08':
            t_alpha = (-np.log(self.one - self.alpha))**(1.25) / k

        elif self.form == 'AE09':
            t_alpha = (-np.log(self.one - self.alpha))**(1/0.9) / k

        elif self.form == 'AE1':
            t_alpha = (-np.log(self.one-self.alpha)) / k

        elif self.form == 'AE15':
            t_alpha = ((-np.log(self.one-self.alpha))**(2/3)) / k
        
        elif self.form == 'AE2':
            t_alpha = ((-np.log(self.one-self.alpha))**(.5)) / k

        elif self.form == 'AE3':
            t_alpha = ((-np.log(self.one-self.alpha))**(1/3)) / k

        elif self.form == 'R1':
            t_alpha = self.alpha / k

        elif self.form == 'R2':
            t_alpha = (self.one - (self.one-self.alpha)**(.5)) / k

        elif self.form == 'R3':
            t_alpha = (self.one - (self.one-self.alpha)**(1/3)) / k

        else:
            print("Please define form as one of ['D1', 'D2', 'D3', 'D4', 'AE08', 'AE09', 'AE1', 'AE15', 'AE2', 'AE3', 'R1', 'R2', 'R3']")
        
        t_norm = np.interp(0.5, self.alpha, t_alpha) #scale with t(0.5)
        t_n = t_alpha / t_norm #normalize with t(alpha = .5)
        t_n = t_n / t_n[-1] #scale to [0..1]

        #t_n = t_alpha / t_alpha[-1]

        return torch.from_numpy(t_n.astype(np.float32)) #torch.from_numpy(t_alpha.astype(np.float32))

class TestDataset(Dataset):
    """Load experimental data for testing"""
    def __init__(self, path: str = 'Data/', num_steps: int = 64, alpha_range: list = [.1, .85])-> Dataset:
        """
        Arguments:
            path (str): path to data folder
            alpha_range (list): [alpha_min, alpha_max] range of the relative alpha
            num_steps (int): Number of steps to define the alpha interval
        """
        self.path = path
        self.num_steps = num_steps
        self.alpha = np.linspace(alpha_range[0], alpha_range[1], num_steps)
        self.data = glob.glob(self.path + "*.csv")

        #self.r2 = pd.read_csv('R2.csv', delimiter=';')

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        file_path = self.data[idx]
        t_alpha, file_dict = self.__prepro(file_path)

        t_alpha_tensor = torch.from_numpy(t_alpha.astype(np.float32))
        t_alpha_tensor = t_alpha_tensor.detach().to("mps")
        #class_id_tensor = torch.tensor([class_id], dtype=torch.float32, device="mps")
        return [[t_alpha_tensor, idx], file_dict] #idx is just a placeholder
    
    def __prepro(self, path):
        '''
        Interpolates and relativates the alphas to get correct t values
        '''
        try:
            t = np.loadtxt(path, delimiter=',', skiprows=1)
        except ValueError:
            print(f'>>>>>>>>> Could not load the file {path}')

        t_alpha = np.interp(self.alpha, t[:,2], t[:,0]) #smoothed:t[:,2], t[:,0] #map self.alpha on measured alpha

        t_norm = np.interp(0.5, self.alpha, t_alpha) #scale with t(0.5)
        t_n = t_alpha / t_norm #normalize with t(alpha = .5)
        t_n = t_n / t_n[-1] #scale to [0..1]
        
        #t_n = t_alpha / t_alpha[-1] 

        top1=''
        top2=''
        #top3=''

        file = path.split("/")[-1]
        p_classes = file.split("_")
        #fname = file.split("_(")[0]

        #rs = self.r2[self.r2['File']==fname]

        file_dict = {'fname': file[:-4], 'batch': p_classes[0], 'reaction': p_classes[1], 'zyclus': str(p_classes[2]),
                     'T': float(p_classes[3][:-4]), 'p': float(p_classes[4][:-3])}

        for clas in ['D1', 'D2', 'D3', 'D4', 
                     'AE08', 'AE09','AE1', 'AE15', 'AE2', 'AE3', 
                     'R1', 'R2', 'R3']:
            
            if clas in p_classes[-2]:
                top1 += clas
            elif clas in p_classes[-1]:
                top2 += clas
            #elif clas in p_classes[-1]:
            #    top3 += clas

            #file_dict['R^2_'+clas] = rs[clas].iloc[0]
        
        file_dict = file_dict | {'TOP1': top1, 'TOP2': top2#,'TOP3': top3
                                 }
        
        return t_n, file_dict #t_alpha
