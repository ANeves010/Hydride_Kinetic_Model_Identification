###Pipeline
import os
import numpy as np
import pytorch_lightning as pl

from torch.utils.data import DataLoader

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from PrePro import TimeDataModule, TestDataset
from NeuralNet import AutoEncoder

import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

methods = ['D1', 'D2', 'D3', 'D4', 'AE08', 'AE09', 'AE1', 'AE15', 'AE2', 'AE3', 'R1', 'R2', 'R3']

def genmodels(num_steps, shape, num_samples, acc, batch_size, max_epoch, drop):
    max_epochs = max_epoch
    
    for m in methods:
        checkpoint_callback = ModelCheckpoint(save_top_k=1,monitor="val_loss",mode="min",dirpath=f'lightning_logs/{m}/',filename=m) #min

        dm = TimeDataModule(m, num_steps=num_steps, num_samples=num_samples, batch_size=batch_size)
        net = AutoEncoder(shape=shape, drop=drop)

        callbacks=[EarlyStopping(monitor="val_loss", mode="min", stopping_threshold=acc, patience=max_epochs), #min
                    checkpoint_callback]

        trainer = pl.Trainer(max_epochs=max_epochs, callbacks=callbacks, enable_progress_bar=True, 
                             accelerator='mps', log_every_n_steps=5, num_sanity_val_steps=0, default_root_dir=f'lightning_logs/{m}')
        trainer.fit(net, dm)

def testmodels(num_steps, dir = "lightning_logs/", prin = False, save = False):
    head = ['File', 'LabelT1','LabelT2', 'Predict', 'TOP1', 'TOP2',
            'ML_R^2_D1', 'm_D1', 'b_D1', 'ML_R^2_D2', 'm_D2', 'b_D2', 'ML_R^2_D3', 'm_D3', 'b_D3', 'ML_R^2_D4', 'm_D4', 'b_D4', 
            'ML_R^2_AE08', 'm_AE08', 'b_AE08', 'ML_R^2_AE09', 'm_AE09', 'b_AE09', 'ML_R^2_AE1', 'm_AE1', 'b_AE1', 'ML_R^2_AE15', 'm_AE15', 'b_AE15', 'ML_R^2_AE2', 'm_AE2', 'b_AE2', 'ML_R^2_AE3', 'm_AE3', 'b_AE3', 
            'ML_R^2_R1', 'm_R1', 'b_R1', 'ML_R^2_R2', 'm_R2', 'b_R2', 'ML_R^2_R3', 'm_R3', 'b_R3',
            'NumTop1', 'NumTop2', 'NumAll',
            'batch', 'reaction', 'T', 'p']
    datastream = []

    #dfR2 = pd.read_csv('Data/R2.csv', delimiter=';', header=None)
    #dfR2.columns = ['File', 'R2']

    models = [AutoEncoder.load_from_checkpoint(f'{dir}{m}/{m}.ckpt').eval().to('mps') for m in methods]

    loader = DataLoader(TestDataset(num_steps=num_steps), shuffle=False, num_workers=0, batch_size=1)
    num_TOP1 = 0
    num_TOP2 = 0
    for num, file in enumerate(loader):
        data, f_dict = file
        #loss = [model._get_reconstruction_loss(data).item() for model in models]
        specs = np.array([linreg(model, data) for model in models])
        r2 = specs[:,0]
        m = specs[:,1]
        b = specs[:,2]

        d_r2 = dict(zip(methods, r2))
        d_m = dict(zip(methods, m))
        d_b = dict(zip(methods, b))
        maximum = max(d_r2, key=d_r2.get)

        correctTOP1 = maximum in f_dict['TOP1'][0] #minimum
        if correctTOP1 == True:
            num_TOP1+=1
        correctTOP2 = maximum in f_dict['TOP2'][0] #minimum
        if correctTOP2 == True:
            num_TOP2+=1
        #anpassen
        datastream.append([f_dict['fname'][0], f_dict['TOP1'][0], f_dict['TOP2'][0], maximum, correctTOP1, correctTOP2, 
                           d_r2['D1'], d_m['D1'], d_b['D1'], d_r2['D2'], d_m['D2'], d_b['D2'], d_r2['D3'],d_m['D3'],d_b['D3'], d_r2['D4'],d_m['D4'],d_b['D4'],
                           d_r2['AE08'], d_m['AE08'], d_b['AE08'], d_r2['AE09'], d_m['AE09'], d_b['AE09'], d_r2['AE1'],d_m['AE1'], d_b['AE1'] , d_r2['AE15'], d_m['AE15'], d_b['AE15'], d_r2['AE2'], d_m['AE2'], d_b['AE2'], d_r2['AE3'], d_m['AE3'], d_b['AE3'],
                           d_r2['R1'], d_m['R1'], d_b['R1'], d_r2['R2'], d_m['R2'], d_b['R2'], d_r2['R3'], d_m['R3'], d_b['R3'],
                           num_TOP1, num_TOP2, num+1,
                           f_dict['batch'][0], f_dict['reaction'][0], f_dict['T'].item(), f_dict['p'].item()]) #f_dict['zyclus'].item()

    df = pd.DataFrame(datastream, columns=head)
    if prin == True:
        print(df)
    
    if save != False:
        df.to_csv(save, index=False)

    return num_TOP1, num_TOP2

def linreg(model, data):
    '''
    Perform linear regression and return r^2, m b
    '''

    y = model(data[0]).detach().cpu().numpy()[0].reshape(-1, 1)
    x = data[0].detach().cpu().numpy()[0].reshape(-1, 1)
    lin_m = LinearRegression()

    lin_m.fit(x,y)
    #predictions = lin_m.predict(x) #r_2 prediction from lin_reg on the basis y
    # Retrieve the offset (intercept) and slope
    offset = lin_m.intercept_[0]
    slope = lin_m.coef_[0][0]

    # Calculate the R^2 value
    r_2 = r2_score(x, y) #predictions/ true, predict of lin reg
    return [r_2, slope, offset]

if __name__ == "__main__": 
    num_steps = 64 #64                                                    
    batch_size = 64                                                       
    num_samples = 2048
    acc = .01
    max_epoch = 500  
    drop = .1                                                     
    shapes = [[num_steps, 64, 32, 32], [32, 32, 64, num_steps]]
          
    
    #genmodels(num_steps=num_steps, shape=shapes, num_samples=num_samples, 
    #         acc=acc, batch_size=batch_size, max_epoch=max_epoch, drop = drop)
    #testmodels(num_steps=num_steps, prin=False, save='240731_result_smoothed.csv') #result_r2-sum

  
