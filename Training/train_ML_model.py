## The model below is designed for the HadGEM3-GC3.1-LL grid. If you are using a different resolution data,
## regrid the data when you load it in

import sys
import iris
import numpy as np
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
import time
from iris.time import PartialDateTime
import subprocess
sys.path.append('../ML_Model/')
from cnn import ConvNet
import os

#################################################
### PREPARE DATA
#################################################

# variable (e.g. dN, dLWcs) choosen by input from shell script
var=sys.argv[1]

# Define start year, end yr and year which splits data into
# training and validation
xstartyr=1960
xendyr=2019
xsplityr=2005
nlats=144 # GC3.1
nlons=192 # GC3.1

DATADIR=os.getenv('DATADIR')
indir=DATADIR+'/ML/data/'
infile='amip-piForcing_u-cw549_HadGEM3-GC31-LL_Monthly.nc'

x_data = iris.load_cube(indir+infile,'dT')
y_data = iris.load_cube(indir+infile,var)
constraint_time=iris.Constraint(time=lambda cell: \
     PartialDateTime(year=xstartyr,month=1,day=1) <= cell <= \
     PartialDateTime(year=xendyr,month=12,day=30))
x_data=x_data.extract(constraint_time)
y_data=y_data.extract(constraint_time)

ntime=len(x_data.data) # number of time elements of data
nsplit=np.where(x_data.coord('year').points == xsplityr)[0][0] # where the split element occurs

# Loose the cube
x_data=x_data.data
y_data=y_data.data

# Create an extra dimension which is used as a 'channel' in ML model
x_data_reshaped = np.zeros((ntime, 1, nlats, nlons))
y_data_reshaped = np.zeros((ntime, 1, nlats, nlons))
for i in range(ntime):
    x_data_reshaped[i, :, :, :] = x_data[i, :, :]
    y_data_reshaped[i, :, :, :] = y_data[i, :, :]

# Reshape and normalise data to between 0 and 1 using max - min scaling
x_min = x_data_reshaped.min()
x_max = x_data_reshaped.max()
y_min = y_data_reshaped.min()
y_max = y_data_reshaped.max()
scaled_x = (x_data_reshaped - x_min) / (x_max - x_min)
scaled_y = (y_data_reshaped - y_min) / (y_max - y_min)

# Split data into training and validation
x_train, x_val = np.split(scaled_x, [nsplit])		# Splits over first dimension which in this case is time	
y_train, y_val = np.split(scaled_y, [nsplit])		# Splits over first dimension which in this case is time

# Convert to tensor arrays
x_train_tensor = torch.Tensor(x_train)
x_val_tensor = torch.Tensor(x_val)
y_train_tensor = torch.Tensor(y_train)
y_val_tensor = torch.Tensor(y_val)

# Save so can ML output can be rescaled back
np.save('../Output/'+var+'/Saves/x_min', x_min)
np.save('../Output/'+var+'/Saves/x_max', x_max)
np.save('../Output/'+var+'/Saves/y_min', y_min)
np.save('../Output/'+var+'/Saves/y_max', y_max)
torch.save(x_train_tensor, '../Output/'+var+'/Saves/x_train_tensor.t')
torch.save(x_val_tensor, '../Output/'+var+'/Saves/x_val_tensor.t')
torch.save(y_train_tensor, '../Output/'+var+'/Saves/y_train_tensor.t')
torch.save(y_val_tensor, '../Output/'+var+'/Saves/y_val_tensor.t')

#################################################
# DEFINE ML MODEL
#################################################

# Create an instance of the ConvNet class
model = ConvNet()

#################################################
# TRAIN MODEL
#################################################

loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = 1000
batch_size = 10

start_time = time.time()
losses = []
losses_val = []
for epoch in range(n_epochs):
    for i in range(0, len(x_train), batch_size):
        Xbatch = x_train_tensor[i:i+batch_size]
        y_pred = model(Xbatch)
        ybatch = y_train_tensor[i:i+batch_size]
        loss = loss_fn(y_pred, ybatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Finished epoch {epoch}, latest loss {loss}')
  
    losses.append(loss.data)

    y_val_pred = model(x_val_tensor)
    loss_val = loss_fn(y_val_pred, y_val_tensor)
    losses_val.append(loss_val.data)
end_time = time.time()



#################################################
# PLOT LOSS FUNCTION
#################################################

plt.figure()
plt.plot(losses, label = 'train')
plt.plot(losses_val, label = 'validation')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss Function')
plt.title('time = {:.02}, min loss = {:.02}, min val loss = {:.02}'.format(end_time - start_time, min(losses), min(losses_val)))
plt.savefig('../Output/'+var+'/Figures/Fig_loss_plot.png')

### You can then save your trained model
torch.save(model, '../Output/'+var+'/Saves/saved_ML_model')
