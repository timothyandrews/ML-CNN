import sys
import iris
import numpy as np
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
from iris.cube import Cube
import iris.plot as iplt
import iris.quickplot as qplt
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.colors as mc
import iris.coord_categorisation
from iris.time import PartialDateTime
sys.path.append('../ML_Model/')
from cnn import ConvNet
import os


def global_mean(cube):
  xxx=cube.copy()
  xxx.coord('longitude').bounds=None
  xxx.coord('latitude').bounds=None
  xxx.coord('longitude').guess_bounds()
  xxx.coord('latitude').guess_bounds()
  grid_areas=iris.analysis.cartography.area_weights(xxx)
  mean=xxx.collapsed(['latitude','longitude'],iris.analysis.MEAN,weights=grid_areas)
  return mean

def monthly2annual(cube):
  iris.coord_categorisation.add_year(cube,'time',name='year')
  cube=cube.aggregated_by('year',iris.analysis.MEAN)
  return cube

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

# Load in the radiation min and max to rescale ML model output
# and the T validation data to pass through ML model.
N_min=np.load('../Output/'+var+'/Saves/y_min.npy')
N_max=np.load('../Output/'+var+'/Saves/y_max.npy')
T_val_tensor=torch.load('../Output/'+var+'/Saves/x_val_tensor.t')

# Load in validation data and get coord info to make
# cube with later
all_data = iris.load_cube(indir+infile,var)
constraint_time=iris.Constraint(time=lambda cell: \
     PartialDateTime(year=xstartyr,month=1,day=1) <= cell <= \
     PartialDateTime(year=xendyr,month=12,day=30))
all_data=all_data.extract(constraint_time)
nsplit=np.where(all_data.coord('year').points == xsplityr)[0][0] # where the split element occurs
val_data=all_data[nsplit:]

val_data.remove_coord('year')
time_coord = val_data.coord('time')
lats_coord = val_data.coord('latitude')
lons_coord = val_data.coord('longitude')


#################################################
# DEFINE ML MODEL
#################################################

# Create an instance of the ConvNet class
model = ConvNet()

# Load pre trained model
model=torch.load('../Output/'+var+'/Saves/saved_ML_model')

###################################################
### RUN THE VALIDATION TEMPERATURE THROUGH ML MODEL
###################################################

# Put the validation data through the ML model (which requries tensor arrays)
# to get prediction output
N_val_pred=model(T_val_tensor)

# Convert output to numpy array and rescale and turn into cube.
N_val_pred=N_val_pred.detach().numpy()
N_val_pred_rescaled = (N_val_pred * (N_max - N_min)) + N_min
ML_prediction=Cube(N_val_pred_rescaled[:,0], dim_coords_and_dims=[(time_coord, 0),(lats_coord, 1),(lons_coord, 2)])

###################################################
### PLOT GLOBAL MEAN COMPARISON
###################################################

xval_data=global_mean(val_data).data
xml_pred=global_mean(ML_prediction).data
plt.figure()
plt.plot(xval_data, label = 'validation data')
plt.plot(xml_pred, label = 'ML model')
plt.legend()
plt.xlabel('Year from 1971')
plt.ylabel('Net TOA radiation (W/m^2)')
plt.title('')
plt.savefig('../Output/'+var+'/Figures/Fig_Validation_'+var+'.png')


zval_data=monthly2annual(val_data)
zML_pred=monthly2annual(ML_prediction)
xval_data=global_mean(zval_data).data
xml_pred=global_mean(zML_pred).data
plt.figure()
plt.plot(xval_data, label = 'validation data')
plt.plot(xml_pred, label = 'ML model')
plt.legend()
plt.xlabel('Year from 1971')
plt.ylabel('Net TOA radiation (W/m^2)')
plt.title('')
plt.savefig('../Output/'+var+'/Figures/Fig_Validation_Annual_'+var+'.png')


###################################################
### PLOT REGIONAL COMPARISON
###################################################

def make_cmap(colors, position=None, bit=False, N=256):
    '''
    make_cmap takes a list of tuples which contain RGB values. The RGB
    values may either be in 8-bit [0 to 255] (in which bit must be set to
    True when called) or arithmetic [0 to 1] (default). make_cmap returns
    a cmap with equally spaced colors.
    Arrange your tuples so that the first color is the lowest value for the
    colorbar and the last is the highest.
    position contains values from 0 to 1 to dictate the location of each color.
    '''
    bit_rgb = np.linspace(0,1,256)
    if position == None:
        position = np.linspace(0,1,len(colors))
    else:
        if len(position) != len(colors):
            sys.exit("position length must be the same as colors")
        elif position[0] != 0 or position[-1] != 1:
            sys.exit("position must start with 0 and end with 1")
    if bit:
        for i in range(len(colors)):
            colors[i] = (bit_rgb[colors[i][0]],
                         bit_rgb[colors[i][1]],
                         bit_rgb[colors[i][2]])
    cdict = {'red':[], 'green':[], 'blue':[]}
    for pos, color in zip(position, colors):
        cdict['red'].append((pos, color[0], color[0]))
        cdict['green'].append((pos, color[1], color[1]))
        cdict['blue'].append((pos, color[2], color[2]))

    cmap = mpl.colors.LinearSegmentedColormap('my_colormap',cdict,N)
    return cmap

def plot_row(row_number,cube1,cube2,cube1_name,cube2_name,var_name,units,scaler):
  if row_number == 1: row=4 # invert rows so they run top to bottom
  if row_number == 2: row=3
  if row_number == 3: row=2
  if row_number == 4: row=1
  plt.rcParams["lines.linewidth"] = 0 # removes lines between contour lines
  
  # Define colour table and levels
  r=[36,   24,  40,  61,  86, 117, 153, 188,   234, 255,   255, 255, 255, 255, 255, 247, 216, 165]
  g=[0,    28,  87, 135, 176, 211, 234, 249,   255, 255,   241, 214, 172, 120,  61,  39,  21,   0]
  b=[216, 247, 255, 255, 255, 255, 255, 255,   255, 234,   188, 153, 117, 86,   61,  53,  47,  33]

  # Do cube 1
  levels=np.array([-9.0,-8.0,-7.0,-6.0,-5.0,-4.0,-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0])*scaler
  cols = list(zip(r,g,b))
  cmap = make_cmap(cols,bit=True,N=len(r))
  cmap.set_under([0,0,0.5]) # set lower out of bound colours [r,g,b; 0:1]
  cmap.set_over([0.5,0,0]) # set upper out of bound colours [r,g,b; 0:1]
  norm_levs=mc.BoundaryNorm(levels[:], cmap.N)
  cube=cube1
  cube_label=title1
  ax=fig.add_axes([left0+0*h_offset,bot0+row*v_offset,h_width,v_height],projection=ccrs.Robinson(central_longitude=-180.0))
  cube_avg=global_mean(cube).data
  ax.set_title(cube_label,fontsize=8)
  coldata=iplt.contourf(cube,levels,norm=norm_levs,cmap=cmap,extend='both')
  ax.coastlines(lw=0.6)
  cbar=fig.colorbar(coldata,ax=ax,extend='both',orientation='horizontal')
  cbar.set_label(var_name,fontsize=6)
  cbar.set_ticks(levels*2)
  cbar.ax.tick_params(labelsize=6)

  # Do cube 2
  levels=np.array([-9.0,-8.0,-7.0,-6.0,-5.0,-4.0,-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0])*scaler
  cols = list(zip(r,g,b))
  cmap = make_cmap(cols,bit=True,N=len(r))
  cmap.set_under([0,0,0.5]) # set lower out of bound colours [r,g,b; 0:1]
  cmap.set_over([0.5,0,0]) # set upper out of bound colours [r,g,b; 0:1]
  norm_levs=mc.BoundaryNorm(levels[:], cmap.N)
  cube=cube2
  cube_label=title2
  ax=fig.add_axes([left0+1*h_offset,bot0+row*v_offset,h_width,v_height],projection=ccrs.Robinson(central_longitude=-180.0))
  cube_avg=global_mean(cube).data
  ax.set_title(cube_label,fontsize=8)
  coldata=iplt.contourf(cube,levels,norm=norm_levs,cmap=cmap,extend='both')
  ax.coastlines(lw=0.6)
  cbar=fig.colorbar(coldata,ax=ax,extend='both',orientation='horizontal')
  cbar.set_label(var_name,fontsize=6)
  cbar.set_ticks(levels*2)
  cbar.ax.tick_params(labelsize=6)

val_data_mean=val_data.collapsed('time',iris.analysis.MEAN)
cubed_data_mean=ML_prediction.collapsed('time',iris.analysis.MEAN)
fig=plt.figure(figsize=(8,12))
left0 = 0.01
h_offset = 0.32
h_width=0.31
bot0  = 0.01
v_offset = 0.2
v_height = 0.15
units=''
title1='(a) Validation data'
title2='(b) ML prediction'
scaler=1.
if var == 'dN': scaler=10.0
if var == 'dNETcre': scaler=7
plot_row(1,val_data_mean,cubed_data_mean,title1,title2,'Net TOA radiative flux (W/m^2)',units,scaler)
plt.savefig('../Output/'+var+'/Figures/Fig_Validation_Regional_'+var+'.pdf')
