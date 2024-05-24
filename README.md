# ML - Convolution Neural Net

This repo contains code to apply a machine learning CNN to climate data. The ML  model is trained on monthly 2D radiative fluxes and surface temperature as simulated by HadGEM3-GC3.1-LL, learning the relationship between these quantities. The seasonal cycle and geographical climatologies of net TOA radiative flux and cloud radiatve effect are then well produced by the ML model from surface temperature inputs alone.

Note this code was developed to aid our personal introduction into ML. We're not experts, and don't really know what we're doing (yet). Bugs are very likely. Use as a learning tool only.

Note also that this code is mostly set up to be used with computing architecture used within the Met Office like SPICE. Outside, you'll have to adapt the scripts to run on your own architecture.

Tim Andrews & Harry Mutton

Met Office Hadley Centre.

May 2024.

_Acknowledgements_: a round of applause to Philip Brohan and Mark Webb for many useful discussions along our ML journey.

## Instructions

### 1. Git

First fork the repo to create your version on the git servers, then clone it to get it onto your local machine.

### 2. Set up environment

Create a `conda` environment from the `yml` file found in the `environment` directory:

```
$ conda env create -f ML_env.yml
```

Activate the `conda` environment:

```
$ conda activate ML_env
```

### 3. Data

Input is a monthly timeseries of 2D (lat,lon) `net TOA radiation`, `cloud radiative effect` and `surface-air-temperature`. These can in principal be changed to whatever you want with a couple of small changes to the code.

* If within the Met Office there are some scripts in the `DataRetrieval` folder that can be sent to SPICE that retrieves data from MASS, then processes the required fields into a single `.nc` file. It is currently set up to retrieve and process surface tempeature and radiation fields from an `amip-piForcing` run with `HadGEM3-GC3.1-LL` covering 1871-2019. Simply execute `$ sbatch toplevel_retrieval.sh` to retrieve the data, then once finished process it with `$ sbatch top_level_process_data.sh`.

* If outsied the Met Office, then you'll need to process this bit yourself, which would be pretty straight forward with CMIP6 data or anything else. In `CMIP6` parlance net TOA radiation is given by `N = rsdt - rlut -rsut` and net cloud radiative effect `NETcre = (rlutcs - rlut) + (rsutcs -rsut)`. Surface-air-temperature is `tas`. `piControl` AOGCM data would work well I suspect.

### 4. Train the ML model

In the `Training` directory execute the top level script:
```
$ ./top_level_run_ML_training.sh
```

 This creates:
 * `../Output/$var/Saves/` directory for saving the trained ML model
 * `../Output/$var/Figures/` directory for loss plot and validation figures
 * `SpiceOutput/` directory for output from SPICE HPC

 It then copies the `slurm.template` and edits it for the calls over our choosen diagnostics, in this case ML model is trained once for `N` and once for `NETcre`, and submits both scripts to SPICE to begin training the ML model. Outside of the Met Office you'll have to figure this bit out yourself to fit your own computing architecture.

 Current setup for training is defined in `Training/train_ML_model.py`:
```
xstartyr=1960
xendyr=2019
xsplityr=2005
nlats=144 # GC3.1
nlons=192 # GC3.1
```
So here ML model is trainined on monthly data from 1960 to 2004. The subequent data, 2005 to 2019, is held back from the model for validation purposes. You can alter these if you wish. Though note if you do, you will need to alter the equivalent lines in `Validation/validate.py` when doing validation. Yes I could write this better, as a single input to both, but I've run out of time on this for now.

The code normalises the data using a simple `max-min` method, then converts it to `tensors` before passing to the ML model for training.

The number of epochs is defined at the following line and can be readily changed:
```
n_epochs = 1000
```
Once the model has trained the output is saved and the loss plot can be found in `../Output/'+var+'/Figures/Fig_loss_plot.png`, where var is `N` or `NETcre`

Note the cnn ML model is defined in `ML_Model/cnn.py`. 

Here is an example loss plot from the training

![Loss Plot](https://github.com/timothyandrews/ML-CNN/blob/main/Fig_loss_plot.png)

### 5. Validation

In the `Validation` directory execute the top level script:
```
$ ./top_level_validate.sh
```

Similar to the training, this copies a slurm template script and edits it for the call to `N` and `NETcre`, sending both to SPICE. The hard work is done in `validate.py`, note if you changed the start/end years or validation year split in the training step, you'll need to change it here as well.

The validation step reads in the input data again, to retrieve the validation data (years 2005-2019) that was held back from training, it then passes the trained ML model the surface temperature fields for this period to generate the predicted radiaitve flux values, then compares to the validation data.

There are a few steps, converting between `tensors` and `numpy arrays`, and undoing the rescaling using the `max-min` normalisation stored in the training step.

The figures from the validation step are sent to `../Output/'+var+'/Figures/`

Here is the validation of the global-monthly mean timeseries and mean geographical cloud radiative effect:

![Seasonal Cyle](https://github.com/timothyandrews/ML-CNN/blob/main/Fig_Validation_dNETcre.png)


![Geographical](https://github.com/timothyandrews/ML-CNN/blob/main/Fig_Validation_Regional_dNETcre.pdf)


### 6. Enjoy

It seems to work well for seasonal cycle and climatology. We've had mixed success on cloud feedback and EEI trends. Part of the issue might be non-linear extrapolation, i.e. under a changing climate with a strong trend we immediately take the ML model into untrained terrority when using the past as training and present as validation.