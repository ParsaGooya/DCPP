## This script contains necessary functions to load CMIP6 data from internal sources,
## organize and save them
## NOTE: all data are downloaded from ESGF using the search_esgf package:
## https://gitlab.com/JamesAnstey/search_esgf

import warnings
warnings.filterwarnings('ignore') # don't output warnings
import os
# import packages
import xarray as xr
# xr.set_options(display_style='html')
import intake
import cftime
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import numpy as np
import xesmf as xe
import pandas as pd
import gcsfs 
import glob
from tqdm import tqdm


############# functions ###############

### edit xesmf regridded dataset coordinates to time x lat x lon :
def coords_edit(ds):
    
    lat = ds.lat.values[:,0]
    lon = ds.lon.values[0,:]
    ds_like = xr. DataArray(ds.values, dims = ds.dims).rename({'x' :'lon', 'y': 'lat'}).assign_coords({'lat': lat, 'lon': lon })

    if 'member' in ds.dims:
        ds_like = ds_like.assign_coords({'member' : ds.member})
    if 'time' in ds.dims:
        ds_like = ds_like.assign_coords({'time' : ds.time})
    return ds_like



### Load data from internal sources: change the "dirr" to the path where your data is stored and adjust the time period as required



### findng the linear fit over a fixed period of time (1980 - 2020 here).
### note: the model data you use might extend beyond 2020. However, the obseravational data used for bias correction only carries over to 2020. 
### So we calculate the trend and mean over the same period where we have observations and extend the trend to the same time length of the model data. 

def poli(data):
  ds = data.sel(time = slice('1982','2020')).copy()  ## select 1982-2020 from model data

  time = np.arange(1982,2021,1/12) 
  T = ds['time']
  ds['time'] = time
  m = ds.polyfit( dim  = 'time', deg = 1).polyfit_coefficients  ## Calculate monthly linear trend over this period

  extend = xr.DataArray(np.arange(1982,1982+len(data['time'])/12,1/12), dims = 'time') 
    
  extended_trend = m[0]*extend +  m[1]
  extended_trend['time'] = data['time']  ## extend the linear trend to the same lebgth of data 

  del ds
  del T
  return m[0], extended_trend 


### This function makes sure all CMIP6 models have the same naming convention for coordinates
def wrapper(ds):
    
    try:
        return ds.rename({'nav_lat':'lat', 'nav_lon':'lon'})
    except:
        try:
            return ds.rename({'latitude':'lat', 'longitude':'lon'})
        except:
            return ds
    
 

######################## load data ###############
## Based on our search through ESGF, only the following models (except CanESM5) had submitted
## hindcast ensembles. 
#  
#  ['CESM1-1-CAM5-CMIP5', 'IPSL-CM6A-LR', 'MPI-ESM1-2-HR', 'NorCPM1']

model = input('model :') ## Choose from above

########  HINDCAST #######

ds_out = xe.util.grid_global(1, 1)
dirr = f'/home/acrnrpg/CMIP6/DCPP/*/{model}/dcppA-hindcast'  ## Change as needed

realization  = []  ## extract the available realizations for the model dcppA-hindcast
for dir in glob.glob(dirr + '/*'):
    name = dir.split('/')[-1]
    realization.append(name.split('-')[-1])

realization = np.unique(realization)
print(realization)


dirr = f'/home/acrnrpg/CMIP6/DCPP/*/{model}/dcppA-hindcast'        
rlm = 'Omon'    
hindcast_dict = {}

## load all realizarions for each year:
for year in tqdm(range(1970,2020)): ## grab all hindcast runs from 10 years before your desired start date (to have lead year 10 prediction)
    dsets = {}
    for r in realization:
        try:
            path = glob.glob(dirr + '/' + f's{year}-' + r + f'/{rlm}/fgco2/gn/*/*nc')
            dsets[r] = xr.open_dataset(path[0])
        except:
            print(r,year)
    try:        
        hindcast_dict[year] = xr.concat([ds for key, ds in dsets.items()], dim = 'member').assign_coords(member = list(dsets.keys()) ).mean('member') # remove .mean('member') if you need large ensemble instead of ensemble means
    except:
        pass

### resampling hindcast data based on target period of prediction and lead year:

ly_dict = {}

for leadtime in tqdm(range(1,11)):  ## resample the hindcast data above in a way that we have prediction on lead years 1 to 10 that are 
                                    ## all projecting a target period starting in the same year (1980 here) and continuing as long as we have data.
                                    ## For instance, on lead year 2, we have predictions for 1980-2021. For lead year 10, 1980-2029. 

    ls = [hindcast_dict[ly]['fgco2'].sel(time = slice(f'{ly + leadtime}',f'{ly + leadtime}'))  for ly in range(1980-leadtime,2017)]  ## Here the target period is 1980 to 2019 + lead year
    ds = xr.concat(ls ,  dim = 'time')
    ds = wrapper(ds) ## correct coordinates naming for xesmf regridder
    regridder = xe.Regridder(ds, ds_out, 'bilinear', ignore_degenerate=True, periodic=True)
    ds = coords_edit(regridder(ds)).sel(time = slice('1990','2017'))
    ds = ds.resample(time = 'Y').mean() - ds.mean('time') ## this line is optional. It alculates anomalies.
    ly_dict[leadtime] = ds
    ly_dict[leadtime].to_netcdf(f'Multimodel_Ensemble/{model}_fgco2_hindcast_ly{leadtime}_EM.nc')
    
######## HISTORICAL ######

### we aim to load the same realizataions as hindcasts, thus we extract available hindcast
### simulations:

ds_out = xe.util.grid_global(1, 1)
dirr = f'/home/acrnrpg/CMIP6/DCPP/*/{model}/dcppA-hindcast'

realization  = []
for dir in glob.glob(dirr + '/*'):
    name = dir.split('/')[-1]
    realization.append(name.split('-')[-1])

realization = np.unique(realization)
print(realization)


## Load all realizations from historical runs
dirr = f'/home/acrnrpg/CMIP6/DCPP/*/{model}/historical'      ## Change as needed   
rlm = 'Omon'    


hist_dict = {}
for r in tqdm(realization):

        path = glob.glob(dirr + '/' + r + f'/{rlm}/fgco2/gn/*/*nc')
        hist_dict[r] = xr.open_dataset(path[0]).sel(time = slice('1980','2015'))

## Load all realizations from ssp245 to extend historical data 
dirr = f'/home/acrnrpg/CMIP6/DCPP/*/{model}/ssp245'        
rlm = 'Omon'  

ssp245_dict = {}

for r in tqdm(realizations):

    path = glob.glob(dirr + '/' + r + f'/{rlm}/fgco2/gn/*/*nc')
    ssp245_dict[r] = xr.open_dataset(path[0]).sel(time = slice('1958','2020'))


### concatenate the ensemble members
ssp245 = xr.concat([item[var] for key, item in ssp245_dict.items()], dim = 'member').assign_coords( member = list(ssp245_dict.keys())).mean('member') ## concatenate all realizations and calculate ensemble mean
hist = xr.concat([item[var] for key, item in hist_dict.items()], dim = 'member').assign_coords( member = list(hist_dict.keys())).mean('member') ## concatenate all realizations and calculate ensemble mean
try:
    hist = hist.drop('depth') 
    ssp245 = ssp245.drop('depth') 

except:
    pass
### concatenate the historical and ssp245 data along 'time' dimention
historical  = xr.concat([hist, ssp245], dim = 'time')

### regrid to a normal 1-1 degree greed
regridder = xe.Regridder(historical, ds_out, 'bilinear', ignore_degenerate=True,  periodic=True)
historical = regridder(historical)
historical = coords_edit(historical).sel(time = slice('1990','2017'))

historical = historical.resample(time = 'Y').mean() - historical.mean('time') ## This line is optional
historical.to_netcdf(f'Multimodel_Ensemble/{model}_fgco2_hist_EM.nc')
        

