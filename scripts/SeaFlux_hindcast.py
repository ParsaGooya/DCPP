#!/usr/bin/env python



###################################################################

# This script uses saved NN models from SeaFlux_NN.py  
# to make historical, assimilation and hindcast estiamtes
# using bias corrected predictors from CanESM5.

######################### Load Packages ############################


import xesmf as xe
import xarray as xr
import wget
import glob
import numpy as np
import warnings
warnings.filterwarnings('ignore') # don't output warnings
import scipy.io
import tensorflow as tf
from datetime import datetime
from multiprocessing import Pool
import time
from functools import partial
import tensorflow as tf
import traceback


########################### functions ###############################
### remove repeating 1990-2019 seasonal cycle : (you can change the time period)

def deseason(ds):
    clim = ds.sel(time = slice('1990','2019')).groupby('time.month').mean()
    ds2 = ds.copy()
    for i in range(0,len(ds.time),12):

        ds2[i:i+12,:,:] = clim
    return ds - ds2



#### Calculate the mode over time set at each grid cell.

def _mode(*args, **kwargs):
    vals = scipy.stats.mode(*args, **kwargs)
    # only return the mode (discard the count)
    return vals[0].squeeze()

def mode(obj, dim = None):
    # note: apply always moves core dimensions to the end
    # usually axis is simply -1 but scipy's mode function doesn't seem to like that
    # this means that this version will only work for DataArray's (not Datasets)
    assert isinstance(obj, xr.DataArray)
    axis = obj.ndim - 1
    return xr.apply_ufunc(_mode, obj,
                          input_core_dims=[[dim]],
                          kwargs={'axis': axis})

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



### define a smoother function as in Landschutzer et al. (2016) using a 3x3 lat-lon moving mean window. This is applied to NN outputs to remove sharp biome edges.

def smoother(ds):

    from scipy.ndimage import convolve

    kernel_lat = np.ones((1, 3, 1)) / 3

    # Apply the moving average filter in latitude using convolution
    smoothed_data_lat = convolve(ds, kernel_lat, mode='constant')

    # Create the filter kernel for longitude
    kernel_lon = np.ones((1, 1, 3)) / 3

    # Apply the moving average filter in longitude using convolution
    smoothed_data_lon = convolve(smoothed_data_lat, kernel_lon, mode='constant')

    # Print the smoothed data in latitude and longitude
    return smoothed_data_lon


### prepare predictors at each biome. 

def biome_predictor(predictors, b, CHL):
    
    if CHL:
        
        obs_biome = predictors.where(biomes == b).stack(x = ['time','lat','lon']).dropna(dim = 'x') ## sample predictors in each biome and stack them into 1-D data
        out = np.c_[ obs_biome.atm_anom.values,  obs_biome.chl_anom.values, obs_biome.sss_anom.values , obs_biome.sst_anom.values, obs_biome.wind_anom.values] ## put 1D predictor data into a large matrix where columns are predictors
        return out, xr.full_like(obs_biome.atm, fill_value = np.nan)
    
    else:
        
        obs_biome = predictors.drop(['chl','chl_anom']).where(biomes == b).stack(x = ['time','lat','lon']).dropna(dim = 'x')

        out = np.c_[ obs_biome.atm_anom.values,  obs_biome.sss_anom.values ,obs_biome.sst_anom.values, obs_biome.wind_anom.values]
        return out, xr.full_like(obs_biome.atm, fill_value = np.nan) 
    

### make predictions at a given biome b using predictors and the NN saved models. 

def predict_NN(predictors, b, CHL):
    
    if CHL:
        try:
        
            print(f' biome = {b} chl')
            
            inp, o_chl = biome_predictor(predictors, b, CHL)  ### load predictors at a given biome
            out = models_chl[b].predict(inp) ### use the model trained in that biome to make prediction
            o_chl[:] = out[:,0] ### Add appropriate coordinates
            return o_chl.unstack()  ### Change back to 3-D data
        
        except Exception as e:
            # Print the error message and traceback
            print(f"Error occurred in train_model for biome {b}:")
            print(traceback.format_exc())  # Print the traceback
            raise e 
    
    else:
        
        try:
            print(f' biome = {b} no chl')
            
            inp,o_no_chl = biome_predictor(predictors, b, CHL)
            out = models_no_chl[b].predict(inp)
            o_no_chl[:] = out[:,0]
            return o_no_chl.unstack()
        
        except Exception as e:
                # Print the error message and traceback
                print(f"Error occurred in train_model for biome {b}:")
                print(traceback.format_exc())  # Print the traceback
                raise e 


### make predictions at a given leadyear ly using predictors and the saved NN models. 

def Hindcast(ly, target):
    
    print( f'started ly{ly}')
    for i in range(1,17):  ### Iterate over biomes
    
        o_chl = predict_NN(ds_dict[ly].resample(time = 'Y').mean(), i, True)  ### Predict at that biome with CHL
        o_no_chl = predict_NN(ds_dict[ly].resample(time = 'Y').mean(), i, False)   ### Predict at that biome without CHL
 

        if i == 1:
            ds = o_chl.unstack().combine_first(o_no_chl.unstack()) ### Combine the predictions with priority given to the model with CHL. 
                                                                    ### Note that there are very limited grid cells where annual CHL estiamtes are not available
                                                                    ### unlike monthly means. The final results is bascially identical to the model with CHL.

         ### Combine biomes together one by one
        else:

            ds1 = o_chl.unstack().combine_first(o_no_chl.unstack())
            ds = ds.combine_first(ds1)
    

    ### Creat an output with the same coverage as the predictant
    hindcast = ds.combine_first(xr.full_like(target.resample(time = 'Y').mean(), fill_value = np.nan))
    hindcast[:,:,:] = smoother(hindcast)
    

    return hindcast
    



################################ load data #################################
#### See SeaFlux_Linear.py for detailed notes.
#### Load biomes
biomes = xr.open_dataset('~/SOM_4_4_biomes.nc').biomes
biomes = mode(biomes, dim = 'time')
### obseravational predictors. 

atm = xr.open_dataset('~/atm_historical_obs_1972-2020.nc').rename({'__xarray_dataarray_variable__' : 'xCO2atm'}).xCO2atm.sel(time = slice(f'{1990}',f'{2019}'))
obs = xr.open_dataset('~/obs_predictors.nc').drop('mld').sel(time = slice(f'{1990}',f'{2019}'))

wind = xr.open_dataset('~/wind_obs_1982-2020.nc').wind.sel(time = slice(f'{1990}',f'{2019}'))
obs = xr.combine_by_coords([obs, wind, deseason(wind).to_dataset(name = 'wind_anom')])
obs['atm_anom'] = deseason(obs.atm)
obs['sst_anom'] = deseason(obs.sst)
obs['sss_anom'] = deseason(obs.sss)
obs['chl_anom'] = deseason(obs.chl)

###################### CanESM5 bias corrected historical predictors #################
hist = xr.open_dataset(f'~/CanESM5_historical_predictors_EM_1980_2020_bias_corrected.nc').sel(time = slice('1990','2019')).drop(['mld'])
hist['sst_anom'] = deseason(hist.sst)
hist['sss_anom'] = deseason(hist.sss)
hist['chl_anom'] = deseason(hist.chl)
wind = xr.open_dataset(f'~/CanESM5_historical_wind_bias_corrected.nc').wind.sel(time = slice(f'{1990}',f'{2019}'))
hist = xr.combine_by_coords([hist, atm.to_dataset(name = 'atm'), deseason(atm).to_dataset(name = 'atm_anom')])
hist = xr.combine_by_coords([hist,  deseason(wind).to_dataset(name = 'wind_anom')])


###################### CanESM5 bias corrected hindcast predictors #################
ds_dict = {}
apco2 =  xr.open_dataset(f'~/CMIP6_atm_forecast_1982-2029.nc').ssp245.sel(time = slice('1990',None))
for i in range(1,11):
    
    ds_dict[i] = xr.open_dataset(f'~/CanESM5_hindcast_predictors_ly{i}_EM_1980_{2019+i}_bias_corrected.nc').sel(time = slice('1990',None)).drop(['mld']) 
    ds_dict[i]['sst_anom'] = deseason(ds_dict[i].sst)
    ds_dict[i]['sss_anom'] = deseason(ds_dict[i].sss)
    ds_dict[i]['chl_anom'] = deseason(ds_dict[i].chl)
    ds_dict[i] = xr.combine_by_coords([ds_dict[i],apco2.sel(time = slice('1990',f'{2019+i}')).to_dataset(name = 'atm'), deseason(apco2.sel(time = slice('1990',f'{2019+i}'))).to_dataset(name = 'atm_anom')])
    ds_dict[i] = xr.combine_by_coords([ds_dict[i],  deseason(ds_dict[i].wind).to_dataset(name = 'wind_anom')])



###################### CanESM5 assimilation predictors #################
assim = xr.open_dataset('~/CanESM5_assim_predictors_r1i1p2f1_1980-2020_bias_corrected.nc').sel(time = slice('1990','2019')).drop(['mld'])
assim['sst_anom'] = deseason(assim.sst)
assim['sss_anom'] = deseason(assim.sss)
assim['chl_anom'] = deseason(assim.chl)
wind = xr.open_dataset(f'~/CanESM5_assim_sfcWind_1980-2020_r1i1p2f1.nc').wind.sel(time = slice('1990','2019'))
assim = xr.combine_by_coords([assim,atm.to_dataset(name = 'atm'), deseason(atm).to_dataset(name = 'atm_anom')])
assim = xr.combine_by_coords([assim,  deseason(wind).to_dataset(name = 'wind_anom')])

##################### Load SeaFlux observation based product for ocean carbon flux as predictant ##############
sink =  xr.open_dataset('~/SeaFlux_v2021.04_fgco2_all_winds_products.nc?download=1', engine = 'netcdf4').sel(wind = 'ERA5').fgco2
sink['time'] = obs['time']

########################## use the saved models to make estimates ##############
### Iterate over each of the six SeaFlux data products, load NN models, and make reconstruction (using obsearvational predictors) as well as 
### historical, assimilation and hindcast predictions using CanESM predictors. 

from keras.models import load_model

for prd in sink.product:

    target = deseason(sink.sel(product = prd)).resample(time = 'Y').mean()  ### Select the SeaFlux product
    models_chl = {}
    models_no_chl = {}

    ### Load the models trained based on "prd" as predictant at each biome
    for i in range(1,17):
        models_chl[i] = load_model(f"~/NN_models/SeaFlux/{prd}/models/{prd}_annual_SOMFFN_sigmoid15_biome{i}_chl.h5") 
        models_no_chl[i] =  load_model(f"~/NN_models/SeaFlux/{prd}/models/{prd}_annual_SOMFFN_sigmoid15_biome{i}_no_chl.h5")


    #### Reconstruct "prd" using the NN model and observational predictors

    for i in range(1,17):
        
        o_chl = predict_NN(obs.resample(time = 'Y').mean(), i, True)  ### predict With CHL
        o_no_chl = predict_NN(obs.resample(time = 'Y').mean(), i, False)  ### predict Without CHL
        
        
        if i == 1:
            ds = o_chl.unstack().combine_first(o_no_chl.unstack()) ## unstack to 3-D data and combine with priority given to the model with CHL.

        ## Combine biomes one by one
        else:
            ds1 = o_chl.unstack().combine_first(o_no_chl.unstack())
            ds = ds.combine_first(ds1) ## Combines each biome with the prior
        
      
    recons = ds.combine_first(xr.full_like(sink.sel(product = prd).resample(time = 'Y').mean(), fill_value = np.nan))   ### Creat an output with the same coverage as "prd"
    recons[:,:,:] = smoother(recons) ### Smooth the output data in lat x lon space to remove sharpe biome edges
    recons.to_dataset(name = 'fgco2').to_netcdf(f'~/NN_models/SeaFlux/{prd}/Output/recons_{prd}_annual_SOMFFN_sigmoid15.nc') ### Save

    ######### Use the NN models to make predictions using CanESM5 historical predictors 

    for i in range(1,17):

        o_chl = predict_NN(hist.resample(time = 'Y').mean(), i, True)
        o_no_chl = predict_NN(hist.resample(time = 'Y').mean(), i, False)
        
        if i == 1:
            ds = o_chl.unstack().combine_first(o_no_chl.unstack())

        ## Combine biomes one by one
        else:
            ds1 = o_chl.unstack().combine_first(o_no_chl.unstack())
            ds = ds.combine_first(ds1)
            
    historical = ds.combine_first(xr.full_like(sink.sel(product = prd).resample(time = 'Y').mean(), fill_value = np.nan))
    historical[:,:,:] = smoother(historical)                 
    historical.to_dataset(name = 'fgco2').to_netcdf(f'~/NN_models/SeaFlux/{prd}/Output/hist_{prd}_annual_SOMFFN_sigmoid15.nc')

    ######### Use the NN models to make predictions using CanESM5 hindcast predictors 
    
    hindcast_dict = {}
    for i in range(1,11):
        
        hindcast_dict[i] = Hindcast(i, sink.sel(product = prd))  ## Call hindcast function at each lead year
        hindcast_dict[i].to_dataset(name = 'fgco2').to_netcdf(f"~/NN_models/SeaFlux/{prd}/Output/hindcast_{prd}_annual_SOMFFN_sigmoid15_ly{i+1}_no_chl.nc")


    ######### Use the NN models to make predictions using CanESM5 assimilation predictors 
    for i in range(1,17):
        
        o_chl = predict_NN(assim.resample(time = 'Y').mean(), i, True)
        o_no_chl = predict_NN(assim.resample(time = 'Y').mean(), i, False)
        
        if i == 1:
            ds = o_chl.unstack().combine_first(o_no_chl.unstack())
        else:
            ds1 = o_chl.unstack().combine_first(o_no_chl.unstack())
            ds = ds.combine_first(ds1)
            
    assimilation = ds.combine_first(xr.full_like(sink.sel(product = prd).resample(time = 'Y').mean(), fill_value = np.nan))
    assimilation[:,:,:] = smoother(assimilation)               
    assimilation.to_dataset(name = 'fgco2').to_netcdf(f'~/NN_models/SeaFlux/{prd}/Output/assim_{prd}_annual_SOMFFN_sigmoid15.nc')




#### Load the NN models trained on the MEAN and run repeat the same steps


prd = "Mean"
target = deseason(sink.mean('product')).resample(time = 'Y').mean()
models_chl = {}
models_no_chl = {}

for i in range(1,17):
    models_chl[i] = load_model(f"~/NN_models/SeaFlux/{prd}/{prd}_annual_SOMFFN_sigmoid15_biome{i}_chl.h5") 
    models_no_chl[i] =  load_model(f"~/NN_models/SeaFlux/{prd}/{prd}_annual_SOMFFN_sigmoid15_biome{i}_no_chl.h5")



for i in range(1,17):
    
    o_chl = predict_NN(obs.resample(time = 'Y').mean(), i, True)
    o_no_chl = predict_NN(obs.resample(time = 'Y').mean(), i, False)
    
    if i == 1:
        ds = o_chl.unstack().combine_first(o_no_chl.unstack())
    else:
        ds1 = o_chl.unstack().combine_first(o_no_chl.unstack())
        ds = ds.combine_first(ds1)
    
recons = ds.combine_first(xr.full_like(sink.mean('product').resample(time = 'Y').mean(), fill_value = np.nan))
recons[:,:,:] = smoother(recons)
recons.to_dataset(name = 'fgco2').to_netcdf(f'~/NN_models/SeaFlux/{prd}/Output/recons_{prd}_annual_SOMFFN_sigmoid15.nc')


for i in range(1,17):

    o_chl = predict_NN(hist.resample(time = 'Y').mean(), i, True)
    o_no_chl = predict_NN(hist.resample(time = 'Y').mean(), i, False)
    
    if i == 1:
        ds = o_chl.unstack().combine_first(o_no_chl.unstack())
    else:
        ds1 = o_chl.unstack().combine_first(o_no_chl.unstack())
        ds = ds.combine_first(ds1)
        
historical = ds.combine_first(xr.full_like(sink.mean('product').resample(time = 'Y').mean(), fill_value = np.nan))
historical[:,:,:] = smoother(historical)            
historical.to_dataset(name = 'fgco2').to_netcdf(f'~/NN_models/SeaFlux/{prd}/Output/hist_{prd}_annual_SOMFFN_sigmoid15.nc')

hindcast_dict = {}
for i in range(1,11):
    
    hindcast_dict[i] = Hindcast(i, sink.mean('product'))
    hindcast_dict[i].to_dataset(name = 'fgco2').to_netcdf(f"~/NN_models/SeaFlux/{prd}/Output/hindcast_{prd}_annual_SOMFFN_sigmoid15_ly{i+1}_no_chl.nc")

for i in range(1,17):
    
    o_chl = predict_NN(assim.resample(time = 'Y').mean(), i, True)
    o_no_chl = predict_NN(assim.resample(time = 'Y').mean(), i, False)
    
    if i == 1:
        ds = o_chl.unstack().combine_first(o_no_chl.unstack())
    else:
        ds1 = o_chl.unstack().combine_first(o_no_chl.unstack())
        ds = ds.combine_first(ds1)
        
assimilation = ds.combine_first(xr.full_like(sink.mean('product').resample(time = 'Y').mean(), fill_value = np.nan))
assimilation[:,:,:] = smoother(assimilation)             
assimilation.to_dataset(name = 'fgco2').to_netcdf(f'~/NN_models/SeaFlux/{prd}/Output/assim_{prd}_annual_SOMFFN_sigmoid15.nc')

