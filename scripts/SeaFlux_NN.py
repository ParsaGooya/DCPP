#!/usr/bin/env python


###################################################################

# This script trains a NN at biome scale with annual resolutions 
# using SeaFlux products as predictants and a combination of five predictors:
# SST, SSS, CHL, SfcWind, xCO2atm.
# The script trains and saves the models for future predictions.

######################### Load Packages ############################

import xesmf as xe
import xarray as xr
import wget
import glob
import numpy as np
import warnings
warnings.filterwarnings('ignore') # don't output warnings
import scipy.io

########################### functions ###############################

### remove repeating 1990-2019 seasonal cycle : (you can change the time period)
def deseason(ds):
    clim = ds.sel(time = slice('1990','2019')).groupby('time.month').mean()
    ds2 = ds.copy()
    for i in range(0,len(ds.time),12):

        ds2[i:i+12,:,:] = clim
    return ds - ds2
        

### edit xesmf regridded data set coordinates to time x lat x lon :

def coords_edit(ds):
    
    lat = ds.lat.values[:,0]
    lon = ds.lon.values[0,:]
    ds_like = xr. DataArray(ds.values, dims = ds.dims).rename({'x' :'lon', 'y': 'lat'}).assign_coords({'lat': lat, 'lon': lon })

    if 'member' in ds.dims:
        ds_like = ds_like.assign_coords({'member' : ds.member})
    if 'time' in ds.dims:
        ds_like = ds_like.assign_coords({'time' : ds.time})
    return ds_like

#### Calculate the mode of a data set at each grid cell.
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

#### Unstack 3-dimentional predictants and predictors into large 1-D matrix. biome referes to the number of
#### the biome in which training is done. 


def unstack(predictors, predictant, biome, chl = True):
    
    obs_biome = predictors.where(biomes == biome) ## sample predictors in the biome
    predictant_biome = predictant.where(biomes == biome) ## sample predictants in the biome
    predictors = obs_biome.where(~np.isnan(predictant_biome)) ## remove nan values in predictants from predictors
    
    if chl:  ### train with CHL
        
        predictors = predictors.where(~np.isnan(predictors.chl))   ### remove CHL nan values from predictors and predictant
        predictant_biome = predictant_biome.where(~np.isnan(predictors.chl)) 
        predictors = predictors.where(~np.isnan(predictors.sss))  ### remove SSS nan values from predictors and predictant
        predictant_biome = predictant_biome.where(~np.isnan(predictors.sss))
        
        ### Check if predictors and predictants are at the same time/space locations.
        if predictors.stack(x = ['time','lat','lon']).dropna(dim = 'x').x.equals(predictant_biome.stack(x = ['time','lat','lon']).dropna(dim = 'x').x):
            return predictors.stack(x = ['time','lat','lon']).dropna(dim = 'x'), predictant_biome.stack(x = ['time','lat','lon']).dropna(dim = 'x')
        else:
            raise TypeError("Locations of data points do not match")
        
    else: ### train without CHL
        
        predictors  = predictors.drop(['chl','chl_anom'])
        predictors = predictors.where(~np.isnan(predictors.sss))
        predictant_biome = predictant_biome.where(~np.isnan(predictors.sss))
        
        if predictors.stack(x = ['time','lat','lon']).dropna(dim = 'x').x.equals(predictant_biome.stack(x = ['time','lat','lon']).dropna(dim = 'x').x):

            return predictors.stack(x = ['time','lat','lon']).dropna(dim = 'x'), predictant_biome.stack(x = ['time','lat','lon']).dropna(dim = 'x')
        else:
            raise TypeError("Locations of data points do not match")
        

        
############################ the NN model #################################

# import tensorflow as tf
from datetime import datetime
from multiprocessing import Pool
import time
from functools import partial
import tensorflow as tf
import traceback
from sklearn.model_selection import train_test_split

#### The predictor are concatenated together into a large num_samples x num_predictors matrix. returns predictor and predictant matrices.

def prepare_data(pred,target,biome, CHL = True):
    
    x, y = unstack(pred, target, biome, CHL)
    if CHL:
        
        x = np.c_[x.atm_anom.values,  x.chl_anom.values,  x.sss_anom.values , x.sst_anom.values, x.wind_anom.values]
        y = (y.values).reshape(-1,1)
        
    else:
        
        x = np.c_[x.atm_anom.values,  x.sss_anom.values , x.sst_anom.values, x.wind_anom.values]
        y = (y.values).reshape(-1,1)
        
    return x, y   



models_chl = {}
models_no_chl = {}
        
### Train model at a given biome        
def train_model( network_id, CHL):
    

    
    if CHL:
        
        ### prepare predictors and predictants at each biome  
        x,y = prepare_data(obs.resample(time = 'Y').mean(),target,network_id + 1, True)
        input_dim = x.shape[1]
        output_dim = 1

        ### define the dense one layer keras model with 15 neurons (acquired after testing with different number of neurons)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
        model = tf.keras.Sequential([
        tf.keras.layers.Dense(units= 15, activation='sigmoid', input_dim=input_dim),
        tf.keras.layers.Dense(units=output_dim, activation='linear')])


        model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=[tf.keras.metrics.RootMeanSquaredError()])
        
        try:
            ### start training the model in a biome and output the model
            print(f'start training biome = {network_id + 1} with CHL')
            model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=0)
            loss, rmse = model.evaluate(x_test, y_test)
            print(f'model saved biome = {network_id + 1} with CHL')
            return network_id , model
    
        except Exception as e:
            # Print the error message and traceback
            print(f"Error occurred in train_model for network_id {network_id}:")
            print(traceback.format_exc())  # Print the traceback
            raise e 

            
    
    else: ### Same as above without CHL
        

        x,y = prepare_data(obs.resample(time = 'Y').mean(),target,network_id + 1, False)  
        input_dim = x.shape[1]
        output_dim = 1
        


        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
        model = tf.keras.Sequential([
        tf.keras.layers.Dense(units= 15, activation='sigmoid', input_dim=input_dim),
        tf.keras.layers.Dense(units=output_dim, activation='linear')])

    

        model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=[tf.keras.metrics.RootMeanSquaredError()])
        try:
            print(f'start training biome = {network_id + 1} without CHL')
            model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=0)
            loss, rmse = model.evaluate(x_test, y_test)      
            print(f'model saved biome = {network_id + 1} without CHL')
            return network_id , model
        
        except Exception as e:
            # Print the error message and traceback
            print(f"Error occurred in train_model for network_id {network_id}:")
            print(traceback.format_exc())  # Print the traceback
            raise e 
            


### Define a pool for parallel computing.

def pool_map(arguments):
    
    results = pool.starmap(train_model, arguments)
    pool.close()
    pool.join()
    return results


########################################### load data #####################################################
### Load biomes
biomes = xr.open_dataset('/fs/ccchome/acrnrpg/DCPP-local/Linear_regression/SOM_4_4_biomes.nc').biomes
biomes = mode(biomes, dim = 'time')
### Obseravational predictors
atm = xr.open_dataset('/fs/ccchome/acrnrpg/DCPP-local/atm_historical_obs_1972-2020.nc').rename({'__xarray_dataarray_variable__' : 'xCO2atm'}).xCO2atm.sel(time = slice(f'{1990}',f'{2019}'))
obs = xr.open_dataset('/fs/ccchome/acrnrpg/DCPP-local/Linear_regression/obs_predictors.nc').drop('mld').sel(time = slice(f'{1990}',f'{2019}'))
wind = xr.open_dataset('/fs/ccchome/acrnrpg/DCPP-local/wind/wind_obs_1982-2020.nc').wind.sel(time = slice(f'{1990}',f'{2019}'))
obs = xr.combine_by_coords([obs, wind, deseason(wind).to_dataset(name = 'wind_anom')])
obs['atm_anom'] = deseason(obs.atm)
obs['sst_anom'] = deseason(obs.sst)
obs['sss_anom'] = deseason(obs.sss)
obs['chl_anom'] = deseason(obs.chl)


##################### Load SeaFlux observation based product for ocean carbon flux as predictant ##############
sink =  xr.open_dataset('SeaFlux_v2021.04_fgco2_all_winds_products.nc?download=1', engine = 'netcdf4').sel(wind = 'ERA5').fgco2
sink['time'] = obs['time']

###############################################################################################################


# ['JENA_MLS', 'MPI_SOMFFN', 'CMEMS_FFNN', 'CSIR_ML6', 'JMA_MLR',  'NIES_FNN', 'Mean']

ds = "Mean"  ### Choose ds from the list above
print(f'training for {ds}')


if ds == "Mean":
    target = deseason(sink.mean('product')).resample(time = 'Y').mean() ### annual mean anomaly of predictant
else:
    target = deseason(sink.sel(product = ds)).resample(time = 'Y').mean()

#### Use a pool of processors to train at each biome parallely

arguments  = [(i, True) for i in range(16)] + [(j, False) for j in range(16)]

pool = Pool(processes=16)
start = datetime.now()
results = pool_map(arguments)
print("End Time Map:", (datetime.now() - start).total_seconds())


#### Save each NN model at each biome with and without CHL

for i in range(16):

    try:
        results[i][1].save(f"NN_models/SeaFlux/{ds}/{ds}_annual_SOMFFN_sigmoid15_biome{i+1}_chl.h5") 
    except:
        print(f'biome {i+1} with CHL is empty')
    try:
        results[i+16][1].save(f"NN_models/SeaFlux/{ds}/{ds}_annual_SOMFFN_sigmoid15_biome{i+1}_no_chl.h5")
    except:
        print(f'biome {i+1} without CHL is empty')

print('Done!')


