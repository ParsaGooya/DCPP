# supress warnings
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



### Load data from internal sources: change the dirr to the path where your data is stored and adjust the time period as required

realm = {'sfcWind' : 'Amon', 'siconc' : 'SImon'}

def grab_data(var, run, realizations):
    
        ds_out = xe.util.grid_global(1, 1)
        if run == 'assim': 
            
            if var in ['sfcWind', 'siconc']:  ## my SfcWind and ice concentration data are stores in a different place than the other predictors. You can remove this
                                              ## if/else clause if you only have one directory.
                
                dirr = '/home/acrnrpg/CMIP6/DCPP/CCCma/CanESM5/dcppA-assim'               
            
            else:
                dirr = '/misc/cccdata/ra40/data/ESGF_DOWNLOADS/CMIP6/DCPP/CCCma/CanESM5/dcppA-assim'
            
            try:
                rlm = realm[var]
            except:
                rlm = 'Omon'
                
            data_dict = {}
            for r in tqdm(realizations):
                path = glob.glob(dirr + '/' + r + f'/{rlm}/{var}/gn/v20190429/*nc') ## Check if assim data are broken into two dataset based on time
                dslist = [xr.open_dataset(dp) for dp in path ]
                data_dict[r] = xr.concat(dslist, dim = 'time')
                
            Assim = xr.concat([item[var] for key, item in data_dict.items()], dim = 'member').assign_coords( member = list(data_dict.keys())).mean('member') ## concatenate all realizations and calculate ensemble mean
            regridder = xe.Regridder(Assim, ds_out, 'bilinear',ignore_degenerate=True,  periodic=True) ## regrid 
            Assim = regridder(Assim).sel(time = slice('1980','2020'))
            Assim = coords_edit(Assim)
            return Assim    

        elif run == "hist":  ##  since historical data carries only over to 2016, we use ssp245 for 2016-2020
            
            if var in ['sfcWind', 'siconc']:

                dirr = '/home/acrnrpg/CMIP6/CMIP/CCCma/CanESM5/historical'
            
            else:
                dirr = '/misc/cccdata/ra40/data/ESGF_DOWNLOADS/CMIP6/CMIP/CCCma/CanESM5/historical'
            
            try:
                rlm = realm[var]
            except:
                rlm = 'Omon'

            hist_dict = {}
            for r in tqdm(realizations):

                    path = glob.glob(dirr + '/' + r + f'/{rlm}/{var}/gn/v20190429/*nc')
                    hist_dict[r] = xr.open_dataset(path[0]).sel(time = slice('1980','2015'))

                
            if var in ['sfcWind', 'siconc']:
               
                dirr = '/home/acrnrpg/CMIP6/ScenarioMIP/CCCma/CanESM5/ssp245'

            else:
                dirr = '/misc/cccdata/ra40/data/ESGF_DOWNLOADS/CMIP6/ScenarioMIP/CCCma/CanESM5/ssp245'
            
            ssp245_dict = {}
            
            for r in tqdm(realizations):
                
                path = glob.glob(dirr + '/' + r + f'/{rlm}/{var}/gn/v20190429/*nc')
                ssp245_dict[r] = xr.open_dataset(path[0]).sel(time = slice('1958','2020'))
                
                
                
            ssp245 = xr.concat([item[var] for key, item in ssp245_dict.items()], dim = 'member').assign_coords( member = list(ssp245_dict.keys())).mean('member') ## concatenate all realizations and calculate ensemble mean
            # remove .mean('member) if you need large ensembles instead of ensemble mean
            hist = xr.concat([item[var] for key, item in hist_dict.items()], dim = 'member').assign_coords( member = list(hist_dict.keys())).mean('member') ## concatenate all realizations and calculate ensemble mean
            try:
                hist = hist.drop('depth') 
                ssp245 = ssp245.drop('depth') 
                
            except:
                pass
                
            historical  = xr.concat([hist, ssp245], dim = 'time')
            

            regridder = xe.Regridder(historical, ds_out, 'bilinear', ignore_degenerate=True,  periodic=True)
            historical = regridder(historical)
            historical = coords_edit(historical)
            return historical

        else: ## for hindcasts, we resample data based on the target prediction year, in a way to have a fixed start for the prediction target period.
            
            if var in ['sfcWind', 'siconc']:
                
                dirr = '/home/acrnrpg/CMIP6/DCPP/CCCma/CanESM5/dcppA-hindcast'
            
            else:
                
                dirr = '/misc/cccdata/ra40/data/ESGF_DOWNLOADS/CMIP6/DCPP/CCCma/CanESM5/dcppA-hindcast'
            
            try:
                rlm = realm[var]
            except:
                rlm = 'Omon'
                
            hindcast_dict = {}
            for year in tqdm(range(1970,2020)): ## grab all hindcast runs from 10 years before your desired start date (to have lead year 10 prediction)
                dsets = {}
                for r in realization:
                    try:
                        path = glob.glob(dirr + '/' + f's{year}-' + r + f'/{rlm}/{var}/gn/v20190429/*nc')
                        dsets[r] = xr.open_dataset(path[0])
                    except:
                        print(r,year)
                try:        
                    hindcast_dict[year] = xr.concat([ds for key, ds in dsets.items()], dim = 'member').assign_coords(member = list(dsets.keys()) ).mean('member') # remove .mean('member') if you need large ensemble instead of ensemble means
                except:
                    pass
            
            
            ly_dict = {}
            for leadtime in tqdm(range(1,11)):  ## resample the hindcast data above in a way that we have prediction on lead years 1 to 10 that are 
                                                ## all projecting a target period starting in the same year (1980 here) and continuing as long as we have data.
                                                ## For instance, on lead year 2, we have predictions for 1980-2021. For lead year 10, 1980-2029. 

                ls = [hindcast_dict[ly][var].sel(time = slice(f'{ly + leadtime}',f'{ly + leadtime}'))  for ly in range(1980-leadtime,2020)]  ## Here the target period is 1980 to 2019 + lead year
                ds = xr.concat(ls ,  dim = 'time')

                regridder = xe.Regridder(ds, ds_out, 'bilinear', ignore_degenerate=True, periodic=True)
                ly_dict[leadtime] = regridder(ds) 
             
            return ly_dict



### findng the linear trend over a fixed period of time (1980 - 2020 here).
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



### remove repeating seasonal cycle over the period where you have observations: (you can change the time period)

def deseason(ds):
    clim = ds.sel(time = slice('1982','2020')).groupby('time.month').mean()
    ds2 = ds.copy()
    for i in range(0,len(ds.time),12):

        ds2[i:i+12,:,:] = clim
    return ds - ds2

### define mean and trend bias adjustment to the observation over the period where we have observation. Note: the bias correction for CHL is different.
def bias_correction(model, obs):
    
    
    ds = obs.copy()
    ds = ds - ds.mean('time') ## calculate anomaly relative to the period where we have obseravations

    time = np.arange(1982,2021,1/12) 
    T = ds['time']
    ds['time'] = time
    m = ds.polyfit( dim  = 'time', deg = 1).polyfit_coefficients
    extend = xr.DataArray(np.arange(1982,1982+len(model['time'])/12,1/12), dims = 'time')  
    
    extended_trend = m[0]*extend +  m[1]
    extended_trend['time'] = model['time'] ## Calculate linear trend based on obsearvation and extend it to the same time as the model data
    
    temp = model - model.sel(time = slice('1982','2020')).mean('time')  ## calculate anomaly relative to the period where we have obseravations for model data
    temp = temp - poli(temp)[1] ## remove a linear trend from model data
    temp = temp + extended_trend + obs.mean('time')  ## replace the linear trend with the linear trend from obsearvation and adjust the mean to the mean of obsearvation
    
    return temp, extended_trend + obs.mean('time')

### define bias correction for CHL. The bias correction is applied on log(CHL) and only for adjusting the mean and not trend.
### Note: observational CHL data only extends back to 1998. For estimates prior to that date, Landschutzer et al (2016)
### uses a repeating seasonal cycle over the period where data is available and extends it back from 1998 to 1982. Since this obsearvational data is used for training
### the NN model, we apply the same procedure to the model data for consistency with what the model is used to seing and acquiring better fits over the historical period.

def chl_edit(ds, obs):
    ds = ds * 1000000  ## change unit for model data to mg/m3 (ds is model data)
    clim = ds.sel(time = slice('1998','2020')).groupby('time.month').mean()  ## Calculate seasonal cycle over 1998-2020 where we have observations
    ds2 = ds.copy()
    for i in range(0,192,12):

        ds2[i:i+12,:,:] = clim  ## replace model data over 1982-1998 with the climatology 
        
        
    ds2 =  np.log(ds2) ## Calculate log(CHL)
    ds2 = ds2.where(ds2 != -np.inf) ## model CHL might have 0 values which will cause -inf values in log(CHL). We remove those here.

    ### A further step is to use the same spatial coverage as the obsearvation for model CHL data. This will keep the estimates with or without CHL consistent per grid cell.
    ref = ds2.copy()
    ref[0:468,:,:] = obs - obs ## For the period where we have observation (1982-2020), we use the same spatial coverage 

    clim = obs.groupby('time.month').mean() - obs.groupby('time.month').mean() ## For future forecasts where we don't observation (2020-2029), 
                                                                               ##  we use a monthly climatology of the obsearvational spatial coverage. 
    for i in range(468,len(ref.time),12): 

        ref[i:i+12,:,:] = clim
    
    
    ds2 = ds2 - ref  ## adjust spatial coverage
  
    chl = ds2  - ds2.sel(time = slice('1982','2020')).mean('time') + obs.mean('time') ##apply mean adjustment to the observation
  
    return  chl


########################### Load CanESM5 predictor data ########################

realization = [f'r{i}i1p2f1' for i in range(1,21)]  ## we use 20 ensemble memners for historical and hindcast runs

parameters = {}
for var in ['tos','sos','chlos','siconc','sfcWind']:  ## download data for each predictor
    
    parameters[var] = grab_data(var, 'dcpp', realization)
    
    
for i in range(1,11): ## resample them based on lead year and target period and add them to a large data set.
    
    ls = [parameters['tos'][i].to_dataset(name = 'tos'),
         parameters['sos'][i].to_dataset(name = 'sos'),
         parameters['chlos'][i].to_dataset(name = 'chlos'),
         parameters['siconc'][i].to_dataset(name = 'siconc').drop('type'),
         parameters['sfcWind'][i].to_dataset(name = 'sfcWind').drop('height')]
    xr.combine_by_coords(ls).to_netcdf(f'CanESM5_hindcast_predictors_ly{i}_EM_1980_{2019+i}.nc')  ## Save hindcast predictors for the target period of 1980 to 2019 + leadtime

### Garb historical data and add them to a data set.

ls = [grab_data('tos', 'hist', realization).to_dataset(name = 'tos'), 
    grab_data('sos', 'hist', realization).to_dataset(name = 'sos'),
    grab_data('chlos', 'hist',  [f'r{i}i1p2f1' for i in range(1,11)]).to_dataset(name = 'chlos'),
    grab_data('siconc', 'hist', realization).to_dataset(name = 'siconc').drop('type'),
    grab_data('sfcWind', 'hist', realization).to_dataset(name = 'sfcWind').drop('height')]

xr.combine_by_coords(ls).to_netcdf(f'CanESM5_historical_predictors_EM_1980_2020.nc')

### NOTE: Do the same to grab and save assimilation predictors. We only have 10 realizations for that.


######################## Bias correction #####################
## Load observational data based on which you want to apply bias correction
obs = xr.open_dataset('/fs/ccchome/acrnrpg/DCPP-local/obs_predictors.nc')
wind_obs = xr.open_dataset('/fs/ccchome/acrnrpg/DCPP-local/wind/wind_obs_1982-2020.nc').wind

## Load hindcast predictors
predictors_hindcast = {}
for i in range(1,11):
    
    predictors_hindcast[i] = xr.open_dataset(f'CanESM5_hindcast_predictors_ly{i}_EM_1980_{2019+i}.nc').sel(time = slice('1982',None))


### apply bias correction for hindcast

bias_corrected_dict = {}
for i in range(1,11):
    
    ds, extend = bias_correction(coords_edit(predictors_hindcast[i].tos), obs.sst) ## SST
    sst = ds.to_dataset(name = 'sst')
    sst_anom = deseason(sst.sst).to_dataset(name = 'sst_anom')
    
    ds, extend = bias_correction(coords_edit(predictors_hindcast[i].sos), obs.sss) ## SSS
    sss = ds.to_dataset(name = 'sss')
    sss_anom = deseason(sss.sss).to_dataset(name = 'sss_anom')
    
    chl, chl_anom = chl_edit(coords_edit(predictors_hindcast[i].chlos), obs.chl) ## CHL
    chl = chl.to_dataset(name = 'chl')
    chl_anom = chl_anom.to_dataset(name = 'chl_anom')

    wind, extend = bias_correction(coords_edit(predictors_hindcast[i].sfcWind **2), wind_obs) ## Wind
    wind = wind.to_dataset(name = 'wind')
    
    ls = [sst,sst_anom,sss,sss_anom, chl, chl_anom, wind] ## combine the data
    xr.combine_by_coords(ls).to_netcdf(f'CanESM5_hindcast_predictors_ly{i}_EM_1980_{2019+i}_bias_corrected.nc')

    
    
### Apply bias correction to historical data
# load data
predictors_hindcast = xr.open_dataset(f'CanESM5_historical_predictors_EM_1980_2020.nc').sel(time = slice('1982',None))

    
ds, extend = bias_correction((predictors_hindcast.tos), obs.sst) ## SST
sst = ds.to_dataset(name = 'sst')
sst_anom = deseason(sst.sst).to_dataset(name = 'sst_anom')

ds, extend = bias_correction((predictors_hindcast.sos), obs.sss) ## SSS
sss = ds.to_dataset(name = 'sss')
sss_anom = deseason(sss.sss).to_dataset(name = 'sss_anom')

chl, chl_anom = chl_edit((predictors_hindcast.chlos), obs.chl) ## CHL
chl = chl.to_dataset(name = 'chl')
chl_anom = chl_anom.to_dataset(name = 'chl_anom')


wind, extend = bias_correction((predictors_hindcast.sfcWind **2), wind_obs) ## Wind
wind = wind.to_dataset(name = 'wind')

ls = [sst,sst_anom,sss,sss_anom, chl, chl_anom, wind] ## combine the data
xr.combine_by_coords(ls).to_netcdf(f'CanESM5_historical_predictors_EM_1980_2020_bias_corrected.nc')

### NOTE: Assimilation predictors are not bias corrected. However, CHL predictors should be unit corrected.
### NOTE: You can load fgco2 and bias correct them to your obsearvational benchmark of choice by adjusting the code above using same functions.