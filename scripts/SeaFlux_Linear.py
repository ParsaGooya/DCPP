###################################################################

# This script trains a multilinear model at each grid cell with monthly resolutions 
# using SeaFlux products as predictants and six predictors:
# SST, SSS, log(CHL), SfcWind squared, linear xCO2atm fit, linearly detrended xCO2atm.
# The script uses the model to make historical, assimilation and hindcast estiamtes
# using bias corrected predictors from CanESM5.

######################### Load Packages ############################
import xarray as xr
import nc_time_axis
import xesmf as xe
import wget
import glob
import numpy as np
import warnings
warnings.filterwarnings('ignore') # don't output warnings
import cftime

########################### functions ###############################

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

### remove repeating 1990-2019 seasonal cycle : (you can change the time period)

def deseason(ds):
    clim = ds.sel(time = slice('1990','2019')).groupby('time.month').mean()
    ds2 = ds.copy()
    for i in range(0,len(ds.time),12):

        ds2[i:i+12,:,:] = clim
    return ds - ds2


### find linear fit to the data using 1990-2019 linear trend : (the linear trend is extened to the same time length of dataset)

def poli(data):
  ds = data.sel(time = slice('1990','2019')).copy()

  time = np.arange(1990,2020,1/12)
  T = ds['time']
  ds['time'] = time
  m = ds.polyfit( dim  = 'time', deg = 1).polyfit_coefficients ## Find the linear fit over 1990-2019

  extend = xr.DataArray(np.arange(1990,1990+len(data['time'])/12,1/12), dims = 'time') ## extend the linear fit to the same time length of data
    
  extended_trend = m[0]*extend +  m[1]
  extended_trend['time'] = data['time']
  del ds
  del T
  return m[0], extended_trend

######################## defining the linear model ##################

from tqdm import tqdm
from sklearn import linear_model
from sklearn import preprocessing
reg = linear_model.LinearRegression()
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


## you can change this function to any preprocessing function on the predictor data or any non linear function such as normalize(), log() or sigmoid():
def normalize(ds):
    
    return ds


### Define the predictors with CHL. NOTE: atm_anom is the linear fit to deseasonalized xCO2atm timeseries. Do not confuse the naming.

variables =  ['atm_anom'  ,'chl_anom' , 'sss_anom' ,'sst_anom','atm_det', 'wind_anom']

### define linear regression function with CHL on each grid cell. 

def regress_chl(atm_anom, chl_anom, sss_anom, sst_anom,atm_det,  wind_anom, predictant):
    
        x = np.c_[atm_anom, chl_anom, sss_anom, sst_anom,atm_det, wind_anom]
        y = (predictant).reshape(-1,1)  
        y1 = y[~np.isnan(x).any(axis=1)]
        x1 = x[~np.isnan(x).any(axis=1), :]
        x = x1[~np.isnan(y1).any(axis=1), :]
        y = y1[~np.isnan(y1).any(axis=1)]
           
        ols= linear_model.LinearRegression()

        
        try:
            model = reg.fit(x,y)
            return np.append(model.intercept_,model.coef_[0])
        except:
            return np.empty((7))* np.nan
    

### Define the predictors without CHL

variables_no_chl =  ['atm_anom'  , 'sss_anom' ,'sst_anom','atm_det', 'wind_anom']

### define linear regression function without CHL on each grid cell. 

def regress_no_chl(atm_anom , sss_anom, sst_anom,atm_det, wind_anom,  predictant):
           
        ols= linear_model.LinearRegression()

        x = np.c_[atm_anom , sss_anom , sst_anom , atm_det, wind_anom ]
        y = (predictant).reshape(-1,1)
        y1 = y[~np.isnan(x).any(axis=1)]
        x1 = x[~np.isnan(x).any(axis=1), :]
        x = x1[~np.isnan(y1).any(axis=1), :]
        y = y1[~np.isnan(y1).any(axis=1)]
        
        
        try:
            model = reg.fit(x,y)
            return np.append(model.intercept_,model.coef_[0])
        except:
            return np.empty((6))* np.nan


### define regress function to use xarray's apply_ufunc for wrapping the above grid wise regression functions over each grid point. This function outputs coefficients 
### of regression at each grid cell. 

def regress(predictors, predictant, chl = True):

    
    
    if chl:
  
        out = xr.apply_ufunc(regress_chl, normalize(predictors.atm_anom),  normalize(predictors.chl_anom), normalize(predictors.sss_anom), normalize(predictors.sst_anom),
                           normalize(predictors.atm_det) , normalize(predictors.wind_anom) , predictant, 
                           input_core_dims=[['time'], ['time'], ['time'], ['time'], ['time'], ['time'], ['time']],
                           exclude_dims=set(("time",)),
                           output_core_dims=[["param"]],
                           vectorize=True,
                           dask="parallelized",
                           )
        return out.assign_coords(param = np.append('const',variables)) 

    else:
        
        out = xr.apply_ufunc(regress_no_chl, normalize(predictors.atm_anom), normalize(predictors.sss_anom), normalize(predictors.sst_anom) ,
                            normalize(predictors.atm_det) , normalize(predictors.wind_anom) , predictant,
                           input_core_dims=[['time'], ['time'], ['time'], ['time'], ['time'], ['time']],
                           exclude_dims=set(("time",)),
                           output_core_dims=[["param"]],
                           vectorize=True,
                           dask="parallelized",
                           )
        return out.assign_coords(param = np.append('const',variables_no_chl)) 
    

###  use the regression coefficients saved from the training above to make linear model predictions using any predictors values.
def predict(coefs, predictors, chl = True):
    
    if chl:
        vars_chl =  ['atm_anom'  ,'chl_anom' , 'sss_anom' ,'sst_anom', 'wind_anom']
        out = sum([normalize(predictors[var]) * coefs.sel(param = var) for var in vars_chl]) + normalize(predictors['atm_det']) * coefs.sel(param = 'atm_det') + coefs.sel(param = 'const')
    
    else:
        vars_no_chl =  ['atm_anom'  , 'sss_anom' ,'sst_anom','wind_anom']
        out = sum([normalize(predictors[var]) * coefs.sel(param = var) for var in vars_no_chl])+ normalize(predictors['atm_det']) * coefs.sel(param = 'atm_det')  + coefs.sel(param = 'const')

    return out



################################ load data #################################
###################### obseravational predictors #################
### atmospheric xCO2
atm = xr.open_dataset('~/atm_historical_obs_1972-2020.nc').rename({'__xarray_dataarray_variable__' : 'xCO2atm'}).xCO2atm.sel(time = slice(f'{1990}',f'{2019}'))

### observational predictors. 
obs = xr.open_dataset('~/obs_predictors.nc').sel(time = slice(f'{1990}',f'{2019}')).drop('mld')

### deseasonalize the predictors 
obs['atm_anom'] = deseason(obs.atm) 
obs['sst_anom'] = deseason(obs.sst)
obs['sss_anom'] = deseason(obs.sss)
obs['chl_anom'] = deseason(obs.chl)

### Loading ERA5 wind squared product
wind = xr.open_dataset('~/wind_obs_1982-2020.nc').wind.sel(time = slice(f'{1990}',f'{2019}'))

### Adding observed atmospheric concentrations of CO2 detrended dataset to predictos:
obs = xr.combine_by_coords([obs, (obs['atm_anom'] - poli(obs['atm_anom'])[1]).to_dataset(name = 'atm_det')])
### Finding the linear fit to atm_anom and replacing it with atm_anom as a predictor :
obs['atm_anom'] = poli(obs['atm_anom'])[1]
### Adding deseasonalized wind squared to predictors:
obs = xr.combine_by_coords([obs, wind, deseason(wind).to_dataset(name = 'wind_anom')])

###################### CanESM5 bias corrected historical predictors #################
### Exactly like above, but this time we load bias corrected historical predictors from CanESM5. Code for bias correction is available in  CanESM5_data.py

hist = xr.open_dataset(f'~/CanESM5_historical_predictors_EM_1980_2020_bias_corrected.nc').sel(time = slice('1990','2019')).drop(['mld'])
hist['sst_anom'] = deseason(hist.sst)
hist['sss_anom'] = deseason(hist.sss)
hist['chl_anom'] = deseason(hist.chl)
wind = xr.open_dataset(f'~/CanESM5_historical_wind_bias_corrected.nc').wind.sel(time = slice(f'{1990}',f'{2019}'))
hist = xr.combine_by_coords([hist, atm.to_dataset(name = 'atm'), deseason(atm).to_dataset(name = 'atm_anom')])
hist = xr.combine_by_coords([hist, (hist['atm_anom'] - poli(hist['atm_anom'])[1]).to_dataset(name = 'atm_det')])
hist['atm_anom'] = poli(hist['atm_anom'])[1]
hist = xr.combine_by_coords([hist,  deseason(wind).to_dataset(name = 'wind_anom')])

###################### CanESM5 bias corrected hindcast predictors #################
### Exactly like above, but this time we load bias corrected hindcast predictors from CanESM5 on lead years 1 to 10. 


ds_dict = {}
apco2 =  xr.open_dataset(f'~/CMIP6_atm_forecast_1982-2029.nc').ssp245.sel(time = slice('1990',None))
for i in range(1,11):
    

    ds_dict[i] = xr.open_dataset(f'~/CanESM5_hindcast_predictors_ly{i}_EM_1980_{2019+i}_bias_corrected.nc').sel(time = slice('1990',None)).drop(['mld'])
    ds_dict[i]['sst_anom'] = deseason(ds_dict[i].sst)
    ds_dict[i]['sss_anom'] = deseason(ds_dict[i].sss)
    ds_dict[i]['chl_anom'] = deseason(ds_dict[i].chl)
    ds_dict[i] = xr.combine_by_coords([ds_dict[i],apco2.sel(time = slice('1990',f'{2019+i}')).to_dataset(name = 'atm'), deseason(apco2.sel(time = slice('1990',f'{2019+i}'))).to_dataset(name = 'atm_anom')])
    ds_dict[i] = xr.combine_by_coords([ds_dict[i], (ds_dict[i]['atm_anom'] - poli(ds_dict[i]['atm_anom'])[1]).to_dataset(name = 'atm_det')])
    ds_dict[i]['atm_anom'] = poli(ds_dict[i]['atm_anom'])[1]
    ds_dict[i] = xr.combine_by_coords([ds_dict[i],  deseason(ds_dict[i].wind).to_dataset(name = 'wind_anom')])

###################### CanESM5 assimilation predictors #################
### Exactly like above, but this time we assimilation predictors from CanESM5. Assimilation predictors are not bias corrected.

assim = xr.open_dataset('~/CanESM5_assim_predictors_r1i1p2f1_1980-2020_bias_corrected.nc').sel(time = slice('1990','2019')).drop(['mld'])
assim['sst_anom'] = deseason(assim.sst)
assim['sss_anom'] = deseason(assim.sss)
assim['chl_anom'] = deseason(assim.chl)
wind = xr.open_dataset(f'~/CanESM5_assim_sfcWind_1980-2020_r1i1p2f1.nc').wind.sel(time = slice('1990','2019'))
assim = xr.combine_by_coords([assim,atm.to_dataset(name = 'atm'), deseason(atm).to_dataset(name = 'atm_anom')])
assim = xr.combine_by_coords([assim, (assim['atm_anom'] - poli(assim['atm_anom'])[1]).to_dataset(name = 'atm_det')])
assim['atm_anom'] = poli(assim['atm_anom'])[1]
assim = xr.combine_by_coords([assim,  deseason(wind).to_dataset(name = 'wind_anom')])

##################### Load SeaFlux observation based product for ocean carbon flux as predictant ##############

sink =  xr.open_dataset('~/SeaFlux_v2021.04_fgco2_all_winds_products.nc?download=1', engine = 'netcdf4').sel(wind = 'ERA5').fgco2  #### Choose ERA5 wind product
sink['time'] = obs['time']

################# Training for each of the six SeaFlux data products ###############
### Iterate over each of the six SeaFlux data products and train the a linear model for each.
for prd in sink.product:



    print(f'training for {prd}') 

    target = sink.sel(product = prd) ### predictant data
    predictors = obs.where(~np.isnan(target)) ### removing data points where no predictant data is available.
    ref = deseason(target) ### deseasonalize predictant data

    predictors = predictors.where(~np.isnan(predictors.chl)) ### remove CHL nan values from predictors and predictant
    ref = ref.where(~np.isnan(predictors.chl))
    predictors = predictors.where(~np.isnan(predictors.sss)) ### remove SSS nan values from predictors and predictant
    ref = ref.where(~np.isnan(predictors.sss))
    out_chl = regress(predictors, ref) #### train linear model at each grid cell with CHL

    predictors = obs.where(~np.isnan(target)) ### removing data points where no predictant data is available.
    ref = deseason(target) ### deseasonalize predictant data
    predictors = predictors.where(~np.isnan(predictors.sss)) ### remove SSS nan values from predictors and predictant
    ref = ref.where(~np.isnan(predictors.sss))
    out_no_chl = regress(predictors, ref, False) #### train linear model at each grid cell without CHL

    ######## Use the model above to make predictions using obseravtaional predictors (reconstruct)
    ds = predict(out_chl, obs) # with CHL
    ds2 = predict(out_no_chl, obs, False) # without CHL
    recons = ds.combine_first(ds2) # Combine with priority gicen to CHL
    recons.to_dataset(name = 'fgco2').to_netcdf(f'~/Linear_models/SeaFlux/{prd}/recons_{prd}_linear.nc') # Save reconstruction using prd as predictant

    ######### Use the model above to make predictions using CanESM5 historical predictors 
    ds = predict(out_chl, hist)
    ds2 = predict(out_no_chl, hist, False)     
    historical = ds.combine_first(ds2)        
    historical.to_dataset(name = 'fgco2').to_netcdf(f'~/Linear_models/SeaFlux/{prd}/hist_{prd}_linear.nc') # Save historical using prd as predictant

    ########## Use the model above to make predictions using CanESM5 assimilation predictors
    ds = predict(out_chl, assim)
    ds2 = predict(out_no_chl, assim, False)
    assimilation = ds.combine_first(ds2)    
    assimilation.to_dataset(name = 'fgco2').to_netcdf(f'~/Linear_models/SeaFlux/{prd}/assim_{prd}_annual_lienar.nc') # Save assimilation using prd as predictant

    #### Use the model above to make predictions using CanESM5 hindcast predictors 
    hindcast_dict = {}
    for i in range(1,11):
        
        print(f'ly {i}')
        ds = predict(out_chl, ds_dict[i])
        ds2 = predict(out_no_chl, ds_dict[i], False)
        hindcast_dict[i] = ds.combine_first(ds2) 
        hindcast_dict[i].to_dataset(name = 'fgco2').to_netcdf(f"~/Linear_models/SeaFlux/{prd}/hindcast_{prd}_linear_ly{i}.nc")



#### Train using the MEAN as predictant



print(f'training for Mean')

target = sink.mean('product')
predictors = obs.where(~np.isnan(target))
ref = deseason(target)

predictors = predictors.where(~np.isnan(predictors.chl))
ref = ref.where(~np.isnan(predictors.chl))
predictors = predictors.where(~np.isnan(predictors.sss))
ref = ref.where(~np.isnan(predictors.sss))
out_chl = regress(predictors, ref)

predictors = obs.where(~np.isnan(target))
ref = deseason(target)
predictors = predictors.where(~np.isnan(predictors.sss))
ref = ref.where(~np.isnan(predictors.sss))
out_no_chl = regress(predictors, ref, False)

ds = predict(out_chl, obs)
ds2 = predict(out_no_chl, obs, False)
recons = ds.combine_first(ds2)
recons.to_dataset(name = 'fgco2').to_netcdf(f'~/Linear_models/SeaFlux/Mean/recons_MEAN_linear.nc')

ds = predict(out_chl, hist)
ds2 = predict(out_no_chl, hist, False)
historical = ds.combine_first(ds2)        
historical.to_dataset(name = 'fgco2').to_netcdf(f'~/Linear_models/SeaFlux/Mean/hist_MEAN_linear.nc')

ds = predict(out_chl, assim)
ds2 = predict(out_no_chl, assim, False)
assimilation = ds.combine_first(ds2)    
assimilation.to_dataset(name = 'fgco2').to_netcdf(f'~/Linear_models/SeaFlux/Mean/assim_MEAN_annual_lienar.nc')


hindcast_dict = {}
for i in range(1,11):
    
    print(f'ly {i}')
    ds = predict(out_chl, ds_dict[i])
    ds2 = predict(out_no_chl, ds_dict[i], False)
    
    hindcast_dict[i] = ds.combine_first(ds2) 
    hindcast_dict[i].to_dataset(name = 'fgco2').to_netcdf(f"~/Linear_models/SeaFlux/Mean/hindcast_MEAN_linear_ly{i}.nc")