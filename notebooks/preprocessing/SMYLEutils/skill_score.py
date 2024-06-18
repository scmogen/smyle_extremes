# Skill Score Functions
import xarray as xr
import numpy as np
import xskillscore as xs
import esmtools as stat2

def detrend_linear(dat, dim):
    """ linear detrend dat along the axis dim """
    params = dat.polyfit(dim=dim, deg=2)
    fit = xr.polyval(dat[dim], params.polyfit_coefficients)
    dat = dat-fit
    return dat

def leadtime_corr_byseas(mod_da,mod_time,obs_da,detrend=False):
    """ 
    Computes the correlation coefficient between two xarray DataArrays, which 
    must share the same lat/lon coordinates (if any). Assumes time coordinates are roughly compatible
    between model and obs.
    
        Inputs
        mod_da: a seasonally-averaged hindcast DataArray dimensioned (Y,L,M,...)
        obs_da: an OBS DataArray dimensioned (time,...)
        mod_time: a hindcast time DataArray dimensioned (Y,L). NOTE: assumes mod_time.dt.month
            returns mid-month of 3-month seasonal average (e.g., mon=1 ==> "DJF").
    """
    mod_da = mod_da.mean('M')

    r_list = []
    r2_list = []
    p_list = []
    
    # run the autocorrelation
    j = 1
    ens = mod_da.sel(L=j).rename({'Y':'time'})
    ens_time_year = mod_time.sel(L=j).time.dt.year.data
    ens_time_month = mod_time.sel(L=j).time.dt.month.data[0]
    ens_ts = ens.assign_coords(time=("time",ens_time_year))
    
    obs_ts = obs_da.where(obs_da['time']['month'] == ens_time_month - 1,drop=True)
    obs_ts['time'] = obs_ts['time'].dt.year
    
    a,b = xr.align(ens_ts,obs_ts)
    
    if detrend:
            a = detrend_linear(a,'time')
            b = detrend_linear(b,'time')
            
    aut = stat2.stats.autocorr(b,nlags=24) # autocorrelation
    aut = aut.rename({'lead':'L'})
    
    
    # use these for calculations of skill!
    for i in mod_da.L.values:
        ens = mod_da.sel(L=i).rename({'Y':'time'})
        ens_time_year = mod_time.sel(L=i).time.dt.year.data
        ens_time_month = mod_time.sel(L=i).time.dt.month.data[0]
        ens_ts = ens.assign_coords(time=("time",ens_time_year))
        
        obs_ts = obs_da.where(obs_da['time']['month'] == ens_time_month,drop=True)
        obs_ts['time'] = obs_ts['time'].dt.year
        
        a,b = xr.align(ens_ts,obs_ts)
        
        if detrend:
                a = detrend_linear(a,'time')
                b = detrend_linear(b,'time')
        r = xr.corr(a,b,dim='time')
        p = xs.pearson_r_eff_p_value(a,b,dim='time')
        
        
        aut = stat2.stats.autocorr(b,nlags=24) # autocorrelation
        aut = aut.isel(lead=slice(1,24))
        aut = aut.rename({'lead':'L'})

        # a = stat2.stats.autocorr(b,nlags=24,dim='time')
        
        r_list.append(r)
        p_list.append(p)
        # a_list.append(a)
        
    corr = xr.concat(r_list,mod_da.L)
    pval = xr.concat(p_list,mod_da.L)
    
    # aval = xr.concat(aut,ens.L)
    aval = aut.isel(L=slice(1,24))

    return xr.Dataset({'corr':corr,'pval':pval,'aval':aut})
