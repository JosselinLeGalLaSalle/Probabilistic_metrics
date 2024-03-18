# -*- coding: utf-8 -*-
"""
Created on Tue Aug 8 16:13:33 2023

@author: jianfan
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 16:19:47 2023

@author: jlegalla
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 20:42:50 2022
@author: r.amaro_e_silva
"""

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import pdb
from scipy.stats import scoreatpercentile


def read_observation_data(dir_, time_col='time', datetime_format='%Y-%m-%d %H:%M:%S'):
    """
    Read observation data from a CSV file and process time-related attributes.

    This function loads observational data from a CSV file, sets a specified column as a time-based index, 
    and performs certain quality control operations, such as filtering based on the solar elevation.

    Parameters
    ----------
    dir_ : str
        Path to the CSV file containing observation data.
    time_col : str, optional
        The column name in the CSV that contains timestamp information. Default is 'time'.
    datetime_format : str, optional
        The format of the datetime in the `time_col` column. Default is '%Y-%m-%d %H:%M:%S'.

    Returns
    -------
    df : pandas.DataFrame
        The loaded observational data with the time column set as an index.
    ix_qc : pandas.Series[bool]
        A quality control index that indicates rows where the solar elevation exceeds 5 degrees."""
    # read csv, define time as index and filter only test period (here 2020)
    df = pd.read_csv(dir_)

    df = df.set_index(time_col)
    df.index = pd.to_datetime(df.index, format=datetime_format)

    # TODO: read metadata (lat,lon)
    # add "sun_position=False,meta" to input list
    # TODO: calculate internally solar position using sg2
    # TODO: generate t-dt/2 timestamps
    # TODO: define QC-approved index (e.g., sun_elev > 5º)
    # if sun_position == True:
    #     lat = meta[0]
    #     lon = meta[1]
    #     h = meta[2]

    #     a = sg2.sun_position([[mm.lon, mm.lat, mm.h]], df.index.values + extra_t,
    #                          ["topoc.gamma_S0","topoc.alpha_S","topoc.toa_hi","topoc.toa_ni"])

    ix_qc = (df['sun_elev'] > 5).rename('QC')

    return df, ix_qc


def build_cdf(fc_eps, cdf_method, ix_qc, min_ensemble_size=None, is_clim_model=False):
    """
    Build the Cumulative Distribution Function (CDF) from ensemble forecast.

    Parameters:
    ----------
    fc_eps : pandas DataFrame
        Ensemble forecast data where each row is an ensemble forecast sample.
    
    cdf_method : str
        Method for building the CDF, accepted values: 'classic', 'uniform', 'non-uniform'.
        
    ix_qc : pandas Series
        Boolean series for Quality Control, indicates valid rows in `fc_eps`.
        
    min_ensemble_size : int, optional
        Minimum ensemble size threshold. Rows with fewer non-NaN values than this threshold will be discarded.
        
    is_clim_model : bool, optional
        Indicates if the provided forecast is from a climate model. Default is False.

    Returns:
    -------
    fc_cdf : dict
        Dictionary with two keys: 'GHI' and 'prob'. 'GHI' contains the sorted ensemble values,
        and 'prob' contains the corresponding probabilities.
        
    ix_qc : pandas Series
        Updated Boolean series after quality control.
    """
    # %% asserts
    # TODO: add assert for fc_eps and ix_qc

    error_msg = '\'' + cdf_method + '\' not accepted as <cdf_method>. Please choose between \'classic\', \'uniform\', and \'non-uniform\'.'
    assert cdf_method in ('classic', 'uniform', 'non-uniform'), error_msg

    if min_ensemble_size != None:
        error_msg = 'When provided, \'min_ensemble_size\' must be an integer.'
        assert isinstance(min_ensemble_size, int), error_msg

    # %% pre-treatment
    ix_qc = ix_qc.copy()

    # number of ensemble members (non-NaN)
    n_EnsMembers = fc_eps.iloc[:, 1:].count(axis=1).values#############

    # if variable-sized ensembles, cdf must be interpolated for consistency
    # if len(np.unique(n_EnsMembers[ix_qc])) > 1:
    #     error_msg = "If ensembles have variable # of members, please define \'interp_cdf\' as True and provide \'interp_cdf_resol\'."
    #     assert (interp_cdf==True) & (interp_cdf_resol!=None), error_msg

    # dismisses ensembles which are smaller than min_ensemble_size
    # only runs if variable is not None
    if min_ensemble_size:
        ix_qc.iloc[:] = ix_qc.values & (n_EnsMembers > min_ensemble_size - 1)

    # sorts each ensemble (row-wise), leaving possible nan values in the end
    fc_eps.iloc[:, 1:] = np.sort(fc_eps.iloc[:, 1:].values, axis=1)

    # %% cdf_building + interpolation
    # TODO: allow user-defined variable instead of GHI
    fc_cdf = {'GHI': [], 'prob': [], 'cdf_method': cdf_method}    
    fc_cdf['GHI'] = fc_eps.iloc[:, 1:].values
    if cdf_method == 'classic':
        # TODO: adjust indexing if an EPS with variable # of members is to be evaluated
        jump = 1 / (np.unique(n_EnsMembers[ix_qc])[0])
        fc_cdf['prob'] = np.arange(0, 1 +jump,jump)#jump##################
    if cdf_method == 'uniform':
        jump = 1 / (np.unique(n_EnsMembers[ix_qc])[0]+1)
        fc_cdf['prob'] = np.arange(0, 1 +jump,jump)#jump##################
    return fc_cdf, ix_qc



def CRPS_classic(obs, fc_cdf, ix_qc):
    CRPS = np.ones(obs.shape[0]) * np.nan
    #n_samples = 3000 #for Monte-Carlo
    for i in range(len(obs)):
        if ix_qc[i] == False:
            continue

        # TODO: check if fcs is dataframe or numpy matrix
        # TODO: instead of adding ifs to manage these differences, convert data
        # into fixed format in the beginning and reconvert it if needed in the
        # end (avoids if's inside loops)
        if isinstance(fc_cdf['prob'], list):
            prob = fc_cdf['prob'][i]
        else:
            prob = fc_cdf['prob'][1:]

        # TODO: allow user-defined column (or simply the one non-prob)
        fc = fc_cdf['GHI'][i, :].astype(float)

        # TODO: check why baseline is providing all-nans in 14515，Core part，Remove all nan values
        if all(np.isnan(fc)):
            continue

        # TODO: if EPS with variable # of members are accepeted, have to handle nans
        
        #If uniform or non-uniform is chosen, 
        #the ensemble members $E_0$ and $E_{M+1}$ corresponding to the extreme values need to be added
        #fc=np.append(fc,-4)
        #fc=np.append(fc,1300)
        #fc.sort()

        # %%
        # TODO: recheck temp in first 2 cases
        if obs['ghi'][i] < fc[0]:  # observation outside of fc_cdf (on the left)
            x = np.append(obs['ghi'][i], fc)
            temp = np.append(1, 1 - prob)  # used in final CRPS calculation
        elif obs['ghi'][i] > fc[-1]:  # observation outside of fc_cdf (on the right)
            x = np.append(fc, obs['ghi'][i] )
            temp = np.append(prob, 1)  # used in final CRPS calculation
        else:  # observation in the middle of fc_cdf
            if fc_cdf['cdf_method'] == 'classic':
                f = interp1d(fc, prob, kind='previous')
                # adds point in fc(x = obs(t)-dx), where vertical jump in H occurs
                temp2 = obs['ghi'][i] - np.array([0, 0.00001])
            else:
                raise Exception("For now, only classic CDF is accepted")
                # f = interp1d(fc,prob,kind='linear')
                # temp2 = obs['ghi'][i]

            #x = np.append(fc, temp2)

            x = np.append(fc, obs['ghi'][i])
            x.sort()

            # interpolating native eCDF to include obs(t)
            prob_i = f(x)
            ix_obs = np.where(x == obs['ghi'][i])[0][0]
            # building Heaviside (for plotting purposes)
            # H = np.ones(x.shape)*np.nan
            # H[:ix_obs] = 0
            # H[ix_obs:] = 1

            # calculating CRPS
            temp = np.ones(x.shape) * np.nan
            temp[:ix_obs] = prob_i[:ix_obs]  # eCDF>H, area under eCDF
            temp[ix_obs:] = 1 - prob_i[ix_obs:]  # eCDF<=H, area above eCDF

        # TODO: consider different types of integration (left,right,...)
        #Trapezoid
        #CRPS[i] = sum(np.diff(x) * (temp[:-1] ** 2+temp[1:] ** 2)/2)
        ######################################++++++++++++++Mid-point-Mehtod++++++++++++++############################
        #mid-point non-uniform point, should have interpolation
        #mid_points = (x[:-1] + x[1:]) / 2
        #temp_mid = np.interp(mid_points, x, temp)
        #CRPS[i] = sum(np.diff(x) * temp_mid**2)
        ######################################++++++++++++++MC-Mehtod++++++++++++++############################
        # Adjust the number of points as required
        '''
        random_samples = np.random.uniform(x.min(), x.max(), n_samples)
        # sorting
        random_samples.sort()

        # interpolation
        temp_interp = np.interp(random_samples, x, temp)

        # Monte Carlo formula
        CRPS[i] = (x.max() - x.min())*np.mean(temp_interp ** 2)'''
        ######################################++++++++++++++Simpson-Mehtod############################
        '''
        # Initialize CRPS[i] to zero
        CRPS[i] = 0

        subintervals = [(x[j], x[j + 1]) for j in range(len(x) - 1)]
        simpson_sum = 0
        
       
        for a, b in subintervals:
                h = b - a
                
                # Find the indices of a and b in the x array
                a_index = np.where(x == a)[0][0]
                b_index = np.where(x == b)[0][0]

                # Get the values of temp at a and b directly from the temp array
                fa = temp[a_index]**2
                fb = temp[b_index]**2
                
                # Interpolate the value of temp at the midpoint c
                fc = np.interp((a + b) / 2, x, temp)**2
                
                # Calculate the contribution of the current subinterval using Simpson's rule
                simpson_sum += (h / 6) * (fa + 4 * fc + fb)
            
            # Assign the final CRPS value for the current observation
        CRPS[i] = simpson_sum
        '''
        ################################################ left/right rectangular method
        ####relates with [1:] or [:-1] indexing of temp####
    
        #right type
        #CRPS[i] = sum(np.diff(x) * temp[1:] ** 2)##########USING Right intregral 
        #left type
        CRPS[i] = sum(np.diff(x) * temp[:-1] ** 2)##########USING left intregral，More resonable here
    return CRPS


def CRPS_Brier(obs, fc_cdf, integ_step,ix_qc):

    # CRPS = np.ones((1,1))*np.nan
    assert isinstance(fc_cdf['prob'], np.ndarray), 'Error to be described #1'
    #variables_df = pd.DataFrame()
    # TODO: user-defined max?
    thresh = np.arange(0, 1300, integ_step)
    d_thresh = thresh[1:] - thresh[:-1]

    # initialiaze components
    REL = np.ones(thresh.shape) * np.nan
    RES = np.ones(thresh.shape) * np.nan
    UNC = np.ones(thresh.shape) * np.nan
    CRPS = np.ones(thresh.shape) * np.nan

    
    for val in range(1, len(thresh)):#
        # converting observations to binary
        O = obs['ghi'][ix_qc].values < thresh[val]
        
        # initialize probabilities as nan
        prob = np.ones(O.shape) * np.nan
        
        # calculate probability of occurence
        # by checking the number of percentiles below the threshold
        # TODO: allow user-defined variable
        base_prob = np.sum(fc_cdf['GHI'][ix_qc, :] < thresh[val], axis=1)

        ix = base_prob > 0
        prob[ix] = fc_cdf['prob'][base_prob[ix]]#Key step to find out the index ### - 1
        prob[base_prob == 0] = 0
        
        #print(len(fc_cdf['GHI'][ix_qc, :][np.isnan(fc_cdf['GHI'][ix_qc, :])]))
        #print(base_prob.shape)
        #print(len(O[np.isnan(O)]))
        #print(len(prob[np.isnan(prob)]))
        # if only overall value is of interest
        CRPS[val] = np.mean((O - prob) ** 2)
        
        # %% CRPS components
        # contigency table, full_quants includes 0
        prob_ix = np.zeros((len(fc_cdf['prob']), 2))
        for qt in range(0, len(fc_cdf['prob'])):
            prob_ix[qt, 0] = (O & (prob == fc_cdf['prob'][qt])).sum() 
            prob_ix[qt, 1] = (~O & (prob == fc_cdf['prob'][qt])).sum()  # base_prob==qt

        l = np.sum(prob_ix, axis=1)  #There are 730 lines in total, and each line is the sum of (M+1)*2,

        # weights
        g = l / sum(l) 
        #print(sum(l))
        # mean observation
        o=prob_ix[:, 0] / l 
        #print(len(o[np.isnan(o)]))
        o[np.isnan(o)] = 0

        # global mean for observation (equivalent to Josselin's result)
        o_ = sum(g * o)
        
        #print(o_)
        REL[val] = sum(g * (o - fc_cdf['prob']) ** 2)
        RES[val] = sum(g * (o - o_) ** 2)
        UNC[val] = o_ * (1 - o_)

    return (CRPS[1:]* d_thresh).sum(),(REL[1:] * d_thresh).sum(),(RES[1:] * d_thresh).sum(),(UNC[1:] * d_thresh).sum()


###############################QS calculation and its decomposition############################
def quantile_score_decomposition(tau, vec_x, vec_xa,num_bins):#,ix_qc
    """This function computes the QS based on the given dataset and realizes the decomposition of QS by different methods 
  
      Calculate QS and their components.
     .. math::
            $S(\hat{F}, Q) = S(\bar{Q} , Q) - d(\bar{Q}, Q) + d(\hat{F}, Q)$
    where ``\hat{F}`` and ``Q`` are the distribution of forecast and observation,
    \bar{Q} is the marginal distribution of observations is usually viewed as a climatological.

     .. note::
        Reference:Decomposition and graphical portrayal of the quantile score Sabrina Bentziena,b* and Petra Friederichs.
        [Link to the article](https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.2284)
        
    Parameters
    ----------
    tau : float
        The probability level used for the QS.
    vec_x : array_like
        Forecast values for the event.
    vec_xa : array_like
        Observed values for the event.
    num_bins : int
        The number of subsample bins to divide the QS decomposition. Different ``num_bins`` have different 
        effects on the decomposition discretization error.

    Returns
    -------
    qs : float
        The Quantile Score for the given probability level τ($\tau$).
    res : float
        The resolution component of the QS for the given probability level τ($\tau$).
    rel : float
        The reliability component of the QS for the given probability level τ($\tau$).
    u : float
        The uncertainty component of the QS for the given probability level τ($\tau$).
    """
        
    vec_x=np.array(vec_x)
    vec_xa=np.array(vec_xa)
    if len(vec_x) != len(vec_xa):
        raise ValueError("The observation vector and the forecast vector are not of the same length")

    #ComputesQS score for a given probability level τ
    def check_loss(obs,prev): 
        cl=np.ones(obs.shape) * np.nan
        cl = (obs -prev) * tau
        mask = obs < prev
        cl[mask] = (obs[mask] - prev[mask]) *(tau-1)
        return cl
    qs = np.nanmean(check_loss(vec_xa,vec_x))
    #Estimate quantiles for corresponding probability levels in Observations by order statistics

    q_climato = scoreatpercentile(np.sort(vec_xa), tau * 100)

    # divide the data into k = 1, ... , K subsamples Ik，K=nimbins,define seuils as a judge of bin boundaries(Method 2)

    seuils = np.linspace(np.min(vec_x), np.max(vec_x), num_bins)

    # Defining the boundaries of bins by quantiles（Method 3）
    #seuils = [np.percentile(vec_x, i * 100.0 / num_bins) for i in range(num_bins + 1)]

    entropie = np.zeros(len(seuils))#cross entrophy
    res = np.zeros(len(seuils))
    rel = np.zeros(len(seuils))
    w = np.zeros(len(seuils))#weights
    u = np.zeros(len(seuils))#incertitude

    for i in range(len(seuils)-1):
        indicesxa=((vec_x < seuils[i+1])&(vec_x>seuils[i]))
        xa = vec_xa[indicesxa]
        indicesx= ((vec_x <seuils[i+1])&(vec_x>seuils[i]))
        x = vec_x[indicesx]
        x=np.nanmean(x)*np.ones(len(x))
        l_ones=np.ones(len(xa))
        q_climatones=q_climato*np.ones(len(xa))
        w[i] = len(xa) / len(vec_xa)
        qi = scoreatpercentile(np.sort(xa), tau * 100)#np.int(scoreatpercentile(np.sort(xa), tau * 100)) 
        #print('qi',qi)
        ######Whether to take an integer affects the average value under the probability level tau in the subset
        qi_ones=qi*l_ones
        #entropie[i] = np.nanmean(check_loss(xa, qi_ones))
        #entropy Sτ (Q, Q)
        res[i] = np.nanmean(check_loss(xa, q_climato * np.ones(len(xa))) - check_loss(xa, qi * np.ones(len(xa))))
        rel[i] = np.nanmean(check_loss(xa, x) - check_loss(xa, qi *np.ones(len(xa))))
        u[i] = np.nanmean(check_loss(xa, q_climato * np.ones(len(xa)))) 
    #Computation of components of QS
    res = np.nansum(w * res)
    rel = np.nansum(w * rel)
    u = np.nansum(w * u)

    return qs, res, rel, u     

    '''
    ########### K=The number of Division in linespace(min,max,K)，Method 1 ##################################
    #Predefine an indices_list to store the indices in each bin
    indices_list = []
    #Find the index in each bins      
    for i in range(len(seuils)):
        diffs = [np.abs(seuil -vec_x) for seuil in seuils]#
        #Find the index closest to Seuils[i] in the (i+1)^th bin
        indices = np.where(diffs[i] < np.min(np.delete(diffs, i, 0), axis=0))[0]
        indices_list.append(indices)

    count=[]
    for i in range(10):
        print(len(indices_list[i]))
        count.append(len(indices_list[i]))
    print('Number of indices_list',count)
    print('verification',np.sum(count))


    for idx, indices in enumerate(indices_list): #########idx=k
        xa = vec_xa[indices]
        x = vec_x[indices]
        l_ones = np.ones(len(xa))
        #Calculate the Weight/Proportion of the observation-prediction pair for each bin to the  vector
        w[idx] = len(xa) / len(vec_xa)
        #Estimate quantiles in each bins for observations by order statistics
        qi = scoreatpercentile(np.sort(xa), tau * 100)#qi = np.int(scoreatpercentile(np.sort(xa), tau * 100))
        qi_ones = qi * l_ones
        res[idx] = np.nanmean(check_loss(xa, q_climato * np.ones(len(xa))) - check_loss(xa, qi * np.ones(len(xa))))
        rel[idx] = np.nanmean(check_loss(xa, x) - check_loss(xa, qi * np.ones(len(xa))))
        u[idx] = np.nanmean(check_loss(xa, q_climato * np.ones(len(xa))))
    #Computation of components of QS
    #print('w',w)
    #print('res',res)
    #print('rel',rel)
    #print('u',u)
    res = np.nansum(w * res)
    rel = np.nansum(w * rel)
    u = np.nansum(w * u)
    return qs, res, rel, u 
    '''
        
   
 ###### Integration of QS and QS components
def CRPS_QS_score(obs,fc_cdf,ix_qc,num_bins):
    """
    'This function integrates the QS scores of the dataset and the components of the QS 
     scores decomposed by the decomposition method of Proper score,
     to calculate CRPS and their components through QS score.'


    .. math::
    CRPS(\hat{F},y) = 2\int_{0}^{1} QS_{\tau} d\tau

    where ``\hat{F}`` and ``o`` are quantile forecast and observation.

    .. note::
    Reference:Verification tools for probabilistic forecasts of continuous hydrological variables.
    [Link to the article](https://www.researchgate.net/publication/29626664_Verification_tools_
    for_probabilistic_forecasts_of_continuous_hydrological_variables)

    Parameters
    ----------
    obs : pandas DataFrame. Used to store and manipulate tabular data where you have rows and columns.
        The observations of the event.
    fc_cdf :dictionary
        A dictionary that contains the sorted ensemble values ('GHI' key) 
        and the associated probabilities ('prob' key), as well as the chosen CDF method ('cdf_method' key).
    ix_qc: pandas Series（boolean type）
        Completed quality control of solar altitude angles and removed nan values from the dataset
    num_bins : numint
    The number of subsample bins to divide the QS decomposition, 
    different ``num_bins`` have different effects on the   decomposition discretization error
   
    Returns
    -------
    CRPS : scalar 
        The CRPS value obtained by integrating  2QS over the probability level $\tau$
    -------
    Rel_crps : scalar 
        Obtained by integrating the reliability of the $QS_{\tau}$ over all probability levels
    -------
    Res_crps : scalar 
        Obtained by integrating the resolution of the $QS_{\tau}$ over all probability levels
    -------
    U_crps : scalar 
        The uncertainty obtained corresponds to the CRPS by the method of QS."""   
        
    QS_total=np.ones(len(fc_cdf['prob'])-1) * np.nan
    res_total=np.ones(len(fc_cdf['prob'])-1) * np.nan
    rel_total=np.ones(len(fc_cdf['prob'])-1) * np.nan
    u_total=np.ones(len(fc_cdf['prob'])-1) * np.nan
    for j in range(len(fc_cdf['prob'])-1):
        vec_x=fc_cdf['GHI'][ix_qc][:,j]
        vec_xa=obs['ghi'][ix_qc].values
        tau=fc_cdf['prob'][j+1]
        QS_total[j],res_total[j],rel_total[j],u_total[j]=quantile_score_decomposition(tau, vec_x, vec_xa,num_bins)
    return np.sum(np.diff(fc_cdf['prob'][1:])*(QS_total[1:]+QS_total[:-1])),np.sum(np.diff(fc_cdf['prob'][1:])*(rel_total[1:]+rel_total[:-1])),np.sum(np.diff(fc_cdf['prob'][1:]) * (res_total[1:]+res_total[:-1])),np.sum(np.diff(fc_cdf['prob'][1:]) * (u_total[1:]+u_total[:-1]))

    
    
    
    
    
    
    








        





