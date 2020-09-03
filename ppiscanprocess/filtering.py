# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 16:08:50 2018

Package for line-of-sight wind speed filtering and interpolation in long range PPI scans.
The algorithms here presented where tested with the balcony experiment data. 

Four filters are implemented:
    
    1.- CNR threshold: 
    2.- Median-like filter
    3.- Kernel density filter (not used)
    4.- Clustering via DBSCAN

@author: 
Leonardo Alcayaga
lalc@dtu.dk

"""
# In[Packages used]
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy as sp

from sklearn.neighbors import KDTree
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import DBSCAN
from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks
import mpl_toolkits.mplot3d as mp3d
from mpl_toolkits.mplot3d import Axes3D


# In[############# [Functions] #################]

# In[DataFrame manipulation and misc]

def runningmedian(x,N):
    idx = np.arange(N) + np.arange(len(x)-N+1)[:,None]
    b = [row[row>0] for row in x[idx]]
    return np.array([np.median(c) for c in b])

def df_ws_grad(df,grad=None):
    """
    Function that calculates the average difference in V_LOS between a
    particular point and its neighbours. V_LOS difference in the same point along ssuccesive
    scans is also calculated. Additional features for each point will be then callled dvdr and dvdt.
    
    Could be inproved estimating an actual gradient instead of
    just a difference
    
    Input:
    -----
        df       - Pandas dataframe containing the positional information
                   (range gate and azimuth angle) and line-of-sight wind speed.
    Output:
    ------
        df_prime - modified Pandas dataframe with additional dvdr and dvdt
                   columns contaning the average of V_LOS difference in 
                   space ant time.  
    """
    #Azimuth angles
    phi = df.azim.unique()
    #V_LOS difference forward in range gate
    dvdr1 = df.ws.diff(axis=1)
    #V_LOS difference backward in range gate
    dvdr2 = df.ws.diff(axis=1,periods=-1)
    #V_LOS difference forward in azimuth
    dvdr3 = df.ws.diff(axis=0)
    dvdr3.loc[df.azim==phi[0]] = np.nan
    #V_LOS difference backward in azimuth
    dvdr4 = df.ws.diff(axis=0,periods=-1)
    dvdr4.loc[df.azim==phi[-1]] = np.nan
    # Median over diferences
    dvdr = pd.concat([dvdr1, dvdr2, dvdr3, dvdr4]).groupby(level=0).median() 
    dvdr.columns = ['dvdr']*(df['ws'].shape[1])
    # V_LOS change in time (between sucessive scans)
    dvdt = df.ws.diff(axis=0,periods=45).fillna(value=0.0)  
    dvdt.columns = ['dvdt']*(df['ws'].shape[1])
    #Additional columns
    if grad == 'dvdr':
        dv = dvdr
    if grad == 'dvdt':
        dv = dvdt   
    df_prime = pd.concat([df,dv], axis=1)
    return(df_prime)

def df_mov_med(df,col,n):
    """
    Function that calculates moving average of the specified feature
    
    Input:
    -----
        df       - Pandas dataframe containing the positional information
                   (range gate and azimuth angle) and line-of-sight wind speed.
                   
        col      - Feature in df on which moving median is estimated.
        
        n        - Moving window size.
        
    Output:
    ------
        df_prime - modified Pandas dataframe with additional column containing 
                   the moving median of specific feature.  
    """

    movmedian = df[col].rolling(n,axis=1,min_periods=1).median()
    movmedian = df[col].sub(movmedian).abs()
    movmedian.columns = ['movmed']*(df[col].shape[1])
    df_prime = pd.concat([df,movmedian], axis=1)
    
    return(df_prime)

# In[############## Filtering functions ################]
    
# In[CNR threshold]
    
def filt_CNR(df,value=[-27.0,-8.0]):
    """
    Filter based on a CNR threshold, measurements with a Carrier-to-Noise value less than value are
    rejected. Acts on V_LOS.
    
    Input:
    -----
        df       - Pandas dataframe containing the positional information
                   (range gate and azimuth angle) and line-of-sight wind speed.
        
        value    - CNR threshold value.
        
    Output:
    ------
        df_prime - Filtered Pandas dataframe. 
    """
    #Output DataFrame
    df_prime=df.copy()
    # Extraction of V_LOS from input DataFrame
    vel = df_prime.ws
    # Mask for V_LOS values with CNR values outside the valid interval
    mask_cnr = (df.CNR<=value[0]) | (df.CNR>=value[1])
    # Convertion of Mask DataFrame columns and indexes to input-like 
    mask_cnr.columns = vel.columns
    # Output DataFrame masking
    df_prime.ws = vel.mask(mask_cnr)
    return df_prime

# In[Median-like filter]

def data_filt_median(df,feature='ws',lim_m=6,lim_t = 10, lim_g = 100, n=10, m = 10, t = 10):
    """
    Filter based on 2-level thresholds for the difference in V_LOS and the moving V_LOS median
    (1st-level filter, acting in both, line-of-sight and azimuth components), and a global V_LOS 
    median (2nd-level filter, acting only in line-of-sight component).
    
    Input:
    -----
        df       - Pandas dataframe containing the positional information
                   (range gate and azimuth angle) and line-of-sight wind speed.
        
        feature  - Feature to use to estimate the median. Default is V_LOS,
                   which is what we are interested in now.
        
        lim_m    - Threshold for the differnce between measured V_LOS and moving median of V_LOS.
        
        lim_g    - Threshold for the differnce between measured V_LOS and global median of V_LOS.
        
        n        - Window size for moving median calculation.
        
    Output:
    ------
        df_prime - Filtered Pandas dataframe. 
    """
    # New DataFrame for output
    df_prime = df.copy()
    # Filter in line-of sight component
    # Local or moving median estimation
    mmr = df_prime[feature].rolling(n,axis=1,min_periods=1).median()
    # Global or line-of-sight median calculation
    gm = df_prime[feature].sub(df[feature].median(axis=1),axis=0)
    # filter in azimuth component
    # Local or moving median estimation
    mmp = df_prime.groupby('scan')[feature].rolling(m,axis=0,min_periods=1).median().reset_index(
            level='scan')[feature]
    
    # filter in time
    # Local or moving median estimation
    #df_dvdt = df_ws_grad(df_prime, grad = 'dvdt').abs()    
    #mmt = df_dvdt.groupby('azim')['dvdt'].rolling(t,center=True,axis=0).median().reset_index(level=('azim'))['dvdt']
    #mmt_std = df_dvdt.groupby('azim')['dvdt'].rolling(t,center=True,axis=0).std().reset_index(level=('azim'))['dvdt'] 
    
    # Substraction of V_LOS moving and global median from original DataFrame
    mmp = (df_prime[feature].sub(mmp))#.div(mmp)
    mmr = (df_prime[feature].sub(mmr))#.div(mmr)  
    #mmt = df_dvdt['dvdt'].sub(mmt.add(3*mmt_std))
  
    # Masking according to threshold values
    mask = []
    mask = (mmr.abs()>lim_m) | (gm.abs()>lim_g) | (mmp.abs()>lim_m)   
    # Output DataFrame masking
    df_prime.ws = df_prime.copy().ws.mask(mask) 
    return df_prime

# In[Clustering Filter with DBSCAN algorithm]
    
def data_filt_DBSCAN(df_ini,features,nn=5,eps=0.3,CNR_n=-24, plot=False, epsCNR=False, feat_lab = 'CNR', ran_feat = [-24, -8],just_noise=False):
    """
    Filter based on clustering algorithm Density-Based Spatial Clustering for Applications
    with Noise or DBSAN (Ester M. et al 1996). Each observation is represented as a point in a 
    k-dimensional space, when k features are specified as input. The DBSCAN algorithm sklearn library
    is used here. The minimum number of nearest neigbours, as recommended by (Ester M. et al 1996),
    is kept at 5, and the non-dimensional distance, eps, is calculated as the average between the 
    distance corresponing to the first knee in the k-distance graph, and a "noise level", which will
    depend on the fraction of "reliable" measurements or CNR values greater than the CNR_noise 
    threshold.
    
    Input:
    -----
        df_ini   - Pandas dataframe containing the positional information
                   (range gate and azimuth angle), line-of-sight wind speed, as well as all the
                   features included in the input, apart from "dvdr", "dvdt" and "movmed", which are
                   added to the input DataFrame inside this function, if required.
        
        feature  - List of all features that will be used to characterize observations.
        
        nn       - Minimum number of neighbouring points to define a "core" point. Default is 5.
        
        eps      - Non-dimensional distance within the nn points should lay in a "core" point. 
        
        CNR_n    - CNR_n threshold that defines "reliable" observations
        
        plot     - Boolean. If True, will plot k-diatance plot (log-scale) and the eps final value
                   used by DBSCAN.
        
    Output:
    ------
        mask     - Pandas dataframe with boolean values to later use in masking the input dataframe,
                   df_ini.
    """
    # Additional features
    df = df_ini.copy()
    if 'dvdr' in features:
        df = df_ws_grad(df_ini, grad = 'dvdr')
    if 'dvdt' in features:
        df = df_ws_grad(df_ini, grad = 'dvdt')
    if 'movmed' in features:
        df = df_mov_med(df,'ws',10) 

    # Initialization of input array of high-dimensional datapoints for DBSCAN
    a,b = df[features[0]].values.shape 
    X = np.empty((a*b,len(features)))   
    # Data preparation
    # If input is already filtered
    ind = ~np.isnan(df.ws.values.flatten())
    X = np.empty((sum(ind),len(features)))
    # DBSCAN input filling with data form df
    for i,f in enumerate(features):
        if ((f =='elev') | (f =='scan')) | (f =='azim') : # Check elev or azim!! Make it general!!!!
            X[:,i] = np.array([list(df[f].values),]*b).transpose().flatten()[ind]
        else:
            X[:,i] = df[f].values.flatten()[ind]
    # After input creation, it is re-scaled to non-dimensional space.
    X = RobustScaler(quantile_range=(25, 75)).fit_transform(X) 
    # Tree creation for k-distance graph
    tree_X = KDTree(X)
    # first k distance estimation
    d,i = tree_X.query(tree_X.data,k = nn)  
    #k-nearest distance
    d=d[:,-1]
    #non-zero values
    d = d[d>0]
    # log transformation to level-up values and easier identification of "knees"
    # d is an array with k-distances sorted in increasing value.
    d = np.log(np.sort(d))
    # x axis (point label)
    l = np.arange(0,len(d)) 
    rsmpl = 200
    # Down sampling to speed up calculations
    # same with point lables
    l_resample = np.unique(np.r_[l[::int(len(l)/rsmpl)],l[-1]])
#    print(l_resample,l_resample.shape)
    d_resample = d[l_resample]
    # Cubic spline of resampled k-distances, lower memory usage and higher calculation speed.
    #spl = UnivariateSpline(l_resample, d_resample,s=0.5)
    std=.001*np.ones_like(d_resample)
    # Changes in slope in the sorted, log transformed, k-distance graph
    t = np.arange(l_resample.shape[0])    
    fx = UnivariateSpline(t, l_resample/(-l_resample[0]+l_resample[-1]), k=4, w=1/std)
    fy = UnivariateSpline(t, d_resample/(-d_resample[0]+d_resample[-1]), k=4, w=1/std)
    x_1prime = fx.derivative(1)(t)
    x_2prime = fx.derivative(2)(t)
    y_1prime = fy.derivative(1)(t)
    y_2prime = fy.derivative(2)(t) 
    kappa = (x_1prime* y_2prime - y_1prime* x_2prime) / np.power(x_1prime**2 + y_1prime**2, 1.5)
#    print(np.min(d),np.max(d),(-d_resample[0]+d_resample[-1]))
#    ##
#    fig, ax1 = plt.subplots()
#    ax1.plot(l_resample,kappa)
#    plt.grid()
#    ax2 = ax1.twinx()
#    ax2.plot(l,100*d)
#    ax2.plot(l_resample,100*d_resample, 'o')
#    fig.tight_layout()
#    
    # location of knee (first point with k-distance above 1 std of k-distance mean)
    #ind_kappa = np.argsort(np.abs(kappa-(np.mean(kappa)+1*np.std(kappa)))) 
    ind_kappa, _ = find_peaks(kappa,prominence=1) 
    # Just after half of the graph
    ind_kappa = ind_kappa[ind_kappa>int(.3*len(kappa))]
    # The first knee...
    l1 = l_resample[ind_kappa][0]
    
#    plt.plot([l1,l1],100*np.array([min(d),max(d)]))
    
    # the corresponding eps distance
    eps0 = np.exp(d[l1])
    eps1 = eps0
    # Noise level is also taken into account
    if epsCNR: 
        x = np.sum(df.CNR.values>=-24)/len(df.CNR.values.flatten())
        # Some limits that can be changed. The distance as function of x, eps(x) is assumed linear.
        # The upper limit x=1 (no noise) specifies eps = 0.53 (this value is obtained just from the data 
        # and is very large). The other bound is x = 0 and eps = 0.075 a very small value.
        c = .075  
        m = .45
        # eps(x) linear relation
        eps = m*x+c
        # Fraction of reliable measurements
        # if reliable data is less than 30% then just the noise level criteria eps(x) is used
        #(it is indeed more strict)
        if (x<.3):
            eps1=eps
    else:
        eps = eps0    
        
    # If not CNR criteria is used    
    # DBSCAN object creation, the eps parameter is chosen as the average between the value
    # estimated with the noise-knee criteria, eps0 and the noise-level criteria, eps.
    clf = []
    clf = DBSCAN(eps=(eps1+eps)*.5, min_samples=nn)
    clf.fit(X)
    # Cluster-core samples and boundary samples identification
    core_samples_mask = np.zeros_like(clf.labels_, dtype=bool)
    core_samples_mask[clf.core_sample_indices_] = True
    # Array with labels identifying the different clusters in the data
    labels = clf.labels_
    # Number of identified clusters (excluding noise)
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0) 
    # Informative plots after clustering:
    #       - k-distance plot with eps value using both criterias, and the average value.
    #       - Scatter plot in 3 dimensions (more features are not shown)  
    if plot:        
        fig, ax = plt.subplots(figsize = (8,8))
        ax.plot(d,label = '$Data$', c = 'k')
        ax.plot([0,X.shape[0]],[np.log(eps),np.log(eps)],'--', c = 'red',
                label = r'$\log(\varepsilon_{CNR})$')
        ax.plot([0,X.shape[0]],[np.log(eps0),np.log(eps0)],'--', c = 'blue',
                label = r'$\log(\varepsilon_{knee})$')
#        if eps != eps1:
#            ax.plot([0,X.shape[0]],[np.log(.5*(eps+eps1)),np.log(.5*(eps+eps1))],
#                    c = 'grey', label = r'$\log(.5(\varepsilon_{CNR}+\varepsilon_{knee}))$')
        ax.set_xlabel('$Data\:point$', fontsize=16, weight='bold')
        ax.set_ylabel('$log(k-dist)$', fontsize=16, weight='bold')
        ax.set_xlim([-X.shape[0]*.03,X.shape[0]*1.03])
        ax.legend(loc=(.15,.8),fontsize=16)
        ax.text(0.05, 0.95, '(a)', transform=ax.transAxes, fontsize=24,
        verticalalignment='top')
        ax.annotate('$Knee$', xy=(l1, d[l1]), xytext=(10000, 0), fontsize = 16,
            arrowprops=dict(facecolor='grey',arrowstyle = "->"),)
        ax.tick_params(labelsize = 16)
        fig.tight_layout()

        for i,f in enumerate(features):
            if (f =='azim') | (f =='scan') :
                X[:,i] = np.array([list(df[f].values),]*b).transpose().flatten()[ind]
            else:
                X[:,i] = df[f].values.flatten()[ind]
        unique_labels = set(labels)
        colors = [plt.cm.Set1(each)
              for each in np.linspace(0, 1, len(unique_labels))]
        fig = plt.figure(figsize = (16,16))
        f = 16
        ax = fig.add_subplot(111, projection='3d')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(False)
        ax.view_init(30, 50)
        #ax.set_title('Estimated number of clusters: %d' % n_clusters_)
        ax.set_xlabel('$V_{LOS}\:[m/s]$', fontsize=f, weight='bold')
        ax.set_ylabel('$Range\:gate\:[m]$', fontsize=f, weight='bold')
        ax.set_zlabel('$CNR$', fontsize=f, weight='bold')
        ax.text(-20, 7000,250, '(b)', transform=ax.transAxes, fontsize=32,
        verticalalignment='top')
#        fig.tight_layout()
        for k, col in zip(unique_labels, colors):
#            if (k!=0) & (k!=-1):
                al = 1
                if k == -1:
                    # Black used for noise.
                    col = [0, 0, 0, 1]
                    al = .3
                class_member_mask = (labels == k)
                xy = X[class_member_mask & core_samples_mask]
                ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2], marker = 'o', c = tuple(col), edgecolor='k',
                           s=20, alpha=al)
                xy = X[class_member_mask & ~core_samples_mask]
                ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2], marker = 'o', c = tuple(col), edgecolor='k',
                           s=6, alpha = al)  
    # First reliable value and the corresponding cluster to which it belongs
    if feat_lab in features:
        
        ind =  ((df[feat_lab].values.flatten()>ran_feat[0]) &
                (df[feat_lab].values.flatten()<ran_feat[1])) & (labels != -1)
        if sum(ind)>0:
            lab = sp.stats.mode(labels[ind])[0]
        else:
            lab = np.nan
    else:
        ind =  (df.range_gate.values.flatten()<300) & (labels != -1)  
        if sum(ind)>0:
            lab = sp.stats.mode(labels[ind])[0]
        else:
            lab = np.nan
    # 2-D array
    labels = np.reshape(labels,(a,b))
    # Mask and re index and column names
    if just_noise:
        mask = pd.DataFrame(labels==-1)        
    else:
        mask = pd.DataFrame(labels!=lab)
    mask.columns = ['ws']*b
    mask=mask.set_index(df.ws.index)
    return mask

# In[Interpolation]    
    
def data_interp_kdtree(df_ini,dt,r=[],phi=[],col_r=[],col_phi=[]):
    """
    Interpolation of missing data after filtering. The interpolation uses a kdtree approach and it
    is done both, spatial and temporal. The "distances" in time are estimated as the prodict of dt
    and V_LOS.
    
    Input:
    -----
        df_ini   - Filtered pandas dataframe containing the positional information
                   (range gate and azimuth angle, optional), line-of-sight wind speed. Index are scans
        
        r        - Numpy array with line-of-sight range gates, if not contained in df.
        
        phi      - Azimuth angles of each laser beam, if not contained in df.
        
        dt       - Time elapsed between sucessive scans. 
        
    Output:
    ------
        df       - Pandas dataframe with missing data, after filtering, filled through interpolation.
        
    """
    print('1')
    df = df_ini#df_ini.copy()
    scan0 = df.index[0]
    #df.set_index('scan')
    # Initialization of the kdtree regressor. The number of neighbours, n_neighbours is set equal to
    # the number of corners and midpoints of a cube.
    print('2')
    neigh = KNeighborsRegressor(n_neighbors=26,weights='distance',algorithm='auto', leaf_size=30,
                                n_jobs=1)
    
    print('3')
    # If df has positional information
    
    if (len(r)==0) & (len(phi)==0):
        
        phi = np.pi-np.radians(df.loc[scan0][col_phi].unique())
        r = np.unique(df.loc[scan0][col_r].values)
    
        #r = np.unique(df[col_r].values).astype(float)
        #phi = np.unique(df[col_phi].values).astype(float)
    print('4')    
    # Meshgrid in polar coordinates and time.
    rg,phig,tg = np.meshgrid(r,phi,np.array([-dt,0,dt]))
    # Central slice in 3D array
    ind_slice = np.isnan(rg)
    ind_slice[:,:,1] = True
    ws_f = []
    for scan in np.unique(df.index.values):
        print(scan)
        #Initial and final scans are interpolated only in space
        if (scan == scan0) | (scan == np.unique(df.index.values)[-1]):
            rg0,phig0 = np.meshgrid(r,phi)
            rp = np.c_[rg0.flatten(),phig0.flatten()]  
            ws = df.ws.loc[scan].values.flatten()
            ind = np.isnan(ws)
            if sum(ind)>0:
                print(rp.shape,ind.shape)
                neigh.fit(rp[~ind,:], ws[~ind])
                # Kdtree Regression
                ws[ind] = neigh.predict(rp[ind,:])
            ws_f.append(np.reshape(ws,rg0.shape))
        # The rest of scans are interpolated in space and time        
        else:
            # 3D array 
            print(df.ws.loc[scan-1].values.shape,df.ws.loc[scan].values.shape,df.ws.loc[scan+1].values.shape)
            ws = np.dstack((df.ws.loc[scan-1].values, df.ws.loc[scan].values,df.ws.loc[scan+1].values))
            # Temporal "distance"
            temp = ws.flatten()*tg.flatten()
            # Invalid points are just put "far" away
            temp[np.isnan(temp)] = tg.flatten()[np.isnan(temp)]# !!!
            # Points in space and "time" to interpolate
            rpt = np.c_[rg.flatten()*np.cos(phig.flatten()),rg.flatten()*np.cos(phig.flatten()),temp]
            # Non valid V_LOS spotted
            ind = np.isnan(ws).flatten()
            # Non valid V_LOS in the current scan to be interpolated
            ind_int = (ind) & (ind_slice.flatten())
            # Interpolation is carried out
            if sum(ind_int)>0:
                ws = ws.flatten()
                neigh.fit(rpt[~ind,:], ws[~ind])
                ws[ind_int] = neigh.predict(rpt[ind_int,:])
            ws_f.append(np.reshape(ws,rg.shape)[:,:,1]) 
    # The wind speeds are replaced by the conyinous field...
    df.ws= pd.DataFrame(data = np.vstack(ws_f), index = df.ws.index, columns = df.ws.columns) 
    df = df.reset_index()
    return (df) 

