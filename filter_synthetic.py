# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 10:13:10 2018

@author: lalc
"""
from os.path import isfile, join, getsize
from os import listdir
import tkinter as tkint
import tkinter.filedialog
import matplotlib.pyplot as plt
import ppiscanprocess.filtering as fl
import pickle

import numpy as np
import pandas as pd
import scipy as sp

import matplotlib.ticker as ticker
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
from mpl_toolkits.axes_grid1 import make_axes_locatable 

# In[]

class FormatScalarFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, fformat="%1.1f", offset=True, mathText=True):
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self,useOffset=offset,
                                                        useMathText=mathText)
    def _set_format(self, vmin, vmax):
        self.format = self.fformat
        if self._useMathText:
            self.format = '$%s$' % matplotlib.ticker._mathdefault(self.format)

fmt = FormatScalarFormatter("%.2f")    
f = 24
def fm(x, pos=None):
    return r'${}$'.format('{:.1f}'.format(x).split('f')[0])

# In[Directory of the input and output data]

root = tkint.Tk()
file_in_path = tkint.filedialog.askdirectory(parent=root,title='Choose an Input dir')
root.destroy()

root = tkint.Tk()
file_out_path = tkint.filedialog.askdirectory(parent=root,title='Choose an Output dir')
root.destroy()

root = tkint.Tk()
file_out_path_df = tkint.filedialog.askdirectory(parent=root,title='Choose an Output dir DataFrame')
root.destroy()

root = tkint.Tk()
file_out_path_db = tkint.filedialog.askdirectory(parent=root,title='Choose an Output dir DataBase')
root.destroy()

root = tkint.Tk()
file_in_path_raw = tkint.filedialog.askdirectory(parent=root,title='Choose a sim. Input dir Raw')
root.destroy()

onlyfiles = [f for f in listdir(file_in_path) if isfile(join(file_in_path, f))]
onlyfiles_raw = [f for f in listdir(file_in_path_raw) if isfile(join(file_in_path_raw, f))]
# In[column labels]

iden_lab = np.array(['num1','num2','start_id','stop_id','start_time','stop_time','azim','elev'])
labels = iden_lab

#Labels for range gates and speed
vel_lab = np.array(['range_gate','ws','CNR','Sb'])

for i in np.arange(99):

    labels = np.concatenate((labels,vel_lab))
    
# In[]
    
filelist_s = [(filename,getsize(join(file_in_path,filename)))
             for filename in listdir(file_in_path) if getsize(join(file_in_path,filename))>1000]
size_s = list(list(zip(*filelist_s))[1])
filelist_s = list(list(zip(*filelist_s))[0])

# In[Different features of the DBSCAN filter]

feat0 = ['range_gate','ws','azim']
feat1 = ['range_gate','ws','dvdr']
feat2 = ['range_gate','azim','ws','dvdr']
    
# In[Geometry of scans]

r_0 = np.linspace(105,7000,198) # It is 105!!!!!!!!!!!!!!!!!!!!!!!!!
r_1 = np.linspace(105,7000,198)
phi_0 = np.linspace(256,344,45)*np.pi/180
phi_1 = np.linspace(196,284,45)*np.pi/180
r_0_g, phi_0_g = np.meshgrid(r_0,phi_0)
r_1_g, phi_1_g = np.meshgrid(r_1,phi_1)

# In[Routine for noisy dataframe and database creation]
############ Routine for noisy dataframe and database creation ################

iden_lab = np.array(['azim'])
labels = iden_lab
labels_new = iden_lab
#Labels for range gates and speed
labels_mask = []
labels_ws = []
labels_rg = []
for i in np.arange(198):
    vel_lab = np.array(['range_gate_'+str(i),'ws_'+str(i)])
    labels_new = np.concatenate((labels_new,vel_lab))
    labels = np.concatenate((labels,np.array(['range_gate','ws']))) 
    labels_rg = np.concatenate((labels_rg,np.array(['range_gate_'+str(i)])))
    labels_ws = np.concatenate((labels_ws,np.array(['ws_'+str(i)])))
labels_new = np.concatenate((labels_new,np.array(['scan'])))            
   
# Mann-model parameters
Dir = np.linspace(90,270,7)*np.pi/180
u_mean = 15
ae = [0.025, 0.05, 0.075]
L = [62,62.5,125,250,500,750,1000]
G = [0,1,2,2.5,3.5]
seed = np.arange(1,10)
ae,L,G,seed = np.meshgrid(ae,L,G,-seed)        
scan = 0
m = np.arange(3,597,3)
azim_unique = phi_0_g[:,0]*180/np.pi
df_noise = pd.DataFrame(columns=labels)
df_raw = pd.DataFrame(columns=labels)
aux_df_noise = np.zeros((len(azim_unique),len(labels)))
aux_df_raw = np.zeros((len(azim_unique),len(labels)))
indx = 0
scan = 0 
################              
param = []
for dir_mean in Dir:
    for ae_i,L_i,G_i,seed_i in zip(ae.flatten(),L.flatten(),G.flatten(),seed.flatten()):
        vlos0_file_name = 'vlos0'+str(u_mean)+str(int(dir_mean*180/np.pi))+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
        if vlos0_file_name in onlyfiles_raw:
            param.append([int(dir_mean*180/np.pi),u_mean,ae_i,L_i,G_i,seed_i,scan])
            scan+=1
param = np.array(param)         
#######################            
for dir_mean in Dir:
    for ae_i,L_i,G_i,seed_i in zip(ae.flatten(),L.flatten(),G.flatten(),seed.flatten()):
        vlos0_file_name = 'vlos0'+str(u_mean)+str(int(dir_mean*180/np.pi))+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
        if vlos0_file_name in onlyfiles_raw:
            param.append([dir_mean,u_mean,ae_i,L_i,G_i,seed_i,scan])
            n = 1
            aux_df_noise[:,0] = azim_unique
            aux_df_raw[:,0] = azim_unique
            dataws = np.reshape(np.fromfile(file_in_path+'/noise0_'+vlos0_file_name, dtype=np.float32),r_0_g.shape)
            dataws_raw = np.reshape(np.fromfile(file_in_path_raw+'/'+vlos0_file_name, dtype=np.float32),r_0_g.shape)
            datar = r_0_g
            for i in range(datar.shape[1]):
                aux_df_noise[:,n:n+2] = np.c_[datar[:,i],dataws[:,i]]
                aux_df_raw[:,n:n+2] = np.c_[datar[:,i],dataws_raw[:,i]]                       
                n = n+2
            df_noise = pd.concat([df_noise, pd.DataFrame(data=aux_df_noise,
                       index = indx+np.arange(datar.shape[0]),columns = labels)])  
            df_raw   = pd.concat([df_raw, pd.DataFrame(data=aux_df_raw,
                       index = indx+np.arange(datar.shape[0]),columns = labels)])  
            scan+=1
            indx = indx + np.arange(datar.shape[0]) + 1
            print(scan)

for dir_mean in Dir:
    for ae_i,L_i,G_i,seed_i in zip(ae.flatten(),L.flatten(),G.flatten(),seed.flatten()):
        vlos0_file_name = 'vlos0'+str(u_mean)+str(int(dir_mean*180/np.pi))+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
        if vlos0_file_name in onlyfiles_raw:
            param.append([dir_mean,u_mean,ae_i,L_i,G_i,seed_i,scan])
            scan+=1
            
df_noise['scan'] = df_noise.groupby('azim').cumcount()
df_raw['scan'] = df_raw.groupby('azim').cumcount()

df_noise.reset_index(inplace = True)
df_raw.reset_index(inplace = True)

# In[Filtering of synthetic and storing in database]
##############################################################################

with open(file_out_path_df+'/df0_noise.pkl', 'wb') as writer:
    pickle.dump(df_noise,writer)
with open(file_out_path_df+'/df0_raw.pkl', 'wb') as writer:
    pickle.dump(df_raw,writer)    


with open(file_out_path_df+'/df0_noise.pkl', 'rb') as reader:
    df_noise = pickle.load(reader)
with open(file_out_path_df+'/df0_raw.pkl', 'rb') as reader:
    df_raw = pickle.load(reader)

from sqlalchemy import create_engine

csv_database0 = create_engine('sqlite:///'+file_out_path+'/synthetic.db')

t_step = 3
mask0 = pd.DataFrame()
mask1 = pd.DataFrame()
mask2 = pd.DataFrame()
ind=np.unique(df_noise.scan.values)%t_step==0
times= np.unique(np.append(np.unique(df_noise.scan.values)[ind], df_noise.scan.values[-1]))  
     
for i in range(len(times)-1):
    print(times[i],times[i+1])
    if i == len(times)-2:
        loc = (df_noise.scan>=times[i]) & (df_noise.scan<=times[i+1])
    else:
        loc = (df_noise.scan>=times[i]) & (df_noise.scan<times[i+1])
    mask0 = pd.concat([mask0,fl.data_filt_DBSCAN(df_noise.loc[loc],feat0)])
    mask1 = pd.concat([mask1,fl.data_filt_DBSCAN(df_noise.loc[loc],feat1)])
    mask2 = pd.concat([mask2,fl.data_filt_DBSCAN(df_noise.loc[loc],feat2)])
mask0.columns = df_noise.ws.columns
mask1.columns = df_noise.ws.columns
mask2.columns = df_noise.ws.columns
df_noise.columns = np.concatenate((np.array(['index']),labels_new))
df_raw.columns = np.concatenate((np.array(['index']),labels_new))
df_raw.to_sql('raw', csv_database0, if_exists='append')
df_noise.to_sql('noise', csv_database0, if_exists='append')
df_noise.columns = np.concatenate((np.array(['index']),labels,np.array(['scan'])))

df_fil = df_noise.copy()
df_fil.ws = df_fil.ws.mask(mask0.ws)
#with open(file_out_path_df+'/df0_fil0.pkl', 'wb') as writer:
#    pickle.dump(df_fil,writer) 
df_fil.columns = np.concatenate((np.array(['index']),labels_new))
df_fil.to_sql('filtered0', csv_database0, if_exists='append')

df_fil = df_noise.copy()
df_fil.ws = df_fil.ws.mask(mask1.ws)
#with open(file_out_path_df+'/df0_fil1.pkl', 'wb') as writer:
#    pickle.dump(df_fil,writer) 
df_fil.columns = np.concatenate((np.array(['index']),labels_new))
df_fil.to_sql('filtered1', csv_database0, if_exists='append')

df_fil = df_noise.copy()
df_fil.ws = df_fil.ws.mask(mask2.ws)
#with open(file_out_path_df+'/df0_fil2.pkl', 'wb') as writer:
#    pickle.dump(df_fil,writer) 
df_fil.columns = np.concatenate((np.array(['index']),labels_new))
df_fil.to_sql('filtered2', csv_database0, if_exists='append')
  
df_median = fl.data_filt_median(df_noise, lim_m= 2.33 , n = 5, m = 3) 
#with open(file_out_path_df+'/df0_median.pkl', 'wb') as writer:
#    pickle.dump(df_median,writer)  
df_median.columns = np.concatenate((np.array(['index']),labels_new))
df_median.to_sql('filtered_med', csv_database0, if_exists='append') 

# In[Noise from database]
##############################################################################
from sqlalchemy import create_engine

csv_database0 = create_engine('sqlite:///'+file_out_path+'/synthetic.db')
col = 'SELECT '
i = 0
for w,r in zip(labels_ws, labels_rg):
    if i == 0:
        col = col + ' azim, ' + w + ', ' + r + ', '
    elif (i == len(labels_ws)-1):
        col = col + ' ' + w + ', ' + r + ', scan'
    else:
        col = col + ' ' + w + ', ' + r + ','       
    i+=1
selec_noise = col + ' FROM "noise"'
selec_raw = col + ' FROM "raw"'

df_noise = pd.read_sql_query(selec_noise, csv_database0)
df_raw = pd.read_sql_query(selec_raw, csv_database0)
iden_lab = np.array(['azim'])
labels = iden_lab
for i in np.arange(198):
    labels = np.concatenate((labels,np.array(['ws','range_gate']))) 
labels = np.concatenate((labels,np.array(['scan']))) 
df_noise.columns = labels
df_raw.columns = labels

# In[Noisy scans plots and noise visualization]
###############################################################################
scan_t = 2010
vmax = np.max(df_raw.loc[df_raw.scan==scan_t].ws.values)
vmin = np.min(df_raw.loc[df_raw.scan==scan_t].ws.values)
noise_id = (df_noise.loc[df_noise.scan==scan_t].ws.values-
            df_raw.loc[df_noise.scan==scan_t].ws.values) != 0
noisy = df_noise.loc[df_noise.scan==scan_t].ws.values
noisy[~noise_id] = np.nan
non_noisy = df_noise.loc[df_noise.scan==scan_t].ws.values
non_noisy[noise_id] = np.nan

phi_0 = np.where(np.pi/2-phi_1_g<0, 2*np.pi+(np.pi/2-phi_1_g), np.pi/2-phi_1_g)

f=24
fig, ax = plt.subplots(figsize = (9,9))
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
im0 = ax.scatter((r_0_g*np.cos(phi_0))[noise_id], (r_0_g*np.sin(phi_0))[noise_id],
                c = noisy[noise_id], cmap = 'Greys')
im = ax.scatter((r_0_g*np.cos(phi_0))[~noise_id], (r_0_g*np.sin(phi_0))[~noise_id], 
                c=non_noisy[~noise_id], cmap = 'jet')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad = 0.05)
cbar = fig.colorbar(im, cax=cax, format=ticker.FuncFormatter(fm))
cbar.ax.tick_params(labelsize=f)
ax.tick_params(labelsize=f)
cbar.ax.set_ylabel("$V_{LOS},\:$", fontsize=f)
cbar2.ax.set_ylabel("$V_{LOS},\:contaminated$", fontsize=f)
ax.set_ylabel('$x\:[m]$', fontsize=f, weight='bold')
ax.set_xlabel('$y\:[m]$', fontsize=f, weight='bold')
ax.text(0.05, 0.95, '(c)', transform=ax.transAxes, fontsize=32,verticalalignment='top')
fig.tight_layout()

# In[Sensitivity analysis of median filter]
##############################################################################
##############################################################################    
"""Sensitivity analysis median filter"""
##############################################################################
############################################################################## 

# In[Creation of a list of filtering results, very time consuming]
# Noise fraction   
#############################################################
noise_iden = (df_noise.ws.values-df_raw.ws.values) != 0
#############################################################   
n_w = np.array([3, 5, 7, 9, 11, 13])
m_w = np.array([3, 5, 7, 9, 11, 13])
lim = np.linspace(1,6,6)
n_w, m_w, lim = np.meshgrid(n_w,m_w, lim)

noise_det = []
not_noise_det = []
reliable_scan = []

chunk_size = 45
counting = 0

for nw, mw, l in zip(n_w.flatten(), m_w.flatten(), lim.flatten()):
    df_noise_median_i = fl.data_filt_median(df_noise,lim_m=l,lim_g=100,n=nw, m=mw)
    reliable = ~np.isnan(df_noise_median_i.ws.values) & ~noise_iden
    n = np.isnan(df_noise_median_i.ws.values) & noise_iden
    nn = np.isnan(df_noise_median_i.ws.values) & ~noise_iden
    reliable_scan.append([np.sum(reliable[i:i+chunk_size,:])/
                          np.sum(~noise_iden[i:i+chunk_size,:])
                          for i in range(0, n.shape[0], chunk_size)])
    noise_det.append([np.sum(n[i:i+chunk_size,:])/
                      np.sum(noise_iden[i:i+chunk_size,:])
                      for i in range(0, n.shape[0], chunk_size)])
    not_noise_det.append([np.sum(nn[i:i+chunk_size,:])/
                          np.sum(~noise_iden[i:i+chunk_size,:])
                          for i in range(0, n.shape[0], chunk_size)])
    counting+=1
    print(counting)
# In[Storing of results]
with open(file_out_path+'/reliable_scan0.pkl', 'rb') as reader:
    reliable_scan = pickle.load(reader)
with open(file_out_path+'/noise_det0.pkl', 'rb') as reader:
    noise_det = pickle.load(reader)      
with open(file_out_path+'/not_noise_det0.pkl', 'rb') as reader:
    not_noise_det = pickle.load(reader)
# In[2D arrays for sensitivity analysis]    
noise_weight = np.array([np.sum(noise_iden[i:i+chunk_size,:])/
                         len(noise_iden[i:i+chunk_size,:].flatten())
                         for i in range(0, noise_iden.shape[0], chunk_size)])
mean_noise_det = np.reshape(np.array([np.mean(nd) for nd in noise_det]),n_w.shape)   
mean_not_noise_det = np.reshape(np.array([np.mean(nd) for nd in not_noise_det]),n_w.shape)  
mean_reliable = np.reshape(np.array([np.mean(rel) for rel in reliable_scan]),n_w.shape)
mean_tot_w = np.reshape(np.array([np.mean((rel*(1-noise_weight)+nd*noise_weight))
                        for rel,nd in zip(np.array(reliable_scan),np.array(noise_det))]),n_w.shape) 
# In[Interpolation for sensitivity analysis]
###############################################################################
n_w = np.array([3, 5, 7, 9, 11, 13])
m_w = np.array([3, 5, 7, 9, 11, 13])
lim = np.linspace(0,6,6)
n_w, m_w, lim = np.meshgrid(n_w,m_w, lim)
nml = np.c_[n_w.flatten(), m_w.flatten(), lim.flatten()]
#############################
opt = []
for n in np.array([3, 5, 7, 9, 11, 13]):
    n_w = n
    m_w = np.linspace(3,13,80)#np.array([3, 5, 7, 9, 11, 13])
    lim = np.linspace(1,6,80)
    n_w, m_w, lim = np.meshgrid(n_w,m_w, lim) 
    nml0 = np.c_[n_w.flatten(), m_w.flatten(), lim.flatten()]
    mean_tot = np.reshape(sp.interpolate.griddata(nml,mean_tot_w, (nml0[:,0], nml0[:,1], nml0[:,2]), method='linear'),n_w.shape)  
    mean_rel = np.reshape(sp.interpolate.griddata(nml,mean_reliable, (nml0[:,0], nml0[:,1], nml0[:,2]), method='linear'),n_w.shape)  
    mean_noise = np.reshape(sp.interpolate.griddata(nml,mean_noise_det, (nml0[:,0], nml0[:,1], nml0[:,2]), method='linear'),n_w.shape)  
    m_w, lim, mean_tot, mean_rel, mean_noise = np.squeeze(m_w), np.squeeze(lim), np.squeeze(mean_tot), np.squeeze(mean_rel), np.squeeze(mean_noise) 
 
    low = np.min(mean_tot)  
    optimal = np.r_[np.c_[lim.flatten(),m_w.flatten()][np.argmax(mean_tot)],n,
                          mean_noise.flatten()[np.argmax(mean_tot)],mean_rel.flatten()[np.argmax(mean_tot)],
                          np.max(mean_tot)]
    opt.append(np.r_[np.max(mean_tot),optimal])

# In[1D graph]    
l = lim.flatten()
nw = n_w.flatten()/l
mw = m_w.flatten()/l
tot = (mean_tot).flatten()
noise = (mean_noise).flatten()
rec = (mean_rel).flatten()
#################################
def mov_ave(mylist,N):
    cumsum, moving_aves = [0], []
    for i, x in enumerate(mylist, 1):
        cumsum.append(cumsum[i-1] + x)
        if i>=N:
            moving_ave = (cumsum[i] - cumsum[i-N])/N
            #can do stuff with moving_ave here
            moving_aves.append(moving_ave)
    return np.array(moving_aves)
##################################Figure eta curve
indexnw = np.argsort(nw)
indexmw = np.argsort(mw)
indexnmw = np.argsort(nw*mw*l)

totm = mov_ave(tot[indexnmw], 20)
noisem = mov_ave(noise[indexnmw], 20)
recm = mov_ave(rec[indexnmw], 20)
off = -totm.shape[0]+tot[indexnmw].shape[0] 
############ Some figures
# In[Figures sensitivity analysis and noise fraction]


fig, ax = plt.subplots(figsize = (8,8))
ax.hist(noise_weight,bins=50,histtype='step',lw=2, color = 'k')
ax.set_xlabel('$f_{noise}$', fontsize=16)
ax.set_ylabel('$Counts$', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.text(0.05, 0.95, '(b)', transform=ax.transAxes, fontsize=24,verticalalignment='top')
fig.tight_layout()
        
fig, ax = plt.subplots(figsize=(8,8))
ax.plot((nw*mw*l)[indexnmw][off:],totm, label = '$\eta_{tot}$', c = 'k', lw=2)
ax.scatter((nw*mw*l)[indexnmw],tot[indexnmw],s = 2, c = 'k', alpha = .2)
ax.plot((nw*mw*l)[indexnmw][off:],noisem, label = '$\eta_{noise}$', c = 'r', lw=2)
ax.scatter((nw*mw*l)[indexnmw], noise[indexnmw],s = 2, c = 'r', alpha = .2)
ax.plot((nw*mw*l)[indexnmw][off:],recm, label = '$\eta_{rec}$', c = 'b', lw=2)
ax.scatter((nw*mw*l)[indexnmw],rec[indexnmw],s = 2, c = 'b', alpha = .2)

ax.set_xscale('log')
ax.set_xlabel('$n_rn_\phi/\Delta V_{LOS,threshold}$',fontsize = 16)
ax.set_ylabel('$\eta$',fontsize = 16)
ax.legend(fontsize = 16)
ax.tick_params(labelsize = 16)
ax.set_xlim(2,100)
ax.set_ylim(.4, 1)
fig.tight_layout()    

nml_opt = optimal[1]*optimal[2]/optimal[0]
ax.scatter(nml_opt,optimal[-1],s = 100, c = 'grey',edgecolors = 'k', alpha = 1,lw = 4)
ax.scatter(nml_opt,optimal[3],s = 100, c ='grey',edgecolors = 'r', alpha = 1,lw = 4)
ax.scatter(nml_opt,optimal[4],s = 100, c ='grey',edgecolors = 'b', alpha = 1,lw = 4)
ax.text(0.05, 0.95, '(a)', transform=ax.transAxes, fontsize=24,verticalalignment='top')

##############################################################################
##############################################################################    
"""End of Sensitivity analysis median filter"""
# In[End of Sensitivity analysis median filter]
##############################################################################
############################################################################## 
# In[Loading dataframes]

with open(file_out_path_df+'/df0_noise.pkl', 'rb') as reader:
    df_noise = pickle.load(reader)
with open(file_out_path_df+'/df0_raw.pkl', 'rb') as reader:
    df_raw = pickle.load(reader)
with open(file_out_path_df+'/df0_median.pkl', 'rb') as reader:
    df_median = pickle.load(reader)
with open(file_out_path_df+'/df0_fil2.pkl', 'rb') as reader:
    df_fil = pickle.load(reader)    
chunk_size = 45    
noise_iden = (df_noise.ws.values-df_raw.ws.values) != 0  
noise_weight = np.array([np.sum(noise_iden[i:i+chunk_size,:])/
                         len(noise_iden[i:i+chunk_size,:].flatten())
                         for i in range(0, noise_iden.shape[0], chunk_size)])  
noise_det_clust = np.isnan(df_fil.ws.values) & noise_iden
recov_rel_clust = ~np.isnan(df_fil.ws.values) & ~noise_iden
noise_det_median = np.isnan(df_median.ws.values) & noise_iden
recov_rel_median = ~np.isnan(df_median.ws.values) & ~noise_iden
noise_iden_clust = np.array([np.sum(noise_det_clust[i:i+chunk_size,:])/
                      np.sum(noise_iden[i:i+chunk_size,:])
                      for i in range(0, noise_det_clust.shape[0], chunk_size)])     
relia_iden_clust = np.array([np.sum(recov_rel_clust[i:i+chunk_size,:])/
                      np.sum(~noise_iden[i:i+chunk_size,:])
                      for i in range(0, recov_rel_clust.shape[0], chunk_size)])
noise_iden_median = np.array([np.sum(noise_det_median[i:i+chunk_size,:])/
                      np.sum(noise_iden[i:i+chunk_size,:])
                      for i in range(0, noise_det_clust.shape[0], chunk_size)])    
relia_iden_median = np.array([np.sum(recov_rel_median[i:i+chunk_size,:])/
                      np.sum(~noise_iden[i:i+chunk_size,:])
                      for i in range(0, recov_rel_clust.shape[0], chunk_size)])
tot_clust = noise_weight*noise_iden_clust + (1-noise_weight)*relia_iden_clust
tot_median = noise_weight*noise_iden_median + (1-noise_weight)*relia_iden_median

# In[Histograms]
########################################################################################
f = 24
Density = False
n_bin = 100
fig, ax = plt.subplots(figsize=(8,8))
bins = np.linspace(np.min(np.r_[noise_iden_clust,noise_iden_median]),
                   np.max(np.r_[noise_iden_clust,noise_iden_median]),n_bin)
ax.hist(noise_iden_clust,bins=bins, histtype = 'step',
                            label = 'Clustering', lw = 2, color = 'r', density = Density)
ax.hist(noise_iden_median,bins=bins, histtype = 'step',
                    label = 'Median', lw = 2, color = 'k', density = Density)
fig.legend(loc = (.2,.7),fontsize = f)
ax.tick_params(axis='both', which='major', labelsize = f)
ax.set_xlabel('$\eta_{noise}$',fontsize = f)
ax.set_ylabel('$Prob.\:density$',fontsize = f)
ax.text(0.05, 0.95, '(a)', transform=ax.transAxes, fontsize=28,
    verticalalignment='top')
fig.tight_layout()    

fig, ax = plt.subplots(figsize=(8,8))
bins = np.linspace(np.min(np.r_[relia_iden_clust,relia_iden_median]),
                   np.max(np.r_[relia_iden_clust,relia_iden_median]),n_bin)
ax.hist(relia_iden_clust,bins=bins, histtype = 'step', label = 'Clustering', lw = 2, color = 'r', density = Density)
ax.hist(relia_iden_median,bins=bins, histtype = 'step', label = 'Median', lw = 2, color = 'k', density = Density)
fig.legend(loc = (.2,.7),fontsize = f)
ax.tick_params(axis='both', which='major', labelsize = f)
ax.set_xlabel('$\eta_{rec}$',fontsize = f)
ax.set_ylabel('$Prob.\:density$',fontsize = f)
ax.text(0.05, 0.95, '(b)', transform=ax.transAxes, fontsize=28,
    verticalalignment='top')
fig.tight_layout()    

fig, ax = plt.subplots(figsize=(8,8))
bins = np.linspace(np.min(np.r_[tot_clust,tot_median]),
                   np.max(np.r_[tot_clust,tot_median]),n_bin)
ax.hist(tot_clust,bins=bins, histtype = 'step', label = 'Clustering', lw = 2, color = 'r', density = Density)
ax.hist(tot_median,bins=bins, histtype = 'step', label = 'Median', lw = 2, color = 'k', density = Density)
fig.legend(loc = (.2,.7),fontsize = f)
ax.tick_params(axis='both', which='major', labelsize = f)
ax.set_xlabel('$\eta_{tot}$',fontsize = f)
ax.set_ylabel('$Prob.\:density$',fontsize = f)
ax.text(0.05, 0.95, '(c)', transform=ax.transAxes, fontsize=28,
    verticalalignment='top')
fig.tight_layout()