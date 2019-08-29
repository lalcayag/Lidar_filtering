# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 20:31:30 2018

@author: lalc

"""
import numpy as np
import pandas as pd
import scipy as sp
import pickle
import tkinter as tkint
import tkinter.filedialog

import matplotlib.ticker as ticker
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True


import ppiscanprocess.filtering as filt
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.neighbors.kde import KernelDensity
from sklearn.preprocessing import RobustScaler
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True

# In[For figures]

import mpl_toolkits.mplot3d.art3d as art3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

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

def fm(x, pos=None):
    return r'${}$'.format('{:.2f}'.format(x).split('f')[0])

# In[CSV Data loading]

root = tkint.Tk()
file_df = tkint.filedialog.askopenfilenames(parent=root,title='Choose a data file')
root.destroy()
root = tkint.Tk()
file_mask = tkint.filedialog.askopenfilenames(parent=root,title='Choose a mask file')
root.destroy()

# In[Labels for CSV files of PPI scans]

iden_lab = np.array(['start_id','stop_id','start_time','stop_time','azim','elev'])
labels = iden_lab
vel_lab = np.array(['range_gate','ws','CNR','Sb'])
for i in np.arange(198):
    labels = np.concatenate((labels,vel_lab))
sirocco_loc = np.array([6322832.3,0])
vara_loc = np.array([6327082.4,0])
d = vara_loc-sirocco_loc

# In[First figures]
DF = []

for f,m in zip(file_df,file_mask):
    with open(m, 'rb') as reader:
        mask = pickle.load(reader)        
    df = pd.read_csv(f,sep=";", header=None) 
    df.columns = labels      
    df['scan'] = df.groupby('azim').cumcount()
  
    mask_CNR = (df.CNR>-24) & (df.CNR<-8)
    mask_CNR.columns =  mask.columns
    mask.mask(mask_CNR,other=False,inplace=True)
    df.ws=df.ws.mask(mask)
    DF.append(df)
    
################ Figure for CNR comb #############
   
def mycmap(c,alpha):
    import matplotlib
    c = []
    for i in range(alpha.shape[0]):
            c.append(matplotlib.colors.to_rgba('grey', alpha=alpha[i]))
    return c    
#30 scans
ind_can = (df.scan > 9000) & (df.scan < 9030)
df_sel = df.loc[ind_can]
df= []

df_sel_mask = df_sel.copy()
df_sel_vlos = df_sel.copy()
df_sel_clust = df_sel.copy()
mask_CNR = (df_sel.CNR>-27)
mask_vlos = (df_sel.ws>-21) & (df_sel.ws<0)
mask_CNR.columns =  df_sel_mask.ws.columns
df_sel_mask.ws=df_sel_mask.ws.mask(~mask_CNR)
df_sel_vlos.ws=df_sel_vlos.ws.mask(~mask_vlos)
df_sel_clust.ws=df_sel_vlos.ws.mask(mask)

scan_n = 9020

phi0 = df_sel.azim.unique()-90
r0 = df_sel.iloc[0].range_gate.values
r_0, phi_0 = np.meshgrid(r0, np.radians(phi0)) # meshgrid

ind_can_s = df_sel.scan==scan_n

shape = df_sel.loc[ind_can_s].ws.values.shape
pos = np.c_[df_sel.loc[ind_can_s].ws.values.flatten(),
            df_sel.loc[ind_can_s].CNR.values.flatten()]
kernel = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(pos)
Z = kernel.score_samples(pos)
Z = np.exp(Z)

f=16
fig, ax = plt.subplots(2,2,figsize=(16,16))
ax[0,0].set_aspect('equal')
ax[0,0].use_sticky_edges = False
ax[0,0].margins(0.1)
im = ax[0,0].contourf(r_0*np.cos(phi_0),r_0*np.sin(phi_0),df_sel.ws.loc[df_sel.scan==scan_n].values,cmap='jet')
ax[0,0].set_xlabel('$Easting\:m$', fontsize=f)
ax[0,0].set_ylabel('$Northing\:m$', fontsize=f)
divider = make_axes_locatable(ax[0,0])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(im, cax=cax, format=ticker.FuncFormatter(fm))
cbar.ax.tick_params(labelsize=f)
ax[0,0].tick_params(labelsize=14)
cbar.ax.set_ylabel("$V_{LOS}$ m/s", fontsize=f)
ax[0,0].text(0.05, 0.95, '(a)', transform=ax[0,0].transAxes, fontsize=24,
        verticalalignment='top')


ax[0,1].set_aspect('equal')
ax[0,1].use_sticky_edges = False
ax[0,1].margins(0.1)
im = ax[0,1].scatter(pos[:,0],pos[:,1], s=20, c = np.log(Z), cmap='jet')  #df.loc[ind_can].plot.scatter(x='ws', y='CNR', c='grey', s=20, edgecolor='k',cmap = 'jet')  
ax[0,1].set_xlabel('$V_{LOS}$ m/s',fontsize=16) 
ax[0,1].set_ylabel('$CNR\:dB$',fontsize=16)  
ax[0,1].plot([-40,40],[-27,-27],'--r',lw=2)
ax[0,1].plot([-21,-21],[-40,-0],'--k',lw=2)
ax[0,1].plot([0,0],[-40,-0],'--k',lw=2)
ax[0,1].set_xlim([-35,35])
ax[0,1].set_ylim([-40,0])
ax[0,1].tick_params(axis='both', which='major', labelsize=14)
divider = make_axes_locatable(ax[0,1])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(im, cax=cax, format=ticker.FuncFormatter(fm))
cbar.ax.tick_params(labelsize=f)
ax[0,1].tick_params(labelsize=14)
cbar.ax.set_ylabel("log(KDE)", fontsize=f)
ax[0,1].text(0.05, 0.95, '(b)', transform=ax[0,1].transAxes, fontsize=24,
        verticalalignment='top')

ax[1,0].set_aspect('equal')
ax[1,0].use_sticky_edges = False
ax[1,0].margins(0.1)
im = ax[1,0].contourf(r_0*np.cos(phi_0),r_0*np.sin(phi_0),df_sel_mask.ws.loc[df_sel_mask.scan==scan_n].values,cmap='jet')
ax[1,0].set_xlabel('$Easting\:m$', fontsize=f)
ax[1,0].set_ylabel('$Northing\:m$', fontsize=f)
divider = make_axes_locatable(ax[1,0])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(im, cax=cax, format=ticker.FuncFormatter(fm))
cbar.ax.tick_params(labelsize=f)
ax[1,0].tick_params(labelsize=14)
cbar.ax.set_ylabel("$V_{LOS}\:m/s$", fontsize=f)
ax[1,0].text(0.05, 0.95, '(c)', transform=ax[1,0].transAxes, fontsize=24,
        verticalalignment='top')

ax[1,1].set_aspect('equal')
ax[1,1].use_sticky_edges = False
ax[1,1].margins(0.1)
im = ax[1,1].contourf(r_0*np.cos(phi_0),r_0*np.sin(phi_0),df_sel_vlos.ws.loc[df_sel_vlos.scan==scan_n].values,cmap='jet')
ax[1,1].set_xlabel('$Easting\:m$', fontsize=f)
ax[1,1].set_ylabel('$Northing\:m$', fontsize=f)
divider = make_axes_locatable(ax[1,1])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(im, cax=cax, format=ticker.FuncFormatter(fm))
cbar.ax.tick_params(labelsize=f)
ax[1,1].tick_params(labelsize=14)
cbar.ax.set_ylabel("$V_{LOS}\:m/s$", fontsize=f)
ax[1,1].text(0.05, 0.95, '(d)', transform=ax[1,1].transAxes, fontsize=24,
        verticalalignment='top')
fig.tight_layout()

########################
# 3D plot
ind_can_s = df_sel.scan==scan_n

shape = df_sel.loc[ind_can_s].ws.values.shape
pos3d = np.c_[df_sel.loc[ind_can_s].ws.values.flatten(), df_sel.loc[ind_can_s].range_gate.values.flatten(),
            df_sel.loc[ind_can_s].CNR.values.flatten()]
X = RobustScaler(quantile_range=(25, 75)).fit_transform(pos3d) 
kernel = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X)
Z = kernel.score_samples(X)
size = (np.exp(Z)-np.min(np.exp(Z)))/(np.max(np.exp(Z))-np.min(np.exp(Z)))

fig = plt.figure()
fig.set_size_inches(20,8)
xx,yy,zz = np.meshgrid(np.array([-35,35]),np.array([0,7000]),np.array([-40,-27,0]))

ax = fig.add_subplot(1,2,1,projection='3d')
ax.voxels(xx, yy, zz, zz[:-1,:-1,:-1] < -27, facecolors='grey', alpha=.1, edgecolor = 'k')
ax.scatter(pos3d[:,0], pos3d[:,1], pos3d[:,2], s=10, c= np.exp(Z),cmap = 'jet',zorder = 1)
ax.set_xlabel('$V_{LOS}$ m/s',fontsize=16) 
ax.set_zlabel('CNR dB',fontsize=16)  
ax.set_ylabel('Range gate m',fontsize=16)  
ax.set_xlim([-35,35])
ax.set_ylim([7000,0])
ax.tick_params(axis='both', which='major', labelsize=14)
ax.text(10, 10, 250, '(a)', transform=ax.transAxes, fontsize=24,
        verticalalignment='top')
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.grid(False)

xx,yy,zz = np.meshgrid(np.array([-35,-21,0,35]),np.array([0,7000]),np.array([-40,0]))
cub = (xx>=0) | (xx<-21)

ax = fig.add_subplot(1,2,2,projection='3d')
ax.scatter(pos3d[:,0], pos3d[:,1], pos3d[:,2], s=10, c= np.exp(Z),cmap = 'jet')
ax.voxels(xx, yy, zz, cub[:-1,:-1,:-1], facecolors='grey', alpha=.1, edgecolor = 'k')
ax.set_xlabel('$V_{LOS}$ m/s',fontsize=16) 
ax.set_zlabel('CNR dB',fontsize=16)  
ax.set_ylabel('Range gate m',fontsize=16)  
ax.set_xlim([-35,35])
ax.set_ylim([7000,0])
ax.tick_params(axis='both', which='major', labelsize=14)
ax.text(10, 10, 250, '(b)', transform=ax.transAxes, fontsize=24,
        verticalalignment='top')
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.view_init(30, -100)
ax.grid(False)

fig.tight_layout()

###########################################################################
# In[Synthetic lidars filtering test]
############################## 
##############################    
# In[Noise stats and figures synthetic]
############
#Noise stats
root = tkint.Tk()
file_out_path_df = tkint.filedialog.askdirectory(parent=root,title='Choose an Output dir DataFrame')
root.destroy()
with open(file_out_path_df+'/df0_noise.pkl', 'rb') as reader:
    df_noise = pickle.load(reader)
with open(file_out_path_df+'/df0_raw.pkl', 'rb') as reader:
    df_raw = pickle.load(reader)
with open(file_out_path_df+'/df0_median.pkl', 'rb') as reader:
    df_median = pickle.load(reader)
with open(file_out_path_df+'/df0_fil2.pkl', 'rb') as reader:
    df_fil = pickle.load(reader)

ind_noise = (df_noise.ws-df_raw.ws) != 0
noise_iden = (df_noise.ws.values-df_raw.ws.values) != 0

ws_raw_syn = df_raw.ws.values[~noise_iden]
ws_noise_syn = df_noise.ws.values[noise_iden]
noise_syn = (df_noise.ws.values-df_raw.ws.values)[noise_iden]
ws_median_syn = df_median.ws.values
ws_clust_syn = df_fil.ws.values
quantiles = np.quantile(ws_raw_syn.flatten(),q = [1-.997,.997])
quantiles_noise = np.quantile(noise_syn.flatten(),q = [1-.997,.997])

plt.figure()
h_raw,bine_raw,_ = plt.hist(ws_raw_syn.flatten(),bins=50,alpha=0.5,density=True)
h_noise_pure,bine_noise_pure,_ = plt.hist(noise_syn.flatten(),bins=50,alpha=0.5,density=True)
h_noise,bine_noise,_ = plt.hist(ws_noise_syn.flatten(),bins=50,alpha=0.5,density=True)
plt.close()

f=24
fig, ax = plt.subplots(figsize = (8,8))
ax.step(.5*(bine_raw[1:]+bine_raw[:-1]),h_raw, color='black', lw=3, label = r'$Contaminated$')
ax.step(.5*(bine_noise[1:]+bine_noise[:-1]),h_noise, color='red', lw=3, label = r'$Non\:contaminated$')
ax.fill_between([-33,quantiles[0]], [0,0], [.1,.1], facecolor='grey', alpha=.2)
ax.fill_between([quantiles[1],33], [0,0], [.1,.1], facecolor='grey', alpha=.2)
ax.set_ylim(0,.1)
ax.set_xlim(-33,33)
ax.set_xlabel('$V_{LOS}$',fontsize=f)
ax.set_ylabel('$Probability\:density$',fontsize=f)
ax.tick_params(labelsize=f)
ax.text(0.05, 0.95, '(b)', transform=ax.transAxes, fontsize=32, verticalalignment='top')
fig.tight_layout()

fig, ax = plt.subplots(figsize = (8,8))
ax.step(.5*(bine_noise_pure[1:]+bine_noise_pure[:-1]),h_noise_pure, color='black', lw=3, label = r'$Noise$')
ax.fill_between([-33,quantiles_noise[0]], [0,0], [.1,.1], facecolor='grey', alpha=.2)
ax.fill_between([quantiles_noise[1],33], [0,0], [.1,.1], facecolor='grey', alpha=.2)
ax.set_ylim(0,.07)
ax.set_xlim(-33,33)
ax.set_xlabel('$\Delta V_{LOS, noise}$',fontsize=f)
ax.set_ylabel('$Probability\:density$',fontsize=f)
ax.tick_params(labelsize=f)
ax.text(0.05, 0.95, '(a)', transform=ax.transAxes, fontsize=32, verticalalignment='top')
fig.tight_layout()
   
# In[Real data from a database]
# Phase 1
###########################################################################################
from sqlalchemy import create_engine
from datetime import datetime, timedelta

root = tkint.Tk()
file_in_path_db = tkint.filedialog.askdirectory(parent=root,title='Choose an Input dir')
root.destroy()
root = tkint.Tk()
file_in_path_df_out = tkint.filedialog.askdirectory(parent=root,title='Choose an dataframe output dir')
root.destroy()
root = tkint.Tk()
file_in_path_figures = tkint.filedialog.askdirectory(parent=root,title='Choose an Input dir')
root.destroy()

# In[]
iden_lab = np.array(['stop_time','azim'])
labels = iden_lab
labels_new = iden_lab
#Labels for range gates and speed
labels_mask = []
labels_ws = []
labels_rg = []
labels_CNR = []
for i in np.arange(198):
    vel_lab = np.array(['range_gate_'+str(i),'ws_'+str(i),'CNR_'+str(i),'Sb_'+str(i)])
    mask_lab = np.array(['ws_mask'+str(i)])
    labels_new = np.concatenate((labels_new,vel_lab))
    labels = np.concatenate((labels,np.array(['ws','range_gate','CNR'])))
    labels_mask = np.concatenate((labels_mask,mask_lab))
    labels_ws = np.concatenate((labels_ws,np.array(['ws_'+str(i)])))
    labels_rg = np.concatenate((labels_rg,np.array(['range_gate_'+str(i)])))
    labels_CNR = np.concatenate((labels_CNR,np.array(['CNR_'+str(i)])))
labels_new = np.concatenate((labels_new,np.array(['scan']))) 
labels = np.concatenate((labels,np.array(['scan'])))

csv_database_0 = create_engine('sqlite:///'+file_in_path_db+'/raw_filt_0.db')

col = 'SELECT '
i = 0
for w,r,c in zip(labels_ws,labels_rg,labels_CNR):
    if i == 0:
        col = col + 'stop_time,' + ' azim, ' + w + ', ' + r + ', ' + c + ', '
    elif (i == len(labels_ws)-1):
        col = col + ' ' + w + ', ' + r + ', ' + c + ', scan'
    else:
        col = col + ' ' + w + ', ' + r + ',' + c + ', '        
    i+=1
scann = int(24*60*60/45)
chunksize = 45*scann
tot_scans = 10000
tot_iter = int(tot_scans*45/chunksize)
off = 0
lim = [-10,-24]
selec_fil = col + ' FROM "table_fil"'
selec_raw = col + ' FROM "table_raw"'
df_0 = pd.DataFrame()
df_0_raw = pd.DataFrame()
for i in range(tot_iter):
    print(off/45)
    df = pd.read_sql_query(selec_fil+' LIMIT ' + str(int(chunksize)) + ' OFFSET ' + str(int(off)), csv_database_0)
    df_raw = pd.read_sql_query(selec_raw+' LIMIT ' + str(int(chunksize)) + ' OFFSET ' + str(int(off)), csv_database_0)
    df.columns = labels
    df_raw.columns = labels
    df_0 = pd.concat([df_0,df])
    df_0_raw = pd.concat([df_0_raw,df_raw])
    off+=chunksize    
df = None
df_raw = None

# In[Storing filt and raw]
########################################################    
with open(file_in_path_df_out+'/df_0_phase2.pkl', 'wb') as dfs:
    pickle.dump(df_0,dfs)
with open(file_in_path_df_out+'/df_0_raw_phase2.pkl', 'wb') as dfs:
    pickle.dump(df_0_raw,dfs) 
#######################################################
## In[]
df_median1 = filt.data_filt_median(df_0_raw, lim_m= 2.33 , n = 5, m = 3) 
ind_cnr29 = (df_0_raw.CNR <-29) | (df_0_raw.CNR > 0)
ind_cnr29.columns = df_0_raw.ws.columns
df_29 = df_0_raw.copy()
df_29.ws = df_29.ws.mask(ind_cnr29)   
ind_cnr29 = None
###################################################################
#######################################################
with open(file_in_path_df_out+'/df_median1_phase2.pkl', 'wb') as dfs:
    pickle.dump(df_median1,dfs)    
with open(file_in_path_df_out+'/df_29_phase2.pkl', 'wb') as dfs:
    pickle.dump((df_29),dfs)  
###################################################################
# In[Loading of dataframes]
#######################################################
with open(file_in_path_df_out+'/df_0_phase1.pkl', 'rb') as dfs:
    df_0 = pickle.load(dfs)
with open(file_in_path_df_out+'/df_0_raw_phase1.pkl', 'rb') as dfs:
    df_0_raw = pickle.load(dfs)    
with open(file_in_path_df_out+'/df_median1_phase1.pkl', 'rb') as dfs:
    df_median1 = pickle.load(dfs)    
with open(file_in_path_df_out+'/df_29_phase1.pkl', 'rb') as dfs:
    df_29 = pickle.load(dfs) 
######################################################  
with open(file_in_path_df_out+'/df_0_phase2.pkl', 'rb') as dfs:
    df_0 = pickle.load(dfs)
with open(file_in_path_df_out+'/df_0_raw_phase2.pkl', 'rb') as dfs:
    df_0_raw = pickle.load(dfs)    
with open(file_in_path_df_out+'/df_median1_phase2.pkl', 'rb') as dfs:
    df_median1 = pickle.load(dfs)    
with open(file_in_path_df_out+'/df_29_phase2.pkl', 'rb') as dfs:
    df_29 = pickle.load(dfs) 
######################################################      
# In[Recovery and filtering statistics, first, identification of bad scans]
rel_scan = []
scans = df_0_raw.scan.unique()
chunk = len(df_0_raw.azim.unique())
rt = df_0_raw.loc[df_0_raw.scan==0].ws.values.flatten().shape[0]
ind_cnr = np.sum(((df_0_raw.CNR >-24) & (df_0_raw.CNR < -8)).values, axis = 1)
rel_scan = ind_cnr.reshape((int(len(ind_cnr)/chunk),chunk)).sum(axis=1)/rt
ind_cnr = []
ind_scan = np.isin(df_0_raw.scan.values, scans[rel_scan>.30])
ind = ((df_0_raw.loc[ind_scan].CNR<-24)|(df_0_raw.loc[ind_scan].CNR>-8)).values

indn_clust = np.isnan(df_0.loc[ind_scan].ws.values)
indn_median1 = np.isnan(df_median1.loc[ind_scan].ws.values) 
indn_29 = np.isnan(df_29.loc[ind_scan].ws.values) 

ws_clust_noise = df_0_raw.loc[ind_scan].ws.values[indn_clust]
ws_median1_noise = df_0_raw.loc[ind_scan].ws.values[indn_median1]
ws_29_noise = df_0_raw.loc[ind_scan].ws.values[indn_29]

ws_raw = df_0_raw.loc[ind_scan].ws.values[ind]
ws_clust = df_0.loc[ind_scan].ws.values[ind]
ws_median1 = df_median1.loc[ind_scan].ws.values[ind]
ws_29 = df_29.loc[ind_scan].ws.values[ind]

ws_raw_g = df_0_raw.loc[ind_scan].ws.values[~ind]
ws_clust_g = df_0.loc[ind_scan].ws.values[~ind]
ws_median_g = df_median1.loc[ind_scan].ws.values[~ind]
ws_29_g = df_29.loc[ind_scan].ws.values[~ind]
quantiles = np.quantile(ws_raw_g.flatten(),q = [1-.997,.997])

df_0_raw = []
df_29 = []
df_0 = []
df_median1 = []

# In[Revocery, statistics]
den=False

plt.figure()

h_raw,bine,_ = plt.hist(ws_raw.flatten(),bins=60,alpha=0.5,density=den,label='1')
h_med1,_,_ = plt.hist(ws_median1.flatten(),bins=bine,alpha=0.5,density=den,label='1')
h_clust,_,_ = plt.hist(ws_clust.flatten(),bins=bine,alpha=0.5,density=den)
h_29,_,_ = plt.hist(ws_29.flatten(),bins=bine,alpha=0.5,density=den)


plt.close()
plt.figure()
h_raw_g,bine_g,_ = plt.hist(ws_raw_g.flatten(),bins=60,alpha=0.5,density=den)
h_med_g1,_,_ = plt.hist(ws_median_g.flatten(),bins=bine_g,alpha=0.5,density=den)
h_clust_g,_,_ = plt.hist(ws_clust_g.flatten(),bins=bine_g,alpha=0.5,density=den)
h_29_g,_,_ = plt.hist(ws_29_g.flatten(),bins=bine_g,alpha=0.5,density=den)

plt.close()

# In[Data recovered in numbers and Kolmogorov distance and K-L metric]

indvalid_clust = ~np.isnan(ws_clust.flatten())
indquant_clust = ~((ws_clust.flatten()[indvalid_clust] > quantiles[0]) &
                   (ws_clust.flatten()[indvalid_clust] < quantiles[1])) 
indvalid_med1 = ~np.isnan(ws_median1.flatten())
indquant_med1 = ~((ws_median1.flatten()[indvalid_med1] > quantiles[0]) &
                  (ws_median1.flatten()[indvalid_med1] < quantiles[1]))
indvalid_29 = ~np.isnan(ws_29.flatten())
indquant_29 = ~((ws_29.flatten()[indvalid_29] > quantiles[0]) &
                  (ws_29.flatten()[indvalid_29] < quantiles[1]))
indquant_raw = ~((ws_raw.flatten() > quantiles[0]) & (ws_raw.flatten() < quantiles[1])) 

fraction_clust = np.sum(indquant_clust)/np.sum(indquant_raw)
fraction_med1 = np.sum(indquant_med1)/np.sum(indquant_raw)
fraction_29 = np.sum(indquant_29)/np.sum(indquant_raw)
fraction_noise = np.sum(indquant_raw)/len(indquant_raw)

fraction_recov_clust = np.sum(indvalid_clust)/len(ws_raw_g.flatten())
fraction_recov_median = np.sum(indvalid_med1)/len(ws_raw_g.flatten())
fraction_recov_29 = np.sum(indvalid_29)/len(ws_raw_g.flatten())

recov_results = [[fraction_clust,fraction_recov_clust],
                 [fraction_med1,fraction_recov_median],
                 [fraction_29,fraction_recov_29]]

from scipy import stats
nn = len(ws_raw_g.flatten())
mm = np.sum(~np.isnan(ws_clust.flatten()))
pp = np.sum(~np.isnan(ws_median1.flatten()))
qq = len(ws_raw.flatten())
num = np.log(nn+mm)
den = np.log(nn)+np.log(mm)
alpha = .05
c_alpha = np.sqrt(-.5*np.log(alpha))
test = c_alpha*np.exp(.5*(num-den))
print(test)
al = 1
ks = []
pv = []
for i in range(2000):
    print(i)
    ind0 = np.random.randint(0,nn,size = int(nn*al))
    ind1 = np.random.randint(0,mm,size = int(mm*al))
    ind2 = np.random.randint(0,pp,size = int(pp*al))
    ind3 = np.random.randint(0,qq,size = int(qq*al))
    
    ks_raw, p_value_raw = stats.ks_2samp(ws_raw_g.flatten()[ind0], ws_raw.flatten()[ind3])
    ks_clust, p_value_clust = stats.ks_2samp(ws_raw_g.flatten()[ind0],
                                             ws_clust.flatten()[~np.isnan(ws_clust.flatten())][ind1])
    ks_med, p_value_med = stats.ks_2samp(ws_raw_g.flatten()[ind0],
                                             ws_median1.flatten()[~np.isnan(ws_median1.flatten())][ind2])
    ks.append([ks_raw,ks_clust,ks_med])
    pv.append([p_value_raw,p_value_clust,p_value_med])
    
plt.figure()
plt.hist(np.array(ks)[:,0],histtype = 'step',bins=50)
plt.hist(np.array(ks)[:,1],histtype = 'step',bins=50)
plt.hist(np.array(ks)[:,2],histtype = 'step',bins=50)
plt.figure()
plt.hist(np.array(pv)[:,0],histtype = 'step',bins=50)
plt.hist(np.array(pv)[:,1],histtype = 'step',bins=50)
plt.hist(np.array(pv)[:,2],histtype = 'step',bins=50)

ks_raw, p_value_raw = stats.ks_2samp(ws_raw_g.flatten(), ws_raw.flatten())
ks_clust, p_value_clust = stats.ks_2samp(ws_raw_g.flatten(),
                                         ws_clust.flatten()[~np.isnan(ws_clust.flatten())])
ks_med, p_value_med = stats.ks_2samp(ws_raw_g.flatten(),
                                         ws_median1.flatten()[~np.isnan(ws_median1.flatten())])
ks_29, p_value_med = stats.ks_2samp(ws_raw_g.flatten(),
                                         ws_29.flatten()[~np.isnan(ws_29.flatten())])
###################
#K-L metric
def kl_divergence(p, q):
    return np.sum(np.where((p != 0) & (q != 0) , p * np.log(p / q), 0))
den = True
h_raw_g,bine_g,_ = plt.hist(ws_raw_g.flatten(),bins=60,alpha=0.5,density=den)
h_raw,_,_ = plt.hist(ws_raw.flatten(),bins=bine_g,alpha=0.5,density=den)
h_med1,_,_ = plt.hist(ws_median1.flatten()[~np.isnan(ws_median1.flatten())],bins=bine_g,alpha=0.5,density=den)
h_clust,_,_ = plt.hist(ws_clust.flatten()[~np.isnan(ws_clust.flatten())],bins=bine_g,alpha=0.5,density=den)
h_29,_,_ = plt.hist(ws_29.flatten()[~np.isnan(ws_29.flatten())],bins=bine_g,alpha=0.5,density=den)

kl_raw0 = kl_divergence(h_raw_g, h_raw)
kl_clust0 = kl_divergence(h_raw_g, h_clust)
kl_med0 = kl_divergence(h_raw_g, h_med1)
kl_290 = kl_divergence(h_raw_g, h_29)

kl_raw1 = kl_divergence(h_raw, h_raw_g)
kl_clust1 = kl_divergence(h_clust, h_raw_g)
kl_med1 = kl_divergence(h_med1, h_raw_g)

#############################################################
# In[Recovery non reliable]
h_med0 = h_med1
fig, ax1 = plt.subplots(figsize=(8,8))
#ax1.step(.5*(bine[1:]+bine[:-1])[h_med0>0],h_med0[h_med0>0]/h_raw[h_med0>0],color='black', lw=3, label = '$(4,\:3,\:3)$')
ax1.step(.5*(bine[1:]+bine[:-1])[h_med0>0],h_29[h_med0>0]/h_raw[h_med0>0],color='black', lw=3, label = '$CNR\:>\:-29\:dB$')
ax1.step(.5*(bine[1:]+bine[:-1])[h_med0>0],h_med1[h_med0>0]/h_raw[h_med0>0],color='blue', lw=3, label = '$Median\:filter$')
ax1.step(.5*(bine[1:]+bine[:-1])[h_med0>0],h_clust[h_med0>0]/h_raw[h_med0>0],color='red', lw=3, label = '$Clustering\:filter$')
ax1.fill_between([-30,quantiles[0]], [0,0], [1,1], facecolor='grey', alpha=.2)
ax1.fill_between([quantiles[1],35], [0,0], [1,1], facecolor='grey', alpha=.2)
tol = .3

x1 = (.5*(bine[1:]+bine[:-1])[h_med0>0])[.5*(bine[1:]+bine[:-1])[h_med0>0]>=quantiles[1]-tol]
y1 = (h_med1[h_med0>0]/h_raw[h_med0>0])[.5*(bine[1:]+bine[:-1])[h_med0>0]>=quantiles[1]-tol]
y2 = (h_29[h_med0>0]/h_raw[h_med0>0])[.5*(bine[1:]+bine[:-1])[h_med0>0]>=quantiles[1]-tol]
ax1.fill_between(x1, y1, y2, step = 'pre', facecolor='green', alpha=.5)
x1 = (.5*(bine[1:]+bine[:-1])[h_med0>0])[.5*(bine[1:]+bine[:-1])[h_med0>0]<=quantiles[0]+tol]
y1 = (h_med1[h_med0>0]/h_raw[h_med0>0])[.5*(bine[1:]+bine[:-1])[h_med0>0]<=quantiles[0]+tol]
y2 = (h_29[h_med0>0]/h_raw[h_med0>0])[.5*(bine[1:]+bine[:-1])[h_med0>0]<=quantiles[0]+tol]
ax1.fill_between(x1, y1, y2, step = 'pre', facecolor='green', alpha=.5)

x1 = (.5*(bine[1:]+bine[:-1])[h_med0>0])[.5*(bine[1:]+bine[:-1])[h_med0>0]>=quantiles[1]-tol]
y1 = (h_clust[h_med0>0]/h_raw[h_med0>0])[.5*(bine[1:]+bine[:-1])[h_med0>0]>=quantiles[1]-tol]
y2 = (h_med1[h_med0>0]/h_raw[h_med0>0])[.5*(bine[1:]+bine[:-1])[h_med0>0]>=quantiles[1]-tol]
ax1.fill_between(x1, y1, y2, step = 'pre', facecolor='grey', alpha=.8)
x1 = (.5*(bine[1:]+bine[:-1])[h_med0>0])[.5*(bine[1:]+bine[:-1])[h_med0>0]<=quantiles[0]+tol]
y1 = (h_clust[h_med0>0]/h_raw[h_med0>0])[.5*(bine[1:]+bine[:-1])[h_med0>0]<=quantiles[0]+tol]
y2 = (h_med1[h_med0>0]/h_raw[h_med0>0])[.5*(bine[1:]+bine[:-1])[h_med0>0]<=quantiles[0]+tol]
ax1.fill_between(x1, y1, y2, step = 'pre', facecolor='grey', alpha=.8)

ax1.set_xlabel('$V_{LOS}$',fontsize=16)
ax1.set_ylabel('$Data\:recovery\:fraction$',fontsize=16)
ax1.legend(loc=(.62,.6),fontsize=16)
ax1.set_xlim(-30,35)
ax1.set_ylim(0,1)
ax1.tick_params(labelsize=16)
ax1.text(0.05, 0.95, '(b)', transform=ax1.transAxes, fontsize=24, verticalalignment='top')
fig.tight_layout()
fig.savefig(file_in_path_figures+'/non_rel_comp_phase2.png')

############################
# In[Recovery of reliable]
fig, ax = plt.subplots(figsize=(8,8))
ax.step(.5*(bine_g[1:]+bine_g[:-1])[h_raw_g>1],h_med_g1[h_raw_g>1]/h_raw_g[h_raw_g>1],color='blue',lw=3,label = '$Median\:filter$')
ax.step(.5*(bine_g[1:]+bine_g[:-1])[h_raw_g>1],h_clust_g[h_raw_g>1]/h_raw_g[h_raw_g>1],color='red',lw=3,label='$Clustering\:filter$')
ax.fill_between([-30,quantiles[0]], [0,0], [1.1,1.1], facecolor='grey', alpha=.2)
ax.fill_between([quantiles[1],35], [0,0], [1.1,1.1], facecolor='grey', alpha=.2)
ax.set_xlabel('$V_{LOS}$',fontsize=16)
ax.set_ylabel('$Data\:recovery\:fraction$',fontsize=16)
ax.legend(loc=(.62,.6),fontsize=16)
ax.set_xlim(-30,35)
ax.set_ylim(0,1)
ax.tick_params(labelsize=16)
ax.text(0.05, 0.95, '(a)', transform=ax.transAxes, fontsize=24, verticalalignment='top')
tol = .5
x1 = .5*(bine_g[1:]+bine_g[:-1])[h_med_g1>0]
ind = (x1>=quantiles[0]-tol) & (x1<=quantiles[1]+tol)
x1 = x1[ind]
y1 = (h_clust_g[h_med_g1>0]/h_raw_g[h_med_g1>0])[ind]
y2 = (h_med_g1[h_med_g1>0]/h_raw_g[h_med_g1>0])[ind]
ax.fill_between(x1, y1, y2, step = 'pre', facecolor='grey', alpha=.8)

fig.tight_layout()
fig.savefig(file_in_path_figures+'/rel_comp_phase2.png')
####################
# In[Pdf os reliable and non reliable]
h_raw_g_d,bine_g_d,_ = plt.hist(ws_raw_g.flatten(),bins=50,alpha=0.5,density=True)
h_raw_d,bine_d,_ = plt.hist(ws_raw.flatten()[ws_raw.flatten()<50],bins=50,alpha=0.5,density=True)
plt.close()
fig, ax2 = plt.subplots(figsize=(8,8))
ax2.step(.5*(bine_g_d[1:]+bine_g_d[:-1]),h_raw_g_d,color='black',lw=3,label = r'$CNR\:\in:[-24,-8]$ dB')
ax2.step(.5*(bine_d[1:]+bine_d[:-1]),h_raw_d,color='red',lw=3,label = r'$CNR\:\not\in:[-24,-8]$ dB')
ax2.fill_between([.5*(bine_d[1:]+bine_d[:-1]).min(),quantiles[0]], [0,0], [.1,.1], facecolor='grey', alpha=.2)
ax2.fill_between([quantiles[1],.5*(bine_d[1:]+bine_d[:-1]).max()], [0,0], [.1,.1], facecolor='grey', alpha=.2)
ax2.set_ylim(0,.1)
ax2.set_xlim(-30,30)
ax2.set_xlabel('$V_{LOS}$',fontsize=16)
ax2.set_ylabel('$Probability\:density$',fontsize=16)
ax2.legend(loc=(.5,.7),fontsize=16)
ax2.tick_params(labelsize=16)
ax2.text(0.05, 0.95, '(a)', transform=ax2.transAxes, fontsize=24, verticalalignment='top')
fig.tight_layout()
fig.savefig(file_in_path_figures+'/fig_rel_hist_phase1.png')

###################################
# In[Pdf of noise detected]
h_n_clust,binenc,_ = plt.hist(ws_clust_noise.flatten(),bins=50,alpha=0.5,density=True)
h_n_med,binenm,_ = plt.hist(ws_median1_noise.flatten(),bins=50,alpha=0.5,density=True)
h_n_29,binen29,_ = plt.hist(ws_29_noise.flatten(),bins=50,alpha=0.5,density=True)
h_raw,bine,_ = plt.hist(ws_raw.flatten(),bins=50,alpha=0.5,density=True)
plt.close()
fig, ax2 = plt.subplots(figsize=(8,8))
ax2.step(.5*(binen29[1:]+binen29[:-1]), h_n_29,color='black',lw=3,label = r'$Noise\:detected\:CNR\:>\:29$')
ax2.step(.5*(binenc[1:]+binenc[:-1]), h_n_clust,color='red',lw=3,label = r'$Noise\:detected\:cultering$')
ax2.step(.5*(binenm[1:]+binenm[:-1]), h_n_med,color='blue',lw=3,label = r'$Noise\:detected\:median$')
ax2.step(.5*(bine[1:]+bine[:-1]), h_raw,color='green',lw=3,label = r'$Noise\:detected\:CNR\:>\:24$')
ax2.set_ylim(0,.1)
ax2.set_xlim(-35,35)
ax2.set_xlabel('$V_{LOS}$',fontsize=16)
ax2.set_ylabel('$Probability\:density$',fontsize=16)
ax2.legend(loc=(.5,.7),fontsize=16)
ax2.tick_params(labelsize=16)
ax2.text(0.05, 0.95, '(a)', transform=ax2.transAxes, fontsize=24, verticalalignment='top')
fig.tight_layout()
fig.savefig(file_in_path_figures+'/fig_rel_hist_noise_phase1.png')

###################################
###############################################################################
# In[Recovery, spatial]
###############################################################################
def wrap_angle(x):
    return np.where(x<0 , 2*np.pi+x, x)

r = df_0_raw.loc[ind_scan].range_gate.values
a = (np.ones((r.shape[1],1))*df_0_raw.loc[ind_scan].azim.values.flatten()).transpose()

n = len(np.unique(df_0_raw.loc[ind_scan].scan.values))
phi1w = df_0_raw.azim.unique()
r1w = np.unique(df_0_raw.loc[df_0_raw.scan == 0].range_gate.values)

df_0_raw = []

r_siw, phi_siw = np.meshgrid(r1w, np.flip(wrap_angle(np.radians(90-phi1w)))) # meshgrid

phi_siw0 = wrap_angle((90-np.arange(194, 284, 2))*np.pi/180)

r_siw0, phi_siw0 = np.meshgrid(r1w, phi_siw0)


mask_29 = np.isnan(df_29.loc[ind_scan].ws).values
df_29 = []
mask_29 = (~mask_29.flatten()).astype(int)
df_s_29 = pd.DataFrame({'r': r.flatten(), 'a': a.flatten(), 'm': mask_29})
mask_29 = []
recovery_29 = df_s_29.groupby(['a', 'r'])['m'].agg('sum').values/n
df_s_29 = []
recovery_29 = np.reshape(recovery_29, r_siw.shape) 

mask_median1 = np.isnan(df_median1.loc[ind_scan].ws).values
df_median1 = []
mask_median1 = (~mask_median1.flatten()).astype(int)
df_s_median1 = pd.DataFrame({'r': r.flatten(), 'a': a.flatten(), 'm': mask_median1})
mask_median1 = []
recovery_median1 = df_s_median1.groupby(['a', 'r'])['m'].agg('sum').values/n
df_s_median1 = []
recovery_median1 = np.reshape(recovery_median1, r_siw.shape) 

mask = np.isnan(df_0.loc[ind_scan].ws).values
df_0 = []
mask = (~mask.flatten()).astype(int)
df_s_clust = pd.DataFrame({'r': r.flatten(), 'a': a.flatten(), 'mask': mask.flatten()})
mask = []
recovery_clust=np.reshape(df_s_clust.groupby(['a', 'r'])['mask'].agg('sum').values,r_siw.shape)
recovery_clust=recovery_clust/n
df_s_clust = []

# In[Storing and loading]
#######################################################
with open(file_in_path_df_out+'/recovery_29_phase1.pkl', 'wb') as dfs:
    pickle.dump(recovery_29,dfs)
    
with open(file_in_path_df_out+'/recovery_clust_phase1.pkl', 'wb') as dfs:
    pickle.dump(recovery_clust,dfs)   
    
with open(file_in_path_df_out+'/recovery_median1_phase1.pkl', 'wb') as dfs:
    pickle.dump(recovery_median1,dfs) 
########################################################    
with open(file_in_path_df_out+'/recovery_29_phase2.pkl', 'wb') as dfs:
    pickle.dump(recovery_29,dfs)
    
with open(file_in_path_df_out+'/recovery_clust_phase2.pkl', 'wb') as dfs:
    pickle.dump(recovery_clust,dfs)   
    
with open(file_in_path_df_out+'/recovery_median1_phase2.pkl', 'wb') as dfs:
    pickle.dump(recovery_median1,dfs) 
######################################################## 
with open(file_in_path_df_out+'/recovery_29_phase1.pkl', 'rb') as dfs:
    recovery_29 = pickle.load(dfs)
    
with open(file_in_path_df_out+'/recovery_clust_phase1.pkl', 'rb') as dfs:
    recovery_clust = pickle.load(dfs)   
    
with open(file_in_path_df_out+'/recovery_median1_phase1.pkl', 'rb') as dfs:
    recovery_median1 = pickle.load(dfs)     
######################################################## 
with open(file_in_path_df_out+'/recovery_29_phase2.pkl', 'rb') as dfs:
    recovery_29 = pickle.load(dfs)
    
with open(file_in_path_df_out+'/recovery_clust_phase2.pkl', 'rb') as dfs:
    recovery_clust = pickle.load(dfs)   
    
with open(file_in_path_df_out+'/recovery_median1_phase2.pkl', 'rb') as dfs:
    recovery_median1 = pickle.load(dfs) 
#######################################################    
# In[Some plots]
i, j, k = 0, 1, 2  
letter = ['(a)', '(b)', '(c)']
lev = np.linspace(0.7, 1, 6)
im = [[],[],[]]
limits = [-3400,-2400,500,1500]
cx0 = .5*(limits[0]+limits[1])
cy0 = .5*(limits[2]+limits[3])
cx1 = -6600
cy1 = -1600
cx2 = -150
cy2 = 200
r0 = np.sqrt((limits[1]-limits[0])**2+(limits[3]-limits[2])**2)/2
r1 = 200
r2 = 200

fig, ax = plt.subplots(1,k+1,figsize=((k+1)*8,8))
ax[i].set_aspect('equal')
ax[i].use_sticky_edges = False
ax[i].margins(0.1)
im[i] = ax[i].contourf(r_siw*np.cos(phi_siw),r_siw*np.sin(phi_siw),
                   np.reshape(recovery_29,phi_siw.shape),levels=lev,cmap='jet')
ax[i].set_xlabel(r'$West-East\:[m]$', fontsize = 16, weight='bold')
ax[i].set_ylabel(r'$North-South\:[m]$', fontsize = 16, weight='bold') 
ax[i].tick_params(labelsize = 16)
ax[i].text(0.05, 0.95, letter[i], transform=ax[i].transAxes, fontsize=24,
        verticalalignment='top') 
#fig, ax2 = plt.subplots()
ax[j].set_aspect('equal')
ax[j].use_sticky_edges = False
ax[j].margins(0.1)
im[j] = ax[j].contourf(r_siw*np.cos(phi_siw),r_siw*np.sin(phi_siw),
                   np.reshape(recovery_median1,phi_siw.shape),levels = lev,cmap='jet')
ax[j].set_xlabel(r'$West-East\:[m]$', fontsize=16, weight='bold')
ax[j].set_ylabel(r'$North-South\:[m]$', fontsize=16, weight='bold') 
ax[j].tick_params(labelsize = 16)
ax[j].text(0.05, 0.95, letter[j], transform=ax[1].transAxes, fontsize=24,
        verticalalignment='top') 

circlej0 = plt.Circle((cx0, cy0), r0, facecolor='none',edgecolor='k', lw = 2)
circlej1 = plt.Circle((cx1, cy1), r1, facecolor='none',edgecolor='k', lw = 2)
circlej2 = plt.Circle((cx2, cy2), r2, facecolor='none',edgecolor='k', lw = 2)
ax[j].add_artist(circlej0)
ax[j].add_artist(circlej1)
ax[j].add_artist(circlej2)

ax[k].set_aspect('equal')
ax[k].use_sticky_edges = False
ax[k].margins(0.1)
im[k] = ax[k].contourf(r_siw0*np.cos(phi_siw),r_siw0*np.sin(phi_siw),
                   np.reshape(recovery_clust,phi_siw.shape),levels = lev,cmap='jet')
divider2 = make_axes_locatable(ax[k])
cax2 = divider2.append_axes("right", size="5%", pad=0.05)
cbar2 = fig.colorbar(im[k], cax=cax2, format=ticker.FuncFormatter(fm))
cbar2.ax.set_ylabel('$Data\:recovery\:fraction$', fontsize=16)
cbar2.ax.tick_params(labelsize=16)
ax[k].set_xlabel(r'$West-East\:[m]$', fontsize=16, weight='bold')
#ax[2].set_ylabel(r'$North-South\:[m]$', fontsize=16, weight='bold') 
ax[k].tick_params(labelsize = 16)
ax[k].text(0.05, 0.95, letter[k], transform=ax[k].transAxes, fontsize=24,
        verticalalignment='top')
circlek0 = plt.Circle((cx0, cy0), r0, facecolor='none',edgecolor='k', lw = 2)
circlek1 = plt.Circle((cx1, cy1), r1, facecolor='none',edgecolor='k', lw = 2)
circlek2 = plt.Circle((cx2, cy2), r2, facecolor='none',edgecolor='k', lw = 2)
ax[k].add_artist(circlek0)
ax[k].add_artist(circlek1)
ax[k].add_artist(circlek2)

fig.tight_layout()

# In[Detail of the turbines]
i, j, k = 0, 1, 1  
letter = ['(a)', '(b)', '(c)']
lev = np.linspace(0.1,.5 ,6)

im = [[],[],[]]

recovery_median10 = recovery_median1.copy
recovery_clust0 = recovery_clust
recovery_median10[recovery_median1>.5] = .5
recovery_clust0[recovery_clust>.5] = .5

fig, ax = plt.subplots(1,k+1,figsize=((k+1)*8,8))
ax[i].set_aspect('equal')
ax[i].use_sticky_edges = False
ax[i].margins(0.1)
im[i] = ax[i].contourf(r_siw*np.cos(phi_siw),r_siw*np.sin(phi_siw),
                   np.reshape(recovery_median10,phi_siw.shape),levels=lev,cmap='jet')
ax[i].set_xlabel(r'$West-East\:[m]$', fontsize = 16, weight='bold')
ax[i].set_ylabel(r'$North-South\:[m]$', fontsize = 16, weight='bold') 
ax[i].tick_params(labelsize = 16)
ax[i].text(0.05, 0.95, letter[i], transform=ax[i].transAxes, fontsize=24,
        verticalalignment='top') 
ax[i].set_xlim(limits[0],limits[1])
ax[i].set_ylim(limits[2],limits[3])

ax[k].set_aspect('equal')
ax[k].use_sticky_edges = False
ax[k].margins(0.1)
im[k] = ax[k].contourf(r_siw0*np.cos(phi_siw),r_siw0*np.sin(phi_siw),
                   np.reshape(recovery_clust0,phi_siw.shape),levels = lev,cmap='jet')
divider2 = make_axes_locatable(ax[k])
cax2 = divider2.append_axes("right", size="5%", pad=0.05)
cbar2 = fig.colorbar(im[k], cax=cax2, format=ticker.FuncFormatter(fm))
cbar2.ax.set_ylabel('$Data\:recovery\:fraction$', fontsize=16)
cbar2.ax.tick_params(labelsize=16)

ax[k].set_xlabel(r'$West-East\:[m]$', fontsize=16, weight='bold')
ax[k].tick_params(labelsize = 16)
ax[k].text(0.05, 0.95, letter[k], transform=ax[k].transAxes, fontsize=24,
        verticalalignment='top')
ax[k].set_xlim(limits[0],limits[1])
ax[k].set_ylim(limits[2],limits[3]) 
fig.tight_layout()

# In[Some special figures]

# In[Figures to ilustrate epsilon estimation]
#####################################
# 
scan_n = 300
loc = (df_0_raw.scan>=scan_n-1) & (df_0_raw.scan<=scan_n+1)
feat = ['ws','range_gate','CNR','azim','dvdr']

phi1w = df_0_raw.azim.unique()
r1w = np.unique(df_0_raw.loc[df_0_raw.scan == 0].range_gate.values)
r_0, phi_0 = np.meshgrid(r1w, wrap_angle(np.radians(90-phi1w))) # meshgrid

f=16
fig, ax = plt.subplots(figsize=(8,8))
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.1)
im = ax.contourf(r_0*np.cos(phi_0),r_0*np.sin(phi_0), 
       df_0_raw.ws.loc[df_0_raw.scan==scan_n].values,cmap='jet')
ax.set_xlabel('$Easting\:m$', fontsize=f)
ax.set_ylabel('$Northing\:m$', fontsize=f)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(im, cax=cax, format=ticker.FuncFormatter(fm))
cbar.ax.tick_params(labelsize=f)
ax.tick_params(labelsize=14)
cbar.ax.set_ylabel("$V_{LOS}$ m/s", fontsize=f)
ax.text(0.05, 0.95, '(a)', transform=ax.transAxes, fontsize=24,
        verticalalignment='top')
fig.tight_layout()
mask = filt.data_filt_DBSCAN(df_0_raw.loc[loc],feat,plot=True, epsCNR=True, just_noise = False).values[45:90,:]

vel = df_0_raw.ws.loc[df_0_raw.scan==scan_n].values
vel[mask] = np.nan

fig, ax = plt.subplots(figsize=(8,8))
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.1)
im = ax.contourf(r_0*np.cos(phi_0),r_0*np.sin(phi_0), 
       vel,cmap='jet')
ax.set_xlabel('$Easting\:m$', fontsize=f)
ax.set_ylabel('$Northing\:m$', fontsize=f)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(im, cax=cax, format=ticker.FuncFormatter(fm))
cbar.ax.tick_params(labelsize=f)
ax.tick_params(labelsize=14)
cbar.ax.set_ylabel("$V_{LOS}$ m/s", fontsize=f)
ax.text(0.05, 0.95, '(b)', transform=ax.transAxes, fontsize=24,
        verticalalignment='top')
fig.tight_layout()

###############################

# In[]
#############################################
#####################Numerical lidar, figure
from matplotlib.patches import Wedge
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-1300, 1000)
ax.set_ylim(-5500,-3500)
ax.set_zlim(0, 7)

plt.show()
r = 5000
r0 = r
phi0 = -30
dl = 75
for i in range(3):
    phi_0 = phi0   
    for j in range(4):        
        wedge = Wedge((-4000,-3000), r, phi_0, phi_0+5, width=600, fill = False, edgecolor = 'k', linewidth = 1,alpha=.3)
        ax.add_patch(wedge)
        art3d.pathpatch_2d_to_3d(wedge, z=0, zdir="z")
        phi_0+=5
    r = r-600

wedge = Wedge((-4000,-3000), r0, phi0, phi_0, width= r0-r, facecolor ='lightgrey', edgecolor = 'k', linewidth = 1, zorder=10)    

x_int = np.arange(-2000, 2000, 100)
y_int = np.arange(-6000, -3000, 100)
g = np.meshgrid(x_int, y_int)
coords = list(zip(*(c.flat for c in g)))

lim_x = np.array([np.cos(-20*np.pi/180)*3800,np.cos(-20*np.pi/180)*4400])-4000
lim_y = np.array([np.sin(-20*np.pi/180)*3800,np.sin(-20*np.pi/180)*4400])-3000

line_x = np.linspace(lim_x[0],lim_x[1],51)
line_y = np.linspace(lim_y[0],lim_y[1],51) 

h = 2*(line_x-line_x[int(len(line_x)/2)])/(lim_x[1]-lim_x[0])

delta_r = 35
r_F = h*(lim_x[1]-lim_x[0])/4#aux_1-aux_2
rp = dl/(2*np.sqrt(np.log(2)))

erf = sp.special.erf((r_F+.5*delta_r)/rp)-sp.special.erf((r_F-.5*delta_r)/rp)
w = (1/2/delta_r)*erf
#w = .75*(1-h**2)
w = 500*w

points = np.vstack([p for p in coords if wedge.contains_point(p, radius=0)])

ax.scatter(points[:,0],points[:,1],np.zeros(points[:,0].shape),'.',c='k',s=3)

ax.grid(False)
ax.set_frame_on(False)

ax.set_xticks([]) 
ax.set_yticks([]) 
ax.set_zticks([])

# Get rid of the spines
ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
# Get rid of the panes
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

for an in [-10,-15,-25,-30]:
    a_x_s = np.array([np.cos(an*np.pi/180)*2600,np.cos(an*np.pi/180)*5600])-4000
    a_y_s = np.array([np.sin(an*np.pi/180)*2600,np.sin(an*np.pi/180)*5600])-3000
    a_s = Arrow3D(a_x_s, a_y_s, [0, 0], mutation_scale=20, lw=2, arrowstyle="-|>",
                  linestyle = '--', color="grey",alpha=.5)
    ax.add_artist(a_s)

ax.plot(line_x,line_y,w,c='r',lw=2)

h = 0 
v = []
for k in range(0, len(line_x) - 1):
    x = [line_x[k], line_x[k+1], line_x[k+1], line_x[k]]
    y = [line_y[k], line_y[k+1], line_y[k+1], line_y[k]]
    z = [w[k], w[k+1], h, h]
    v.append(list(zip(x, y, z)))
poly3dCollection = Poly3DCollection(v,facecolor='red',alpha=.1)
ax.add_collection3d(poly3dCollection)

a_x = np.array([np.cos(-20*np.pi/180)*2600,np.cos(-20*np.pi/180)*5600])-4000
a_y = np.array([np.sin(-20*np.pi/180)*2600,np.sin(-20*np.pi/180)*5600])-3000
a = Arrow3D(a_x, a_y, [0, 0], mutation_scale=20, lw=3, arrowstyle="-|>", color="b")
ax.add_artist(a)

rangesx = np.array([np.cos(-20*np.pi/180)*(2900 + i*300) for i in range(9)])-4000
rangesy = np.array([np.sin(-20*np.pi/180)*(2900 + i*300) for i in range(9)])-3000
ax.scatter(rangesx,rangesy,np.zeros(len(rangesx)),'.',c='b',s=30)
fig.tight_layout()
ax.text2D(0.63, 0.8, '$w$', transform=ax.transAxes,fontsize=30)
ax.text2D(0.72, 0.4, '$Laser\:beam$', transform=ax.transAxes,fontsize=30)
ax.text2D(0.2, 0.2, '$Range\:gate$', transform=ax.transAxes,fontsize=30)
ax.text2D(0.22, 0.75, '$Beams\:to\:be\:averaged$', transform=ax.transAxes,fontsize=30)
#ax.text2D(0.15, 0.85, '(a)', transform=ax.transAxes,fontsize=24)
a_w = Arrow3D([rangesx[6],rangesx[4]], [rangesy[6],rangesy[4]], [8, w[25]],
              mutation_scale=20, lw=3, arrowstyle="wedge", color="k")
ax.add_artist(a_w)

a_b = Arrow3D([rangesx[8]+150,rangesx[7]], [rangesy[8]+150,rangesy[7]], [3, 0],
              mutation_scale=20, lw=2, arrowstyle="wedge", color="k")
ax.add_artist(a_b)

a_r = Arrow3D([-500,rangesx[4]], [-5200,rangesy[4]], [0, 0],
              mutation_scale=20, lw=2, arrowstyle="wedge", color="k")
ax.add_artist(a_r)

for an in [-10,-15,-25,-30]:
    a_x_s = np.array([np.cos(an*np.pi/180)*2600,np.cos(an*np.pi/180)*5600])-4000
    a_y_s = np.array([np.sin(an*np.pi/180)*2600,np.sin(an*np.pi/180)*5600])-3000

    a_f = Arrow3D([a_x[0],a_x_s[0]], [a_y[0],a_y_s[0]],[4, 0],
                  mutation_scale=20, lw=2, arrowstyle="wedge", color="k")
    ax.add_artist(a_f)
################    
######################################################################################