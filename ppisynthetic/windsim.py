# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 11:24:19 2018

@author: lalc
"""
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import os
import subprocess
import tkinter as tkint
import tkinter.filedialog
from sklearn.neighbors import KDTree
from sklearn.neighbors import KNeighborsRegressor
from matplotlib.tri import Triangulation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import ppiscanprocess.spectraconstruction as sc
import ppiscanprocess.windfieldrec as wr

# In[]
def num_lidar0(r,phi,U,V,x,y,d,rg=35):
    # Translate (x,y) field to lidar origin and transform to polar coordinates
    # Translate
    x_prime = x-d[0]
    y_prime = y-d[1]  
    if (len(x.shape)==1) & (len(y.shape)==1):
        x_prime, y_prime = np.meshgrid(x_prime,y_prime)
    if len(U.shape)>1 :
        U = U.flatten()
        V = V.flatten()
    # Transform to polar
    
    theta_prime = np.arctan2(y_prime.flatten(),x_prime.flatten()) 
    r_prime = np.sqrt(x_prime.flatten()**2+y_prime.flatten()**2)
    
    tree_lid = KDTree(np.c_[r.flatten(),phi.flatten()],metric='manhattan')
    
    ind = tree_lid.query(np.c_[r_prime,theta_prime],return_distance=False)
    
    V_LOS = np.zeros(len(r.flatten()))
    
    index_phi = (theta_prime>np.min(phi.flatten())-np.pi/180) & (theta_prime<np.max(phi.flatten())+np.pi/180)
    index_r = (r_prime>np.min(r.flatten())-35/2) & (r_prime<np.max(r.flatten())+35/2)
    
    for i in range(len(r.flatten())):
        index = ind.flatten() == i
        #print(i,ind.shape,index.shape,index_phi.shape,index_r.shape,U.shape)
        index = ((index) & (index_phi)) & (index_r)
        dr = r_prime[index]-r.flatten()[i]
        w = sp.stats.norm.pdf(2*dr/rg, loc=0, scale=1)
        #norm = sp.trapz(w,dr)
        #w = w/norm
        V_L = np.cos(theta_prime[index])*U[index]+np.sin(theta_prime[index])*V[index]
        V_LOS[i] = np.sum(-V_L*w)/np.sum(w)
        print(np.sum(np.isnan(V_L)),np.sum(np.isnan(w)),V_LOS[i])
        
    return np.reshape(V_LOS,r.shape)

# In[]

def num_lidar(r,phi,U,V,x,y,d,rg=35,deg=np.pi/180,n=10,m=10,kernel='epanechnikov',corr=True):
    # Translate (x,y) field to lidar origin and transform to polar coordinates
    # Translate
    
    phi_loc, r_loc = np.meshgrid(np.linspace(-deg,deg,m),np.linspace(-rg/2,rg/2,n))
    
    x_prime = x-d[0]
    y_prime = y-d[1]  
    if (len(x.shape)==1) & (len(y.shape)==1):
        x_prime, y_prime = np.meshgrid(x_prime,y_prime)
    if len(U.shape)>1 :
        U = U.flatten()
        V = V.flatten()
    # Transform to polar
    
    phi_prime = np.arctan2(y_prime.flatten(),x_prime.flatten()) 
    r_prime = np.sqrt(x_prime.flatten()**2+y_prime.flatten()**2)
      
    tree_lid = KDTree(np.c_[r.flatten(),phi.flatten()],metric='manhattan')
    
    ind = tree_lid.query(np.c_[r_prime,phi_prime],return_distance=False)
    
    V_LOS = np.zeros(len(r.flatten()))
    
    index_phi = (phi_prime>np.min(phi.flatten())-deg) & (phi_prime<np.max(phi.flatten())+deg)
    index_r = (r_prime>np.min(r.flatten())-35/2) & (r_prime<np.max(r.flatten())+35/2)
    n_index=[]
    
    for i in range(len(r.flatten())):        
        index = ind.flatten() == i
        index = ((index) & (index_phi)) & (index_r)
        V_L = np.cos(phi_prime[index])*U[index]+np.sin(phi_prime[index])*V[index]
        if kernel == 'gaussian':
            w = lambda h: sp.stats.norm.pdf(h, loc=0, scale=1)
        elif kernel == 'epanechnikov':
            w = lambda h: .75*(1-(h)**2)
        elif kernel == 'triangle':
            w = lambda h: 1-np.abs(h)
        else:
            w = lambda h: .75*(1-(h)**2)
        n_index.append(np.sum(index))
        # interpolation on beams
        
        if n_index[i]>10:    
            
            dr = -r_prime[index]+r.flatten()[i]
            dphi = -phi_prime[index]+phi.flatten()[i]
            V_L_g = sp.interpolate.griddata(np.c_[dr,dphi], V_L, (r_loc, phi_loc), method='nearest',fill_value = 0.0)
            if corr:
                V_L_TI = np.nanstd(V_L_g)/np.nanmean(V_L_g)
                V_L_TI = np.repeat([V_L_TI],n,axis=0)
                rg_corr = rg/(1+V_L_TI)
            else:
                rg_corr= rg            
            #h = 2*r_loc/rg_corr #!!!
            #h_phi = 
            #wg = w(h)
            #w_phi = 
            #simple average over azimuth
            #V_LOS[i] = np.nansum(np.nansum(V_L_g*wg,axis=0)/np.nansum(wg,axis=0))/m
            V_LOS[i] = np.nansum(np.nansum(V_L_g*wg,axis=0)/np.nansum(wg,axis=0))/m
            print(i,n_index[i],V_LOS[i],np.sum(np.isnan(V_L_g)),np.sum(np.isnan(wg)))
            
        if (n_index[i]<=10) & (n_index[i]>0):    
            dr = -r_prime[index]+r.flatten()[i]
            dphi = -phi_prime[index]+phi.flatten()[i]
            V_L_g = sp.interpolate.griddata(np.c_[dr,dphi], V_L, (r_loc, phi_loc), method='nearest')
            if corr:
                V_L_TI = np.nanstd(V_L_g)/np.nanmean(V_L_g)
                V_L_TI = np.repeat([V_L_TI],n,axis=0)
                rg_corr = rg/(1+V_L_TI)
            else:
                rg_corr= rg
            h = 2*r_loc/rg_corr
            wg = w(h)
            #simple average over azimuth
            V_LOS[i] = np.sum(np.sum(V_L_g*wg,axis=0)/np.sum(wg,axis=0))/m
        if n_index[i]==0:
            V_LOS[i] = np.nan
        
    return (np.reshape(-V_LOS,r.shape),np.array(n_index))

# In[]

root = tkint.Tk()
file_in_path = tkint.filedialog.askdirectory(parent=root,title='Choose a sim. Input dir')
root.destroy()

root = tkint.Tk()
file_out_path = tkint.filedialog.askdirectory(parent=root,title='Choose a sim Output dir')
root.destroy()


cwd = os.getcwd()
os.chdir(file_in_path)


# In[Lidar local coordinates]

r_s = np.linspace(150,7000,198)
r_v = np.linspace(150,7000,198)
phi_s = np.linspace(256,344,45)*np.pi/180
phi_v = np.linspace(196,284,45)*np.pi/180
r_s_g, phi_s_g = np.meshgrid(r_s,phi_s)
r_v_g, phi_v_g = np.meshgrid(r_v,phi_v)  

siro = np.array([6322832.3,0])
vara = np.array([6327082.4,0])
d = vara-siro


r_s_t,phi_s_t = wr.translationpolargrid((r_s_g, np.pi-phi_s_g),-d/2)
r_v_t,phi_v_t = wr.translationpolargrid((r_v_g, np.pi-phi_v_g),d/2)

x_max = np.max(np.r_[(r_s_t*np.cos(phi_s_t)).flatten(),(r_v_t*np.cos(phi_v_t)).flatten()])
x_min = np.min(np.r_[(r_s_t*np.cos(phi_s_t)).flatten(),(r_v_t*np.cos(phi_v_t)).flatten()])

y_max = np.max(np.r_[(r_s_t*np.sin(phi_s_t)).flatten(),(r_v_t*np.sin(phi_v_t)).flatten()])
y_min = np.min(np.r_[(r_s_t*np.sin(phi_s_t)).flatten(),(r_v_t*np.sin(phi_v_t)).flatten()])

L_x = x_max-x_min
L_y = y_max-y_min

# In[]

#Sepctral tensor parameters
os.chdir(file_in_path)
ae = .05
L = 2000
G = 3.1
N_x = 1024
N_y = 1024


input_file = 'sim.inp.txt'

file = open(input_file,'w') 
file.write('2\n') 
file.write('2\n') 
file.write('1\n')
file.write('2\n') 
file.write('3\n')  
file.write(str(N_x)+'\n')
file.write(str(N_y)+'\n')
file.write(str(L_x)+'\n')
file.write(str(L_y)+'\n')
file.write('basic\n')
file.write(str(ae)+'\n')
file.write(str(L)+'\n')
file.write(str(G)+'\n')
file.write('-30\n')
file.write('simu\n')
file.write('simv\n')
file.close() 

arg = 'windsimu'+' '+input_file
p=subprocess.run(arg)

u = np.reshape(np.fromfile("simu", dtype=np.float32),(N_x,N_y))
v = np.reshape(np.fromfile("simv", dtype=np.float32),(N_x,N_y))

os.chdir(cwd)

# In[]
U_mean = 15
V_mean = 0
Dir = 270*np.pi/180

U = U_mean + u
V = V_mean + v

x = np.linspace(x_min,x_max,N_x)
y = np.linspace(y_min,y_max,N_y)

# Rotation

gamma = -(2*np.pi-Dir) # Rotation
# Components in matrix of coefficients
S11 = np.cos(gamma)
S12 = np.sin(gamma)
T = np.array([[S11,S12], [-S12,S11]])
vel = np.array(np.c_[U.flatten(),V.flatten()]).T
vel = np.dot(T,vel)
X = np.array(np.c_[x,y]).T
X = np.dot(T,X)
U_prime = np.reshape(vel[0,:],(N_x,N_y))
V_prime = np.reshape(vel[1,:],(N_x,N_y))

# In[]

tree,tri, wva, neighva, indexva, wsi, neighsi, indexsi = wr.grid_over2((r_v_g, np.pi-phi_v_g),(r_s_g, np.pi-phi_s_g),-d)
r_s_r = np.linspace(150,7000,198*2)
r_v_r = np.linspace(150,7000,198*2)
phi_s_r = np.linspace(256,344,45*2)*np.pi/180
phi_v_r = np.linspace(196,284,45*2)*np.pi/180
r_s_g_r, phi_s_g_r = np.meshgrid(r_s_r,phi_s_r)
r_v_g_r, phi_v_g_r = np.meshgrid(r_v_r,phi_v_r) 
tree_r,tri_r, wva_r, neighva_r, indexva_r, wsi_r, neighsi_r, indexsi_r = wr.grid_over2((r_v_g_r, np.pi-phi_v_g_r),(r_s_g_r, np.pi-phi_s_g_r),-d)

# In[]

#vloss0 = num_lidar0(r_s_g, np.pi-phi_s_g,U_prime,V_prime,x,y,d/2)
#vlosv0 = num_lidar0(r_v_g, np.pi-phi_v_g,U_prime,V_prime,x,y,-d/2)

vloss,_ = num_lidar(r_s_g, np.pi-phi_s_g,U_prime,V_prime,x,y,d/2,corr=False)
vlosv,_ = num_lidar(r_v_g, np.pi-phi_v_g,U_prime,V_prime,x,y,-d/2,corr=False)

#vloss1,_ = num_lidar(r_s_g, np.pi-phi_s_g,U_prime,V_prime,x,y,d/2)
#vlosv1,_ = num_lidar(r_v_g, np.pi-phi_v_g,U_prime,V_prime,x,y,-d/2)

################## Refinement


####################Refinement of lidar data

#r_min = np.max(r_s_g[np.isnan(vloss0)])
#ind_r = int((r_s == r_min).nonzero()[0]+1)
#
#vloss_r0 = sp.interpolate.RectBivariateSpline(phi_s, r_s[ind_r:], vloss0[:,int(ind_r):])(phi_s_r, r_s_r)
#
#r_min = np.max(r_v_g[np.isnan(vlosv0)])
#ind_r = int((r_v == r_min).nonzero()[0]+1)
#
#vlosv_r0 = sp.interpolate.RectBivariateSpline(phi_v, r_v[ind_r:], vlosv0[:,int(ind_r):])(phi_v_r, r_v_r)

r_min = np.max(r_s_g[np.isnan(vloss)])
ind_r = int((r_s == r_min).nonzero()[0]+1)

vloss_r = sp.interpolate.RectBivariateSpline(phi_s, r_s[ind_r:], vloss[:,int(ind_r):])(phi_s_r, r_s_r)

r_min = np.max(r_v_g[np.isnan(vlosv)])
ind_r = int((r_v == r_min).nonzero()[0]+1)

vlosv_r = sp.interpolate.RectBivariateSpline(phi_v, r_v[ind_r:], vlosv[:,int(ind_r):])(phi_v_r, r_v_r)

#r_min = np.max(r_s_g[np.isnan(vloss1)])
#ind_r = int((r_s == r_min).nonzero()[0]+1)
#
#vloss_r1 = sp.interpolate.RectBivariateSpline(phi_s, r_s[ind_r:], vloss1[:,int(ind_r):])(phi_s_r, r_s_r)
#
#r_min = np.max(r_v_g[np.isnan(vlosv)])
#ind_r = int((r_v == r_min).nonzero()[0]+1)
#
#vlosv_r1 = sp.interpolate.RectBivariateSpline(phi_v, r_v[ind_r:], vlosv1[:,int(ind_r):])(phi_v_r, r_v_r)
################ Interpolation of raw data

V_prime_t = sp.interpolate.RectBivariateSpline(y,x, V_prime)(tri.y,tri.x,grid=False)

V_prime_tr = sp.interpolate.RectBivariateSpline(y,x, V_prime)(tri_r.y,tri_r.x,grid=False)

U_prime_t = sp.interpolate.RectBivariateSpline(y,x, U_prime)(tri.y,tri.x,grid=False)

U_prime_tr = sp.interpolate.RectBivariateSpline(y,x, U_prime)(tri_r.y,tri_r.x,grid=False)

################################## Reconstruction
df_s = pd.DataFrame(vloss)
df_v = pd.DataFrame(vlosv)

Lidar_s = (df_s,2*np.pi-phi_s_g,wsi,neighsi,indexsi) 
Lidar_v = (df_v,2*np.pi-phi_v_g,wva,neighva,indexva)

U_rec, V_rec= wr.wind_field_rec(Lidar_v, Lidar_s, tree, tri, d)


#df_s = pd.DataFrame(vloss0)
#df_v = pd.DataFrame(vlosv0)
#
#Lidar_s = (df_s,2*np.pi-phi_s_g,wsi,neighsi,indexsi) 
#Lidar_v = (df_v,2*np.pi-phi_v_g,wva,neighva,indexva)
#
#U_rec0, V_rec0= wr.wind_field_rec(Lidar_v, Lidar_s, tree, tri, d)

#df_s = pd.DataFrame(vloss1)
#df_v = pd.DataFrame(vlosv1)
#
#Lidar_s = (df_s,2*np.pi-phi_s_g,wsi,neighsi,indexsi) 
#Lidar_v = (df_v,2*np.pi-phi_v_g,wva,neighva,indexva)
#
#U_rec1, V_rec1= wr.wind_field_rec(Lidar_v, Lidar_s, tree, tri, d)


## Refined , phi_s_g_r

df_s = pd.DataFrame(vloss_r)
df_v = pd.DataFrame(vlosv_r)

Lidar_s = (df_s,2*np.pi-phi_s_g_r,wsi_r,neighsi_r,indexsi_r) 
Lidar_v = (df_v,2*np.pi-phi_v_g_r,wva_r,neighva_r,indexva_r)

U_rec_r, V_rec_r= wr.wind_field_rec(Lidar_v, Lidar_s, tree_r, tri_r, d)

#df_s = pd.DataFrame(vloss_r0)
#df_v = pd.DataFrame(vlosv_r0)
#
#Lidar_s = (df_s,2*np.pi-phi_s_g_r,wsi_r,neighsi_r,indexsi_r) 
#Lidar_v = (df_v,2*np.pi-phi_v_g_r,wva_r,neighva_r,indexva_r)
#
#U_rec_r0, V_rec_r0= wr.wind_field_rec(Lidar_v, Lidar_s, tree_r, tri_r, d)
#
#df_s = pd.DataFrame(vloss_r1)
#df_v = pd.DataFrame(vlosv_r1)
#
#Lidar_s = (df_s,2*np.pi-phi_s_g_r,wsi_r,neighsi_r,indexsi_r) 
#Lidar_v = (df_v,2*np.pi-phi_v_g_r,wva_r,neighva_r,indexva_r)
#
#U_rec_r1, V_rec_r1= wr.wind_field_rec(Lidar_v, Lidar_s, tree_r, tri_r, d)

###############
#Early interpolation
indv = ~np.isnan(vlosv.flatten())
inds = ~np.isnan(vloss.flatten())
xv = (r_v_t*np.cos(phi_v_t)).flatten()[indv]
yv = (r_v_t*np.sin(phi_v_t)).flatten()[indv]
#
xs = (r_s_t*np.cos(phi_s_t)).flatten()[inds]
ys = (r_s_t*np.sin(phi_s_t)).flatten()[inds]
#
#vlosv_int = sp.interpolate.griddata(np.c_[xv,yv], vlosv.flatten()[indv], (tri_r.x, tri_r.y), method='cubic')
#vloss_int = sp.interpolate.griddata(np.c_[xs,ys], vloss.flatten()[inds], (tri_r.x, tri_r.y), method='cubic')
#
#
#ind = np.isnan(vlosv_int)
#vlosv_int[ind] = sp.interpolate.griddata(np.c_[xv,yv], vlosv.flatten()[indv], (tri_r.x, tri_r.y), method='nearest')[ind]
#ind = np.isnan(vloss_int)
#vloss_int[ind] = sp.interpolate.griddata(np.c_[xs,ys], vloss.flatten()[inds], (tri_r.x, tri_r.y), method='nearest')[ind]

#fig, ax = plt.subplots()
#ax.set_aspect('equal')
## Enforce the margins, and enlarge them to give room for the vectors.
#ax.use_sticky_edges = False
#ax.margins(0.07)
#ax.triplot(tri_r, color='grey',lw=.5)
#imUr = ax.tricontourf(tri_r,vlosv_int,300,cmap='jet')
#fig.colorbar(imUr)
#
#fig, ax = plt.subplots()
#ax.set_aspect('equal')
## Enforce the margins, and enlarge them to give room for the vectors.
#ax.use_sticky_edges = False
#ax.margins(0.07)
#ax.triplot(tri_r, color='grey',lw=.5)
#imUr = ax.tricontourf(tri_r,vloss_int,300,cmap='jet')
#fig.colorbar(imUr)
#
#r_tri = np.sqrt(tri_r.x**2 + tri_r.y**2)
#phi_tri = np.arctan2(tri_r.y,tri_r.x)
#
#r_tri_v, phi_tri_v = wr.translationpolargrid((r_tri, phi_tri),-d/2)
#r_tri_s, phi_tri_s = wr.translationpolargrid((r_tri, phi_tri), d/2)
#
T_i = lambda a_i,b_i,v_los: np.linalg.solve(np.array([[np.cos(a_i),np.sin(a_i)],[np.cos(b_i),np.sin(b_i)]]),v_los)
#
#vel = [T_i(a,b,np.array([v_lv,v_ls])) for a,b,v_lv,v_ls in zip(phi_tri_v,phi_tri_s,-vlosv_int,-vloss_int)]
#
#
#U_early_r= np.array(vel)[:,0]
#V_early_r= np.array(vel)[:,1]

###### Structured grid

grd = np.meshgrid(x,y)

mask_s = np.reshape(tri_r.get_trifinder()(grd[0].flatten(),grd[1].flatten()),grd[0].shape) == -1

vlosv_int_sq = sp.interpolate.griddata(np.c_[xv,yv], vlosv.flatten()[indv], (grd[0].flatten(), grd[1].flatten()), method='cubic')
vloss_int_sq = sp.interpolate.griddata(np.c_[xs,ys], vloss.flatten()[inds], (grd[0].flatten(), grd[1].flatten()), method='cubic')

ind_sq = np.isnan(vlosv_int_sq)
vlosv_int_sq[ind_sq] = sp.interpolate.griddata(np.c_[xv,yv], vlosv.flatten()[indv], (grd[0].flatten(), grd[1].flatten()), method='nearest')[ind_sq]

ind_sq = np.isnan(vloss_int_sq)
vloss_int_sq[ind_sq] = sp.interpolate.griddata(np.c_[xs,ys], vloss.flatten()[inds], (grd[0].flatten(), grd[1].flatten()), method='nearest')[ind_sq]

vloss_int_sq = np.reshape(vloss_int_sq,grd[0].shape)
vlosv_int_sq = np.reshape(vlosv_int_sq,grd[0].shape)

vloss_int_sq[mask_s] = 0.0
vlosv_int_sq[mask_s] = 0.0

#plt.figure()
#plt.contourf(grd[0],grd[1],vloss_int_sq,cmap='jet')
#
#plt.figure()
#plt.contourf(grd[0],grd[1],vlosv_int_sq,cmap='jet')

r_tri_s = np.sqrt(grd[0]**2 + grd[1]**2)
phi_tri_s = np.arctan2(grd[1],grd[0])



r_tri_v_s, phi_tri_v_s = wr.translationpolargrid((r_tri_s, phi_tri_s),-d/2)
r_tri_s_s, phi_tri_s_s = wr.translationpolargrid((r_tri_s, phi_tri_s), d/2)



vel_s = np.array([T_i(a,b,np.array([v_lv,v_ls])) for a,b,v_lv,v_ls in zip(phi_tri_v_s.flatten(),phi_tri_s_s.flatten(),-vlosv_int_sq.flatten(),-vloss_int_sq.flatten())])

U_early_s = np.reshape(vel_s[:,0],grd[0].shape)
V_early_s = np.reshape(vel_s[:,1],grd[0].shape)

U_prime[mask_s] = np.nan
V_prime[mask_s] = np.nan


############################
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
im = ax.contourf(x, y,U_prime,300,cmap='jet')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(im, cax=cax, format=ticker.FuncFormatter(fm))
cbar.ax.tick_params(labelsize=f)
ax.tick_params(labelsize=f)
cbar.ax.set_ylabel("$V_x$ [m/s]", fontsize=f)
ax.set_ylabel(r'West-East [m]', fontsize=f, weight='bold')
ax.set_xlabel(r'North-South [m]', fontsize=f, weight='bold')
ax.text(0.05, 0.95, '(a)', transform=ax.transAxes, fontsize=f,verticalalignment='top')
ax.set_xlim(-3500,3500)
ax.set_ylim(-7000,-200)
#ax.triplot(tri, color='grey',lw=.5)
##########################
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
im = ax.contourf(x, y,V_prime,300,cmap='jet')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(im, cax=cax, format=ticker.FuncFormatter(fm))
cbar.ax.tick_params(labelsize=f)
ax.tick_params(labelsize=f)
cbar.ax.set_ylabel("$V_y$ [m/s]", fontsize=f)
ax.set_ylabel(r'West-East [m]', fontsize=f, weight='bold')
ax.set_xlabel(r'North-South [m]', fontsize=f, weight='bold')
ax.text(0.05, 0.95, '(b)', transform=ax.transAxes, fontsize=f,verticalalignment='top')
ax.set_xlim(-3500,3500)
ax.set_ylim(-7000,-200)
#ax.triplot(tri, color='grey',lw=.5)

######################
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
im = ax.contourf(grd[0],grd[1],V_early_s,np.linspace(np.nanmin(V_prime),np.nanmax(V_prime),300),cmap='jet')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(im, cax=cax, format=ticker.FuncFormatter(fm))
cbar.ax.tick_params(labelsize=f)
ax.tick_params(labelsize=f)
cbar.ax.set_ylabel("$V_y$ [m/s]", fontsize=f)
ax.set_ylabel(r'West-East [m]', fontsize=f, weight='bold')
ax.set_xlabel(r'North-South [m]', fontsize=f, weight='bold')
ax.text(0.05, 0.95, '(b)', transform=ax.transAxes, fontsize=f,verticalalignment='top')
ax.set_xlim(-3500,3500)
ax.set_ylim(-7000,-200)
#ax.triplot(tri, color='grey',lw=.5)
##########################
######################
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
im = ax.contourf(grd[0],grd[1],U_early_s,np.linspace(np.nanmin(U_prime),np.nanmax(U_prime),300),cmap='jet')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(im, cax=cax, format=ticker.FuncFormatter(fm))
cbar.ax.tick_params(labelsize=f)
ax.tick_params(labelsize=f)
cbar.ax.set_ylabel("$V_x$ [m/s]", fontsize=f)
ax.set_ylabel(r'West-East [m]', fontsize=f, weight='bold')
ax.set_xlabel(r'North-South [m]', fontsize=f, weight='bold')
ax.text(0.05, 0.95, '(a)', transform=ax.transAxes, fontsize=f,verticalalignment='top')
ax.set_xlim(-3500,3500)
ax.set_ylim(-7000,-200)
#ax.triplot(tri, color='grey',lw=.5)
##########################
############################
#
######################
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
im = ax.tricontourf(tri,U_rec,np.linspace(np.nanmin(U_prime),np.nanmax(U_prime),300),cmap='jet')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(im, cax=cax, format=ticker.FuncFormatter(fm))
cbar.ax.tick_params(labelsize=f)
ax.tick_params(labelsize=f)
cbar.ax.set_ylabel("$V_x$ [m/s]", fontsize=f)
ax.set_ylabel(r'West-East [m]', fontsize=f, weight='bold')
ax.set_xlabel(r'North-South [m]', fontsize=f, weight='bold')
ax.text(0.05, 0.95, '(a)', transform=ax.transAxes, fontsize=f,verticalalignment='top')
ax.set_xlim(-3500,3500)
ax.set_ylim(-7000,-200)
#ax.triplot(tri, color='grey',lw=.5)
##########################
######################
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
im = ax.tricontourf(tri,V_rec,np.linspace(np.nanmin(V_prime),np.nanmax(V_prime),300),cmap='jet')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(im, cax=cax, format=ticker.FuncFormatter(fm))
cbar.ax.tick_params(labelsize=f)
ax.tick_params(labelsize=f)
cbar.ax.set_ylabel("$V_y$ [m/s]", fontsize=f)
ax.set_ylabel(r'West-East [m]', fontsize=f, weight='bold')
ax.set_xlabel(r'North-South [m]', fontsize=f, weight='bold')
ax.text(0.05, 0.95, '(b)', transform=ax.transAxes, fontsize=f,verticalalignment='top')
ax.set_xlim(-3500,3500)
ax.set_ylim(-7000,-200)
#ax.triplot(tri, color='grey',lw=.5)
##########################
############################

######################
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
im = ax.tricontourf(tri_r,V_rec_r,np.linspace(np.nanmin(V_prime),np.nanmax(V_prime),300),cmap='jet')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(im, cax=cax, format=ticker.FuncFormatter(fm))
cbar.ax.tick_params(labelsize=f)
ax.tick_params(labelsize=f)
cbar.ax.set_ylabel("$V_x$ [m/s]", fontsize=f)
ax.set_ylabel(r'West-East [m]', fontsize=f, weight='bold')
ax.set_xlabel(r'North-South [m]', fontsize=f, weight='bold')
ax.text(0.05, 0.95, '(b)', transform=ax.transAxes, fontsize=f,verticalalignment='top')
ax.set_xlim(-3500,3500)
ax.set_ylim(-7000,-200)
ax.triplot(tri, color='grey',lw=.5)
##########################
######################
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
im = ax.tricontourf(tri_r,U_rec_r,np.linspace(np.nanmin(U_prime),np.nanmax(U_prime),300),cmap='jet')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(im, cax=cax, format=ticker.FuncFormatter(fm))
cbar.ax.tick_params(labelsize=f)
ax.tick_params(labelsize=f)
cbar.ax.set_ylabel("$V_x$ [m/s]", fontsize=f)
ax.set_ylabel(r'West-East [m]', fontsize=f, weight='bold')
ax.set_xlabel(r'North-South [m]', fontsize=f, weight='bold')
ax.text(0.05, 0.95, '(a)', transform=ax.transAxes, fontsize=f,verticalalignment='top')
ax.set_xlim(-3500,3500)
ax.set_ylim(-7000,-200)
ax.triplot(tri, color='grey',lw=.5)
##########################
############################
#################

#fig, ax = plt.subplots()
#ax.set_aspect('equal')
## Enforce the margins, and enlarge them to give room for the vectors.
#ax.use_sticky_edges = False
#ax.margins(0.07)
#ax.triplot(tri, color='grey',lw=.5)
#imUr = ax.tricontourf(tri_r,V_rec_r,np.linspace(np.min(V_prime),np.max(V_prime),300),cmap='jet')
#fig.colorbar(imUr)
#
#
#fig, ax = plt.subplots()
#ax.set_aspect('equal')
## Enforce the margins, and enlarge them to give room for the vectors.
#ax.use_sticky_edges = False
#ax.margins(0.07)
#ax.triplot(tri, color='black',lw=.5)
#imU = ax.contourf(r_s_g*np.cos(np.pi-phi_s_g)+d[0]/2,r_s_g*np.sin(np.pi-phi_s_g),vloss,100,cmap='jet')
#fig.colorbar(imU)
#
#fig, ax = plt.subplots()
#ax.set_aspect('equal')
## Enforce the margins, and enlarge them to give room for the vectors.
#ax.use_sticky_edges = False
#ax.margins(0.07)
##ax.triplot(tri_r, color='black',lw=.5)
#imU = ax.contourf(r_s_g_r*np.cos(np.pi-phi_s_g_r)+d[0]/2,r_s_g_r*np.sin(np.pi-phi_s_g_r),vloss_r,100,cmap='jet')
#fig.colorbar(imU)
#
## End of Refinement
#
#fig, ax = plt.subplots()
#ax.set_aspect('equal')
## Enforce the margins, and enlarge them to give room for the vectors.
#ax.use_sticky_edges = False
#ax.margins(0.07)
##ax.triplot(tri_r, color='black',lw=.5)
#imUr = ax.tricontourf(tri_r,V_prime_t-V_rec_r0,300,cmap='jet')
#fig.colorbar(imUr)
#
#fig, ax = plt.subplots()
#ax.set_aspect('equal')
## Enforce the margins, and enlarge them to give room for the vectors.
#ax.use_sticky_edges = False
#ax.margins(0.07)
##ax.triplot(tri_r, color='black',lw=.5)
#imUr = ax.tricontourf(tri_r,V_prime_t,300,cmap='jet')
#fig.colorbar(imUr)
#
#fig, ax = plt.subplots()
#ax.set_aspect('equal')
## Enforce the margins, and enlarge them to give room for the vectors.
#ax.use_sticky_edges = False
#ax.margins(0.07)
#ax.triplot(tri, color='grey',lw=.5)
#ax.contourf(x, y,U_prime,300,cmap='jet')
#
#fig, ax = plt.subplots()
#ax.set_aspect('equal')
## Enforce the margins, and enlarge them to give room for the vectors.
#ax.use_sticky_edges = False
#ax.margins(0.07)
#ax.triplot(tri, color='grey',lw=.5)
#ax.contourf(x, y,V_prime,300,cmap='jet')
#
# In[]
#fig, ax = plt.subplots()
#ax.set_aspect('equal')
## Enforce the margins, and enlarge them to give room for the vectors.
#ax.use_sticky_edges = False
#ax.margins(0.07)
#ax.triplot(tri, color='black',lw=.5)
#imU = ax.contourf(x, y,U_prime,300,cmap='jet')
#fig.colorbar(imU)
#
#fig, ax = plt.subplots()
#ax.set_aspect('equal')
## Enforce the margins, and enlarge them to give room for the vectors.
#ax.use_sticky_edges = False
#ax.margins(0.07)
#ax.triplot(tri, color='black',lw=.5)
#imUr = ax.tricontourf(tri,U_rec,np.linspace(np.min(U_prime),np.max(U_prime),300),cmap='jet')
#fig.colorbar(imUr)
#
#fig, ax = plt.subplots()
#ax.set_aspect('equal')
## Enforce the margins, and enlarge them to give room for the vectors.
#ax.use_sticky_edges = False
#ax.margins(0.07)
#ax.triplot(tri, color='black',lw=.5)
#imU = ax.contourf(x, y,V_prime,300,cmap='jet')
#fig.colorbar(imU)
#
#fig, ax = plt.subplots()
#ax.set_aspect('equal')
## Enforce the margins, and enlarge them to give room for the vectors.
#ax.use_sticky_edges = False
#ax.margins(0.07)
#ax.triplot(tri, color='black',lw=.5)
#imUr = ax.tricontourf(tri,V_rec,np.linspace(np.min(V_prime),np.max(V_prime),300),cmap='jet')
#fig.colorbar(imUr)
#
#fig, ax = plt.subplots()
#ax.set_aspect('equal')
## Enforce the margins, and enlarge them to give room for the vectors.
#ax.use_sticky_edges = False
#ax.margins(0.07)
#ax.triplot(tri, color='black',lw=.5)
#imU = ax.contourf(r_s_g*np.cos(np.pi-phi_s_g)+d[0]/2,r_s_g*np.sin(np.pi-phi_s_g),vloss,100,cmap='jet')
#fig.colorbar(imU)
#
#fig, ax = plt.subplots()
#ax.set_aspect('equal')
## Enforce the margins, and enlarge them to give room for the vectors.
#ax.use_sticky_edges = False
#ax.margins(0.07)
#ax.triplot(tri, color='black',lw=.5)
#imU = ax.contourf(r_v_g*np.cos(np.pi-phi_v_g)-d[0]/2,r_v_g*np.sin(np.pi-phi_v_g),vlosv,100,cmap='jet')
#fig.colorbar(imU)

# In[Theoretical]
U_th = U_prime
V_th = V_prime

U_th[mask_s] = np.nan
V_th[mask_s] = np.nan

U_mean = np.nanmean(U_th.flatten())
V_mean = np.nanmean(V_th.flatten())

# Wind direction
gamma = np.arctan2(V_mean,U_mean)
# Components in matrix of coefficients
S11 = np.cos(gamma)
S12 = np.sin(gamma)
T = np.array([[S11,S12], [-S12,S11]])
vel = np.array(np.c_[U_th.flatten(),V_th.flatten()]).T
vel = np.dot(T,vel)
X = np.array(np.c_[x,y]).T
X = np.dot(T,X)

U_t = np.reshape(vel[0,:],U_prime.shape).T
V_t = np.reshape(vel[1,:],V_prime.shape).T

U_mean = np.nanmean(U_t.flatten())
V_mean = np.nanmean(V_t.flatten())

U_t = U_t-U_mean
V_t = V_t-V_mean

U_t[np.isnan(U_t)] = 0.0
V_t[np.isnan(V_t)] = 0.0


grid = np.meshgrid(X[0,:],X[1,:])

dx = np.min(np.abs(np.diff(X[0,:])))
dy = np.min(np.abs(np.diff(X[1,:])))

n = grid[0].shape[0]
m = grid[1].shape[0]   
# Spectra

fftU = np.fft.fft2(U_t)
fftV = np.fft.fft2(V_t)

fftU  = np.fft.fftshift(fftU)
fftV  = np.fft.fftshift(fftV) 

Suu = 2*(np.abs(fftU)**2)*dx*dy/(n*m)
Svv = 2*(np.abs(fftV)**2)*dx*dy/(n*m)
kx = 1/(2*dx)
ky = 1/(2*dy)   
k1th = np.fft.fftshift((np.fft.fftfreq(n, d=dx)))
k2th = np.fft.fftshift((np.fft.fftfreq(m, d=dy)))

Su_u_t = sp.integrate.simps(Suu,k2th,axis=1)[k1th>0]
Sv_v_t = sp.integrate.simps(Svv,k2th,axis=1)[k1th>0]

kth = k1th[k1th>0]
F_th = .5*(Su_u_t + Sv_v_t)

# In[Observed]
U_s = U_early_s
V_s = V_early_s

U_s[mask_s] = np.nan
V_s[mask_s] = np.nan
 
U_mean = np.nanmean(U_s.flatten())
V_mean = np.nanmean(V_s.flatten())
# Wind direction
gamma = np.arctan2(V_mean,U_mean)
# Components in matrix of coefficients
S11 = np.cos(gamma)
S12 = np.sin(gamma)
T = np.array([[S11,S12], [-S12,S11]])
vel = np.array(np.c_[U_s.flatten(),V_s.flatten()]).T
vel = np.dot(T,vel)
X = np.array(np.c_[x,y]).T
X = np.dot(T,X)

U_t = np.reshape(vel[0,:],U_s.shape).T
V_t = np.reshape(vel[1,:],V_s.shape).T

U_mean = np.nanmean(U_t.flatten())
V_mean = np.nanmean(V_t.flatten())

U_t = U_t-U_mean
V_t = V_t-V_mean

U_t[np.isnan(U_t)] = 0.0
V_t[np.isnan(V_t)] = 0.0


grid = np.meshgrid(X[0,:],X[1,:])

dx = np.min(np.abs(np.diff(X[0,:])))
dy = np.min(np.abs(np.diff(X[1,:])))

n = grid[0].shape[0]
m = grid[1].shape[0]   
# Spectra

fftU = np.fft.fft2(U_t)
fftV = np.fft.fft2(V_t)

fftU  = np.fft.fftshift(fftU)
fftV  = np.fft.fftshift(fftV) 

Suu = 2*(np.abs(fftU)**2)*dx*dy/(n*m)
Svv = 2*(np.abs(fftV)**2)*dx*dy/(n*m)
kx = 1/(2*dx)
ky = 1/(2*dy)   
k1th = np.fft.fftshift((np.fft.fftfreq(n, d=dx)))
k2th = np.fft.fftshift((np.fft.fftfreq(m, d=dy)))

Su_u_t_s = sp.integrate.simps(Suu,k2th,axis=1)[k1th>0]
Sv_v_t_s = sp.integrate.simps(Svv,k2th,axis=1)[k1th>0]

kth_s = k1th[k1th>0]
F_th_s = .5*(Su_u_t_s + Sv_v_t_s)


# In[Spectra comparison]
U_th, V_th = u, v
U_th[mask_s] = np.nan
V_th[mask_s] = np.nan
k1_th,k2_th,Su_u_t,Sv_v_t,_,F_th,_,_,_ = spatial_spec_sq(x,y,U_th,V_th,transform = False,shrink = False)
# Non refined grid

#Sureal,Svreal,Suvreal,k1real,k2real=sc.spatial_autocorr_fft(tri,U_prime_t,V_prime_t,transform = True,N_grid=1024,interp='cubic')
#Su_ureal = sp.integrate.simps(Sureal,k2real,axis=1)[k1real>0]
#Sv_vreal = sp.integrate.simps(Svreal,k2real,axis=1)[k1real>0]
#kreal = k1real[k1real>0]
#F_real = .5*(Su_ureal + Sv_vreal)
 
Suobs,Svobs,Suvobs,k1obs,k2obs=sc.spatial_autocorr_fft(tri,U_rec,V_rec,transform = True,N_grid=1024,interp='cubic')
Su_uobs = sp.integrate.simps(Suobs,k2obs,axis=1)[k1obs>0]
Sv_vobs = sp.integrate.simps(Svobs,k2obs,axis=1)[k1obs>0]
kobs = k1obs[k1obs>0]
F_obs = .5*(Su_uobs + Sv_vobs)

# Refined grid

#Surealr,Svrealr,Suvrealr,k1realr,k2realr=sc.spatial_autocorr_fft(tri_r,U_prime_tr,V_prime_tr,transform = True,N_grid=1024,interp='cubic')
#Su_urealr = sp.integrate.simps(Surealr,k2realr,axis=1)[k1realr>0]
#Sv_vrealr = sp.integrate.simps(Svrealr,k2realr,axis=1)[k1realr>0]
#krealr = k1realr[k1realr>0]
#F_realr = .5*(Su_urealr + Sv_vrealr)
 
Suobsr,Svobsr,Suvobsr,k1obsr,k2obsr=sc.spatial_autocorr_fft(tri_r,U_rec_r,V_rec_r,transform = True,N_grid=1024,interp='cubic')
Su_uobsr = sp.integrate.simps(Suobsr,k2obsr,axis=1)[k1obsr>0]
Sv_vobsr = sp.integrate.simps(Svobsr,k2obsr,axis=1)[k1obsr>0]
kobsr = k1obsr[k1obsr>0]
F_obsr = .5*(Su_ur + Sv_vr)

plt.figure()
plt.xscale('log')
plt.yscale('log')
plt.plot(kth,kth*Su_u_t,'-', lw = 2, color = 'k', label='theoretical') 
plt.plot(kth_s,kth_s*Su_u_t_s,'-o', label='recostructed, early interp') 
#plt.plot(kreal,kreal*Su_ureal,'-o', label='theo tri-interpolated') 
#plt.plot(krealr,krealr*Su_urealr,'-o', label='theo tri-interpolated, refined') 
plt.plot(kobs,kobs*Su_uobs,'-o', label='reconstructed') 
#plt.plot(kobsr,kobsr*Su_uobsr,'-o', label='reconstructed, refined') 
plt.xlabel('$k$')
plt.ylabel('$S_{uu}$')
#plt.ylim(5*10**-4,3*10**0)
#plt.xlim(np.min(kth), np.max(kth))
plt.legend()

plt.figure()
plt.xscale('log')
plt.yscale('log')
plt.plot(kth,kth*Sv_v_t,'-', lw = 2, color = 'k', label='theoretical') 
plt.plot(kth_s,kth_s*Sv_v_t_s,'-o', label='recostructed, early interp') 
#plt.plot(kreal,kreal*Sv_vreal,'-o', label='theo tri-interpolated') 
#plt.plot(krealr,krealr*Sv_vrealr,'-o', label='theo tri-interpolated, refined') 
plt.plot(kobs,kobs*Sv_v,'-o', label='reconstructed') 
plt.plot(kobsr,kobsr*Sv_vr,'-o', label='reconstructed, refined') 
plt.xlabel('k')
plt.ylabel('$S_{uu}$')
#plt.ylim(5*10**-4,3*10**0)
#plt.xlim(np.min(kth), np.max(kth))
plt.legend()



plt.figure()
plt.xscale('log')
plt.yscale('log')
#plt.plot(kth,kth*Su_u_t,'-', lw = 2, color = 'k', label='theoretical') 
plt.plot(kth_s,Su_u_t_s/Su_u_t,'-', label='recostructed, early interpolation') 
#plt.plot(kreal[1:],Su_ureal[1:]/Su_u_t,'-', label='theo tri-interpolated') 
#plt.plot(krealr[1:],Su_urealr[1:]/Su_u_t,'-', label='theo tri-interpolated, refined') 
plt.plot(kobs,Su_uobs/Su_u_t,'-', label='reconstructed') 
#plt.plot(kobsr,Su_ur/Su_u_t,'-', label='reconstructed, refined')

#plt.plot(kobs,10/(10+(1000*kobs)**2),'-', label='reconstructed, refined')

plt.xlabel('$k$')
plt.ylabel('$S_{uu}^{rec}/S_{uu}^{theo}$')
#plt.ylim(5*10**-4,3*10**0)
#plt.xlim(np.min(kth), np.max(kth))
plt.legend()


plt.figure()
plt.xscale('log')
plt.yscale('log')
plt.plot(kth_s,Su_u_t_s/Su_u_t,'-', label='recostructed, early interp') 
plt.plot(kth_s,Sv_v_t_s/Sv_v_t,'-', label='recostructed, early interp') 
plt.legend()
#Suobs,Svobs,Suvobs,k1obs,k2obs=sc.spatial_autocorr_fft(tri_r,U_rec_r0,V_rec_r0,transform = True,N_grid=1024,interp='cubic')
#Su_ur0 = sp.integrate.simps(Suobs,k2obs,axis=1)[k1obs>0]
#Sv_vr0 = sp.integrate.simps(Svobs,k2obs,axis=1)[k1obs>0]
#kobsr0 = k1obs[k1obs>0]
#F_obsr0 = .5*(Su_ur0 + Sv_vr0)
#
#Suobs,Svobs,Suvobs,k1obs,k2obs=sc.spatial_autocorr_fft(tri_r,U_rec_r1,V_rec_r1,transform = True,N_grid=1024,interp='cubic')
#Su_ur1 = sp.integrate.simps(Suobs,k2obs,axis=1)[k1obs>0] 
#Sv_vr1 = sp.integrate.simps(Svobs,k2obs,axis=1)[k1obs>0] 
#kobsr1 = k1obs[k1obs>0]
#F_obsr1 = .5*(Su_ur1 + Sv_vr1)

#Suobs,Svobs,Suvobs,k1obs,k2obs=sc.spatial_autocorr_fft(tri_r,U_early_r,V_early_r,transform = True,N_grid=1024,interp='cubic')
#Su_ue = sp.integrate.simps(Suobs,k2obs,axis=1)[k1obs>0]
#Sv_ve = sp.integrate.simps(Svobs,k2obs,axis=1)[k1obs>0]
#kobse = k1obs[k1obs>0]
#F_obse = .5*(Su_ue + Sv_ve)
#
#indth = kth>3*10**(-3)
#indth = kth>3*10**(-3)
#indth = kth>3*10**(-3)

plt.figure()
plt.xscale('log')
plt.yscale('log')
plt.plot(kth,kth*F_th,'o', label='theoretical') 
plt.plot(krealr,krealr*F_realr,'o', label='theo tri-interpolated') 
#plt.plot(kobsr0,kobsr0*F_obsr0,'o',label='reconstructed,numlid_0')
#plt.plot(kobsr1,kobsr1*F_obsr1,'o',label='reconstructed,numlid_1') 
#plt.plot(kobse,kobse*F_obse,'o',label='early interpolation') 
plt.plot(kth,kth*F_th_s,'o', label='reconstructed,numlid_1,early')  
plt.legend() 
plt.xlabel('k')
plt.ylabel('$F_{h}$')
plt.ylim(5*10**-4,3*10**0)

plt.figure()
plt.xscale('log')
plt.yscale('log')
plt.plot(kth,kth*Su_u_t,'o', label='theoretical')  
plt.plot(krealr,krealr*Su_urealr,'o', label='theo tri-interpolated') 
plt.plot(krealr,krealr*Su_ur,'o', label='reconstructed, refined') 
#plt.plot(kobsr0,kobsr0*Su_ur0,'o',label='reconstructed,numlid_0')
#plt.plot(kobsr1,kobsr1*Su_ur1,'o',label='reconstructed,numlid_1') 
#plt.plot(kobse,kobse*Su_ue,'o',label='early interpolation') 
plt.plot(kth,kth*Su_u_t_s,'o', label='reconstructed,numlid_1,early')  
plt.legend() 
plt.xlabel('k')
plt.ylabel('$S_{uu}$')
plt.ylim(5*10**-4,3*10**0)

plt.figure()
plt.xscale('log')
plt.yscale('log')
plt.plot(kth,kth*Sv_v_t,'o', label='theoretical')  
plt.plot(krealr,krealr*Sv_vr,'o', label='theo tri-interpolated') 
#plt.plot(kobsr0,kobsr0*Sv_vr0,'o',label='reconstructed,numlid_0')
#plt.plot(kobsr1,kobsr1*Sv_vr1,'o',label='reconstructed,numlid_1') 
#plt.plot(kobse,kobse*Sv_ve,'o',label='early interpolation')
plt.plot(kth,kth*Sv_v_t_s,'o', label='reconstructed,numlid_1,early')   
plt.legend() 
plt.xlabel('k')
plt.ylabel('$S_{vv}$')
plt.ylim(5*10**-4,3*10**0)

# In[]
plt.figure()
plt.xscale('log')
plt.yscale('log')

plt.plot(kreal,kreal*F_real,'o') 
plt.plot(kth,kth*F_th,'o') 
plt.plot(kobs,kobs*F_obs,'o') 



plt.plot(k,6*10**(-3)*k**(-2/3),'--',lw=2) 
     
# In[]



# In[]

U_mean = 15
V_mean = 0
Dir = 270*np.pi/180

ae = .05
L = 1000
G = 3.1
N_x = 512
N_y = 512

runs = 30

F_obs = np.zeros((runs,int(N_x/2)))
F_th = np.zeros((runs,int(N_x/2)))
k_th = np.zeros((runs,int(N_x/2)))
k_obs = np.zeros((runs,int(N_x/2)))

for i in range(runs):
    print(i)
    #Sepctral tensor parameters
    os.chdir(file_in_path)    
    
    input_file = 'sim.inp.txt'
    
    file = open(input_file,'w') 
    file.write('2\n') 
    file.write('2\n') 
    file.write('1\n')
    file.write('2\n') 
    file.write('3\n')  
    file.write(str(N_x)+'\n')
    file.write(str(N_y)+'\n')
    file.write(str(L_x)+'\n')
    file.write(str(L_y)+'\n')
    file.write('basic\n')
    file.write(str(ae)+'\n')
    file.write(str(L)+'\n')
    file.write(str(G)+'\n')
    file.write(str(-(i+1))+'\n')
    file.write('simu\n')
    file.write('simv\n')
    file.close() 
    
    arg = 'windsimu'+' '+input_file
    p=subprocess.run(arg)
    
    u = np.reshape(np.fromfile("simu", dtype=np.float32),(N_x,N_y))
    v = np.reshape(np.fromfile("simv", dtype=np.float32),(N_x,N_y))
    
    os.chdir(cwd)

    U = U_mean + u
    V = V_mean + v
    gamma = -(2*np.pi-Dir) # Rotation
    # Components in matrix of coefficients
    S11 = np.cos(gamma)
    S12 = np.sin(gamma)
    T = np.array([[S11,S12], [-S12,S11]])
    vel = np.array(np.c_[U.flatten(),V.flatten()]).T
    vel = np.dot(T,vel)
    U_prime = np.reshape(vel[0,:],(N_x,N_y))
    V_prime = np.reshape(vel[1,:],(N_x,N_y))
    
    vloss = num_lidar(r_s_g, np.pi-phi_s_g,U_prime,V_prime,x,y,d/2)
    vlosv = num_lidar(r_v_g, np.pi-phi_v_g,U_prime,V_prime,x,y,-d/2)
    df_s = pd.DataFrame(vloss)
    df_v = pd.DataFrame(vlosv)
    Lidar_s = (df_s,2*np.pi-phi_s_g,wsi,neighsi,indexsi) 
    Lidar_v = (df_v,2*np.pi-phi_v_g,wva,neighva,indexva)
    U_rec, V_rec= wind_field_rec(Lidar_v, Lidar_s, tree, tri, d)

    Su,Sv,Suv,k1,k2=spatial_autocorr_fft(tri,U_rec,V_rec,transform = True,N_grid=512,interp='cubic')

    Su_u = sp.integrate.simps(Su,k2,axis=1)
    Sv_v = sp.integrate.simps(Sv,k2,axis=1)
  
    k_obs[i,:] = k1[k1>0]
    F_obs[i,:] = .5*(Su_u[k1>0] + Sv_v[k1>0])

    U_mean = np.mean(U_prime.flatten())
    V_mean = np.mean(V_prime.flatten())
    # Wind direction
    gamma = np.arctan2(V_mean,U_mean)
    # Components in matrix of coefficients
    S11 = np.cos(gamma)
    S12 = np.sin(gamma)
    T = np.array([[S11,S12], [-S12,S11]])
    vel = np.array(np.c_[U_prime.flatten(),V_prime.flatten()]).T
    vel = np.dot(T,vel)
    U_t = np.reshape(vel[0,:],U_prime.shape)
    V_t = np.reshape(vel[1,:],U_prime.shape)
    
    U_mean = np.mean(U_t.flatten())
    V_mean = np.mean(V_t.flatten())
    
    U_t = U_t-U_mean
    V_t = V_t-U_mean
    
    grid = np.meshgrid(x,y)
    
    dx = np.max(np.diff(grid[0].flatten()))
    dy = np.max(np.diff(grid[1].flatten()))
    
    n = grid[0].shape[0]
    m = grid[1].shape[0]   
    # Spectra
    
    fftU = np.fft.fft2(U_t)
    fftV = np.fft.fft2(V_t)
    
    fftU  = np.fft.fftshift(fftU) 
    fftV  = np.fft.fftshift(fftV)  
    
    Suu = 2*(np.abs(fftU)**2)*dx*dy/(n*m)
    Svv = 2*(np.abs(fftV)**2)*dx*dy/(n*m)
    kx = 1/(2*dx)
    ky = 1/(2*dy)   
    k1 = kx*np.linspace(-1,1,len(Suu))
    k2 = ky*np.linspace(-1,1,len(Suu))
    
    Su_u_t = sp.integrate.simps(Suu,k2,axis=1)
    Sv_v_t = sp.integrate.simps(Svv,k2,axis=1)
    
    k_th[i,:] = k1[k1>0]
    F_th[i,:] = .5*(Su_u_t[k1>0] + Sv_v_t[k1>0])


with open('S_sim.pkl', 'wb') as V_t:
     pickle.dump((k_th,k_obs,F_th,F_obs),V_t) 





    

# In[]

