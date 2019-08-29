# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 14:21:05 2019

Main (Example) for wind field simulation and reconstruction

##### Geometry definition (both synthetic and synthetic after reconst.) #####
# Grid points in Cartesian X-Y (2**n)
N_x = 2048
N_y = 2048

# Scan 0 geometry input
# rmin0,rmax0,nr0,phimin0,phimax0,np0,orig0
rmin0,rmax0,nr0,phimin0,phimax0,np0,orig0 = 105,7000,198,256,344,45,[6322832.3,0]
rp0 = (rmin0,rmax0,nr0,phimin0,phimax0,np0,orig0)

# Scan 1 geometry input
# rmin1,rmax1,nr1,phimin1,phimax1,np1,orig1
rmin1,rmax1,nr1,phimin1,phimax1,np1,orig1 = (105,7000,198,196,284,45,[6327082.4,0])
rp1 = (rmin1,rmax1,nr1,phimin1,phimax1,np1,orig1)

# Grids, polar and cartesian
d = orig1-orig0

# Polar grids for Scan 0 (local and translated)
r_0_g, phi_0_g, r_0_t, phi_0_t = geom_polar_grid(rmin0,rmax0,nr0,phimin0,phimax0,np0,-d)

# Polar grids for Scan 1 (local and translated)
r_1_g, phi_1_g, r_1_t, phi_1_t = geom_polar_grid(rmin1,rmax1,nr1,phimin1,phimax1,np1,-d)


L_x, L_y, grid, x, y, tri, grid_new, d = geom_syn_field(rp0, rp1, N_x, N_y)

# Triangulation and weights for each scan
dl = 75
vtx0, wts0, w0, c_ref0, s_ref0, shapes = early_weights_pulsed(r_0_g,np.pi-phi_0_g, dl, dir_mean , tri, -d/2, y[0]/2)
vtx1, wts1, w1, c_ref1, s_ref1, shapes = early_weights_pulsed(r_1_g,np.pi-phi_1_g, dl, dir_mean , tri, d/2, y[0]/2)
  
##### 2D Turbulent wind field generation #####
# Mann-model parameters, example
L_i, G_i, ae_i, seed_i = 750, 2.5, .05, 4

#From file
u_file_name = 'simu'+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
v_file_name = 'simv'+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
u = np.reshape(np.fromfile(u_file_name, dtype=np.float32),(N_x,N_y)).T
v = np.reshape(np.fromfile(v_file_name, dtype=np.float32),(N_x,N_y)).T

#Choose the right path
import tkinter as tkint
import tkinter.filedialog

#Generated
root = tkint.Tk()
file_in_path = tkint.filedialog.askdirectory(parent=root,title='Choose a sim. Input dir')
root.destroy()

u, v = wind_sim(ae_i, L_i, G_i, seed_i, N_x, N_y, L_x, L_y, file_in_path)

# Mean wind speed and Direction
Dir = np.linspace(90,270,5)*np.pi/180
dir_mean = Dir[4]
u_mean = 15

#Turbulent wind field
U_in = u_mean + u
V_in = 0 + v

#Numerical lidar sampling
vlos0 = num_pulsed_lidar(U_in,V_in,vtx0,wts0,w0,c_ref0, s_ref0, shapes)
vlos1 = num_pulsed_lidar(U_in,V_in,vtx1,wts1,w1,c_ref1, s_ref1, shapes)

#Interpolation to cartesian grid
vlos1_int_sq = sp.interpolate.griddata(np.c_[(r_v_t*np.cos(phi_v_t)).flatten(),(r_v_t*np.sin(phi_v_t)).flatten()],
                                             vlos1.flatten(), (grid_new[0].flatten(), grid_new[1].flatten()), method='cubic')
vlos0_int_sq = sp.interpolate.griddata(np.c_[(r_s_t*np.cos(phi_s_t)).flatten(),(r_s_t*np.sin(phi_s_t)).flatten()],
                                             vlos0.flatten(), (grid_new[0].flatten(), grid_new[1].flatten()), method='cubic')

vlos1_int_sq = np.reshape(vlos1_int_sq,grid_new[0].shape)
vlos0_int_sq = np.reshape(vlos0_int_sq,grid_new[0].shape)

#Wind field reconstruction (overlaping are of the two scans)
U,V = dir_rec_rapid(vlos1_int_sq.flatten(),vlos0_int_sq.flatten(), phi_tri_v_s.flatten(),phi_tri_s_s.flatten(),grid_new[0].shape)

@author: lalc
"""

import numpy as np
import scipy as sp
import os
import subprocess
import ppiscanprocess.windfieldrec as wr
from scipy.spatial import Delaunay

# In[]

def wind_sim(ae, L, G, seed, N_x, N_y, L_x, L_y, file_in_path,pre):  
    cwd = os.getcwd()    
    os.chdir(file_in_path)    
    input_file = 'sim.inp.txt'     
    file = open(input_file,'w') 
    file.write('2\n') #fieldDim
    file.write('2\n') #NComp
    file.write('1\n') #u
    file.write('2\n') #v
    file.write('3\n') #w 
    file.write(str(N_x)+'\n')
    file.write(str(N_y)+'\n')
    file.write(str(L_x)+'\n')
    file.write(str(L_y)+'\n')
    file.write('basic\n')
    file.write(str(ae)+'\n')
    file.write(str(L)+'\n')
    file.write(str(G)+'\n')
    file.write(str(seed)+'\n')
    name_u = pre+'u'+str(L)+str(G)+str(ae)+str(seed)
    name_v = pre+'v'+str(L)+str(G)+str(ae)+str(seed)
    file.write(name_u+'\n')
    file.write(name_v+'\n')
    file.close()    
    arg = 'windsimu'+' '+input_file
    p=subprocess.run(arg)    
#    u = np.reshape(np.fromfile(name_u, dtype=np.float32),(N_x,N_y))
#    v = np.reshape(np.fromfile(name_v, dtype=np.float32),(N_x,N_y)) 
    u = np.fromfile(name_u, dtype=np.float32)
    v = np.fromfile(name_v, dtype=np.float32)
    os.chdir(cwd)    
    return (u,v)

# In[Rapid wind field reconstruction]

def dir_rec_rapid(V_a,V_b,a,b,shape):
    Sa = np.sin(a)/np.sin(a-b)
    Sb = np.sin(b)/np.sin(a-b)
    Ca = np.cos(a)/np.sin(a-b)
    Cb = np.cos(b)/np.sin(a-b)
    U = (Sb*V_a-Sa*V_b)
    V = (-Cb*V_a+Ca*V_b)
    return (np.reshape(U,shape),np.reshape(V,shape))

# In[Numerical lidar] 
####################################################################################################################################    
## Comment for Konstantinos: This is the one I ended up using, the weighting function from early weights is the one from Alexander Meyer's paper
####################################################################################################################################    
def num_pulsed_lidar(U_in,V_in,vtx,wts,w,c_ref, s_ref, shapes):
    # Translate (x,y) field to lidar origin and transform to polar coordinates
    # Translate    
    n = shapes[2]
    m = shapes[3]
    #Weight Normalization
    w = w/np.reshape(np.repeat(np.sum(w,axis=1),w.shape[1]),w.shape)
    
    U = np.reshape(interpolate(U_in, vtx, wts, fill_value=np.nan),c_ref.shape)
    V = np.reshape(interpolate(V_in, vtx, wts, fill_value=np.nan),c_ref.shape)   
    V_L = c_ref*U+s_ref*V  
    VLw = np.zeros((V_L.shape[0],int((V_L.shape[1]-1)/(n-1))))
    for i in range(V_L.shape[0]):
        VLw[i,:] = np.dot(w,np.where(np.isnan(V_L.T[:,i]),0,V_L.T[:,i]))    
    w_p = np.ones(VLw.shape)/(m-1) 
    VLw = -(VLw[:-1,:]*w_p[:-1,:])     
     
    return np.flip(np.nansum(VLw.reshape(-1,(m-1),VLw.shape[-1]),axis=1),axis=0)  
####################################################################################################################################
####################################################################################################################################
def num_lidar_rot_del(U_in,V_in,vtx,wts,w,c_ref, s_ref, shapes):
    # Translate (x,y) field to lidar origin and transform to polar coordinates
    # Translate    
    n = shapes[2]
    m = shapes[3]
    
    U = interpolate(U_in, vtx, wts, fill_value=np.nan)
    V = interpolate(V_in, vtx, wts, fill_value=np.nan)
    
    V_L = c_ref*U+s_ref*V
    
    V_L = np.reshape(V_L, (shapes[0],shapes[1]))
    
    V_L = (V_L[:,:-1]*w[:,:-1]).T
    
    V_L = np.nansum(V_L.reshape(-1,(n-1),V_L.shape[-1]),axis=1).T

    w_p = np.ones(V_L.shape)/(m-1)
    
    V_L = -(V_L[:-1,:]*w_p[:-1,:])     
    
    print(U.shape,V.shape,V_L.shape,shapes)
     
    return np.flip(np.nansum(V_L.reshape(-1,(m-1),V_L.shape[-1]),axis=1),axis=0) 

# In[Interpolation for rotated wind fields]
####################################################################################################################################
## Comment for Konstantinos: This is the one I ended up using, the weighting function from early weights is the one from Alexander's paper
####################################################################################################################################    
def interpolate(values, vtx, wts, fill_value=np.nan):
    ret = np.einsum('nj,nj->n', np.take(values, vtx), wts)
    ret[np.any(wts < 0, axis=1)] = fill_value
    return ret

def interp_weights2(uv, tri, d = 2):
    print('triangulation...')
    simplex = tri.find_simplex(uv)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uv - temp[:, d]
    bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))
####################################################################################################################################
## Comment for Konstantinos: This is the one I ended up using, the weighting function from early weights is the one from Alexander's paper
####################################################################################################################################
def early_weights_pulsed(r, phi, dl, dir_mean , tri, d, center, n=21, m=51):
    gamma = (2*np.pi-dir_mean) 
    r_unique = np.unique(r)
    phi_unique = np.unique(phi)
    delta_r = np.min(np.diff(r_unique))
    delta_phi = np.min(np.diff(phi_unique))
    r_refine = np.linspace(r_unique.min()-delta_r/2,r_unique.max()+
                           delta_r/2,len(r_unique)*(n-1)+1)      
    phi_refine = np.linspace(phi_unique.min()-delta_phi/2, phi_unique.max()+
                             delta_phi/2, len(phi_unique)*(m-1)+1)
    r_t_refine, phi_t_refine = np.meshgrid(r_refine,phi_refine)    
    
    #LOS angles        
    s_ref = np.sin(phi_t_refine-gamma)
    c_ref = np.cos(phi_t_refine-gamma)    
    r_t_refine, phi_t_refine = wr.translationpolargrid((r_t_refine, phi_t_refine),d)
    x_t_refine, y_t_refine = r_t_refine*np.cos(phi_t_refine), r_t_refine*np.sin(phi_t_refine)

    # Rotation and translation    
    
    x_trans = -(center)*np.sin(gamma)
    y_trans = (center)*(1-np.cos(gamma))
    S11 = np.cos(gamma)
    S12 = np.sin(gamma)
    T1 = np.array([[1,0,x_trans], [0,1, y_trans], [0, 0, 1]])
    R = np.array([[S11,S12,0], [-S12,S11, 0], [0, 0, 1]])    
    Xx = np.array(np.c_[x_t_refine.flatten(), y_t_refine.flatten(),
                                    np.ones(len(y_t_refine.flatten()))]).T
    Xx = np.dot(T1,np.dot(R,Xx))
    uv = Xx[:2,:].T
    vtx, wts = interp_weights2(uv, tri, d = 2)
          
    aux_1 = np.reshape(np.repeat(r_unique,len(r_refine),axis = 0),(len(r_unique),len(r_refine)))
    aux_2 = np.reshape(np.repeat(r_refine,len(r_unique)),(len(r_refine),len(r_unique))).T
    
    r_F = aux_1-aux_2
    rp = dl/(2*np.sqrt(np.log(2)))
    erf = sp.special.erf((r_F+.5*delta_r)/rp)-sp.special.erf((r_F-.5*delta_r)/rp)
    w = (1/2/delta_r)*erf   
    shapes = np.array([phi_t_refine.shape[0], phi_t_refine.shape[1], n, m])        
    return (vtx, wts, w, c_ref, s_ref, shapes)

####################################################################################################################################
## Comment for Konstantinos: This early weights was used for a different tasks and it is not realistic
####################################################################################################################################
    
def early_weights_kernel(r, phi, dir_mean , tri, d, center, n=21, m=51):
    gamma = (2*np.pi-dir_mean) 
    r_unique = np.unique(r)
    phi_unique = np.unique(phi)
    delta_r = np.min(np.diff(r_unique))
    delta_phi = np.min(np.diff(phi_unique))
    r_refine = np.linspace(r_unique.min()-delta_r/2,r_unique.max()+
                           delta_r/2,len(r_unique)*(n-1)+1)      
    phi_refine = np.linspace(phi_unique.min()-delta_phi/2, phi_unique.max()+
                             delta_phi/2, len(phi_unique)*(m-1)+1)
    r_t_refine, phi_t_refine = np.meshgrid(r_refine,phi_refine)    
    
    #LOS angles        
    s_ref = np.sin(phi_t_refine-gamma)
    c_ref = np.cos(phi_t_refine-gamma)    
    r_t_refine,phi_t_refine = wr.translationpolargrid((r_t_refine, phi_t_refine),d)
    x_t_refine, y_t_refine = r_t_refine*np.cos(phi_t_refine), r_t_refine*np.sin(phi_t_refine)

    # Rotation and translation    
    
    x_trans = -(center)*np.sin(gamma)
    y_trans = (center)*(1-np.cos(gamma))
    S11 = np.cos(gamma)
    S12 = np.sin(gamma)
    T1 = np.array([[1,0,x_trans], [0,1, y_trans], [0, 0, 1]])
    R = np.array([[S11,S12,0], [-S12,S11, 0], [0, 0, 1]])    
    Xx = np.array(np.c_[x_t_refine.flatten(), y_t_refine.flatten(),
                                    np.ones(len(y_t_refine.flatten()))]).T
    Xx = np.dot(T1,np.dot(R,Xx))
    uv = Xx[:2,:].T
    vtx, wts = interp_weights2(uv, tri, d = 2)   
    h= 2*(np.r_[np.repeat(r_unique,(n-1)),r_unique[-1]]-r_refine)/delta_r     
    w = .75*(1-h**2) 
    w = np.reshape(np.repeat(w,phi_t_refine.shape[0]),(phi_t_refine.T).shape).T
    norm = np.sum(w[0,:(n-1)])
    w = w/norm
    shapes = np.array([phi_t_refine.shape[0], phi_t_refine.shape[1], n, m])        
    return (vtx, wts, w, c_ref, s_ref, shapes)


##################
# In[Geometry generation]
# input for numerical lidar
def geom_polar_grid(rmin,rmax,nr,phimin,phimax,nphi,d):
    r = np.linspace(rmin,rmax,nr)
    phi = np.linspace(phimin,phimax,nphi)*np.pi/180  
    r_g, phi_g = np.meshgrid(r,phi)
    r_t,phi_t = wr.translationpolargrid((r_g, np.pi-phi_g),d/2)
    return (r_g, phi_g, r_t, phi_t)
    
def geom_syn_field(rp0, rp1, N_x, N_y):    
    # Polar grid 2 horizontal scans
    rmin0,rmax0,nr0,phimin0,phimax0,np0,orig0 = rp0
    rmin1,rmax1,nr1,phimin1,phimax1,np1,orig1 = rp1 
    d = orig1-orig0     
    r_0_g, phi_0_g, r_0_t, phi_0_t = geom_polar_grid(rmin0,rmax0,nr0,phimin0,phimax0,np0,-d)
    r_1_g, phi_1_g, r_1_t, phi_1_t = geom_polar_grid(rmin1,rmax1,nr1,phimin1,phimax1,np1,d)
    x_max = np.max(np.r_[(r_0_t*np.cos(phi_0_t)).flatten(),(r_1_t*np.cos(phi_1_t)).flatten()])
    x_min = np.min(np.r_[(r_0_t*np.cos(phi_0_t)).flatten(),(r_1_t*np.cos(phi_1_t)).flatten()])
    y_max = np.max(np.r_[(r_0_t*np.sin(phi_0_t)).flatten(),(r_1_t*np.sin(phi_1_t)).flatten()])
    y_min = np.min(np.r_[(r_0_t*np.sin(phi_0_t)).flatten(),(r_1_t*np.sin(phi_1_t)).flatten()]) 
    L_x = x_max-x_min
    L_y = y_max-y_min
    x = np.linspace(x_min,x_max,N_x)
    y = np.linspace(y_min,y_max,N_y)
    grid = np.meshgrid(x,y) 
    tri = Delaunay(np.c_[grid[0].flatten(),grid[1].flatten()], qhull_options = "QJ")        
    # Square grid for reconstruction
    _,tri_overlap,_,_,_,_,_,_ = wr.grid_over2((r_1_g, np.pi-phi_1_g),(r_0_g, np.pi-phi_0_g),-d)
    
    r_min=np.min(np.sqrt(tri_overlap.x**2+tri_overlap.y**2))
    d_grid = r_min*2*np.pi/180
    n_next = int(2**np.ceil(np.log(L_y/d_grid+1)/np.log(2))) 
    x_new = np.linspace(x.min(),x.max(),n_next)
    y_new = np.linspace(y.min(),y.max(),n_next)
    grid_new = np.meshgrid(x_new,y_new)
    
    return (L_x, L_y, grid, x, y, tri, grid_new,d)

# Masks for power spectra calculation (for non-reconstructed wind field)
def win_field_mask_tri(dir_mean,xtrans, ytrans, tri, grid):
    gamma = (2*np.pi-dir_mean)
    S11 = np.cos(gamma)
    S12 = np.sin(gamma)
    T1 = np.array([[1,0,xtrans], [0,1, ytrans], [0, 0, 1]])
    T2 = np.array([[1,0,-xtrans], [0,1, -ytrans], [0, 0, 1]])
    R = np.array([[S11,S12,0], [-S12,S11, 0], [0, 0, 1]])
    T = np.dot(np.dot(T1,R),T2)
    Xx = np.array(np.c_[tri.x,tri.y,np.ones(len(tri.x))]).T
    Xx = np.dot(T,Xx)
    tri_rot = Delaunay(Xx.T[:,:2], qhull_options = "QJ")               
    mask_rot = tri_rot.find_simplex(np.c_[grid[0].flatten(),grid[1].flatten()])==-1                
    return np.reshape(mask_rot,grid[0].shape)                
   
# In[Noise generation]
# Perlin Noise
####################################################################################################################################
## Comment for Konstantinos: This is not used in your project
####################################################################################################################################    
def perlin_noise(x,y,scale=30, azim_frac = .3, rad_lim = .1, dr_max = .3, period = 256, tot= 'no'):
    
    n, m = x.shape   
    x = x.flatten()
    y = y.flatten()
    
    GRAD3 = np.array(((1,1,0),(-1,1,0),(1,-1,0),(-1,-1,0), 
    	(1,0,1),(-1,0,1),(1,0,-1),(-1,0,-1), 
    	(0,1,1),(0,-1,1),(0,1,-1),(0,-1,-1),
    	(1,1,0),(0,-1,1),(-1,1,0),(0,-1,-1),))
    
    perm = list(range(period))
    perm_right = period - 1
    for i in list(perm):
        j = np.random.randint(0, perm_right)
        perm[i], perm[j] = perm[j], perm[i]
    permutation = np.array(tuple(perm) * 2)
    # Simplex skew constants
    F2 = 0.5 * (np.sqrt(3.0) - 1.0)
    G2 = (3.0 - np.sqrt(3.0)) / 6.0
    F3 = 1.0 / 3.0
    G3 = 1.0 / 6.0
	# Skew input space to determine which simplex (triangle) we are in
    s = (x + y) * F2
    i = np.floor(x + s)
    j = np.floor(y + s)
    t = (i + j) * G2
    x0 = x - (i - t) # "Unskewed" distances from cell origin
    y0 = y - (j - t)   
    ind_loc = x0 > y0   
    i1 = np.ones(x.shape)
    j1 = np.zeros(x.shape)  
    i1[~ind_loc] = 0
    j1[~ind_loc] = 1
    
    i1 = i1.astype(int)
    j1 = i1.astype(int)
		
    x1 = x0 - i1 + G2 # Offsets for middle corner in (x,y) unskewed coords
    y1 = y0 - j1 + G2
    x2 = x0 + G2 * 2.0 - 1.0 # Offsets for last corner in (x,y) unskewed coords
    y2 = y0 + G2 * 2.0 - 1.0

    # Determine hashed gradient indices of the three simplex corners
    perm = permutation
    ii = (i % period).astype(int)
    jj = (j % period).astype(int)
       
    gi0 = perm[ii + perm[jj]] % 12
    gi1 = perm[ii + i1 + perm[jj + j1]] % 12
    gi2 = perm[ii + 1 + perm[jj + 1]] % 12

    # Calculate the contribution from the three corners
    noise = np.zeros(x.shape)
    
    tt = 0.5 - x0**2 - y0**2 
    ind_tt = tt > 0
    g = GRAD3[gi0,:]

    noise[ind_tt] = tt[ind_tt]**4 * (g[ind_tt,0] * x0[ind_tt] + g[ind_tt,1] * y0[ind_tt])
    
    tt = 0.5 - x1**2 - y1**2
    ind_tt = tt > 0
    g = GRAD3[gi1,:]
    noise[ind_tt] = noise[ind_tt] + tt[ind_tt]**4 * (g[ind_tt,0] * x1[ind_tt] + g[ind_tt,1] * y1[ind_tt])
    
    tt = 0.5 - x2**2 - y2**2
    ind_tt = tt > 0
    g = GRAD3[gi2,:]
    noise[ind_tt] = noise[ind_tt] + tt[ind_tt]**4 * (g[ind_tt,0] * x2[ind_tt] + g[ind_tt,1] * y2[ind_tt])
    
    if tot=='no':
    # mask definition
        azim_frac = np.random.uniform(azim_frac,1)
        rad_lim = np.random.uniform(rad_lim,1)
        dr = np.random.uniform(.1,dr_max)
        
        # azimuth positions
        n_pos = int(azim_frac*n)
        pos_azim = np.random.randint(0,n,size = n_pos)
        # radial positions
        # center
        r_mean = int(rad_lim*m)
        r_std = int(dr*m)
        # positions
        pos_rad = np.random.randint(r_mean-r_std,r_mean+r_std,size = n_pos)
        #print(n,m)
        n, m = np.meshgrid(np.arange(m),np.arange(n))
        #print(n.shape)
        ind = np.zeros(n.shape)
        
        for i,nn in enumerate(pos_azim):
            #print(pos_rad[i],r_mean,r_mean-r_std,r_mean+r_std)
            ind_r = n[nn,:]>=pos_rad[i]
            ind[nn,ind_r] = 1
        ind = (ind == 1).flatten()
        noise[~ind] = 0.0
    
        #normalize
        a = np.max(noise)
        c = np.min(noise)
        if (a-c) > 0:
            b = 1
            d = -1   
            m = (b - d) / (a - c)
            noise = (m * (noise - c)) + d
        noise[~ind] = 0.0 
        
    if tot=='yes':
        #normalize
        a = np.max(noise)
        c = np.min(noise)
        if (a-c) > 0:
            b = 1
            d = -1   
            m = (b - d) / (a - c)
            noise = (m * (noise - c)) + d
    
    return noise # scale noise to [-1, 1]


    



    

    

