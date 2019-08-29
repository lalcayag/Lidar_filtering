# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 14:41:55 2018

Package for wind field reconstruction from PPI scans (it might be used also with other type of scan)

@author: 
Leonardo Alcayaga
lalc@dtu.dk

"""
# In[Packages used]

import numpy as np
import scipy as sp
from scipy.spatial import Delaunay

from sklearn.neighbors import KDTree
from sklearn.neighbors import KNeighborsRegressor
from matplotlib.tri import Triangulation,TriFinder,TriAnalyzer,CubicTriInterpolator

# In[############# [Functions] #################]

# In[un-structured grid generation]

def translationpolargrid(mgrid,h):  
    """
    Function that performs a linear translation from (r,theta) = (x,y) -> (x0,y0) = (x+h[0],y+h[1])

    Input:
    -----
        mgrid                 - Tuple containing (rho,phi), polar coordinates to transform
        
        h                     - Linear distace of translation   
        
    Output:
    ------
        rho_prime, phi_prime  - Translated polar coordinates      
    """
    # Original polar coordinates
    rho = mgrid[0]
    phi = mgrid[1]
    # Trnasformation to cartesian
    x = rho*np.cos(phi)
    y = rho*np.sin(phi)
    # Translation in cartesian
    x0 = x - h[0]
    y0 = y - h[1]
    # Transformation back to polar coordinates
    rho_prime = np.sqrt(x0**2+y0**2)
    phi_prime = np.arctan2(y0,x0)
    return(rho_prime, phi_prime)

# In[Nearest point]

def nearestpoint(mg0,mg1,dr,dp):
    """
    Function to identify the points inside the overlapping area of two PPI scans. A nearest neighbour
    approach is used, in polar coordinates.

    Input:
    -----
        mg0      - Tuple with (r0,p0), points in polar coordinates of the first scan in a common
                   frame. This means that the local PPI scan coordinates must be translated to a
                   common point with the other scans.
        
        mg1      - Tuple with (r1,p1), points in polar coordinates of the first scan in a common
                   frame. This means that the local PPI scan coordinates must be translated to a
                   common point with the other scans.
        
        dr       - Grid spacing in the r component in polar coordinates.
        
        dp       - Grid spacing in the azimuth component in polar coordinates.
        
    Output:
    ------
        nearest  - Tuple with (r,p,ind), namely, r, all the r coordinates of points within the
                   overlapping area, p, same for the azimuth coordinate and ind, the corresponding
                   index in the original flatten array of coordinates of one scan respect to 
                   the other.     
    """
    # Coordinates extraction. A common frame is assumed.
    r0 = mg0[0].flatten()
    r1 = mg1[0].flatten()
    p0 = mg0[1].flatten()
    p1 = mg1[1].flatten()
    # Initialization of list with nearest points to one scan.
    raux = []
    paux = []
    iaux = []
    for i in range(len(r1)):
        # Distances of all points in the two scans. In polar coordinates
        dist1 = np.sqrt((r0-r1[i])**2)
        dist2 = np.sqrt((p0-p1[i])**2)
        # Index of points in scan 1 within a neighbourhood dr x dp
        ind = ((dist1<=dr) & (dist2<=dp)).nonzero()[0] 
        # Append of those points
        raux.append(r0[ind])
        paux.append(p0[ind])
        iaux.append(ind)
    # Flatten list    
    r_flat= [item for sublist in raux for item in sublist]
    p_flat= [item for sublist in paux for item in sublist]
    i_flat= [item for sublist in iaux for item in sublist]
    # List with corresponding r and azimuth coordinates of nearest points, and corresponding index
    # in original array
    polar = zip(r_flat, p_flat, i_flat)
    # Unique values
    unique = [list(t) for t in zip(*list(set(polar)))] 
    # Final output
    nearest = (np.array(unique[0]),np.array(unique[1]),np.array(unique[2]))
    return nearest

# In[Overlapping grid]

def grid_over2(mg0, mg1, d):
    """
    Function to define coordinates (in a common frame) of the intersection of the laser-beams from
    two PPI scans. This function uses a kd-tree appoach that make the grid generation independent of 
    scan geometry, since finds intersection points only finding nearest neighbours between scans

    Input:
    -----
        mg0      - Tuple with (r0,p0), points in polar coordinates of the first scan in local frame
                   and non translated.
        
        mg1      - Tuple with (r1,p1), points in polar coordinates of the first scan in local frame
                   and non translated.
        
        d        - Linear distance between LiDARs.
        
    Output:
    ------
        tree_g   - Kd-tree of laser beam intersection points
        
        tri      - Unstructured grid from Delaunay triangulation with triangles corners as
                   intersection points
        
        w0       - Weights to be used in wind field reconstruction, depending on distance of 
                   current scan 0 point to the nearest intersection point.
        
        n0       - Corresponding label of scan-0 points within the neighbourhood of each 
                   intersection point.
        
        i_o_0    - Original index of neigbouring points in scan 0.
        
        w1       - Same weights but for scan 1
        
        n1       - Same labels bit for scan1 
        
        i_o_1    - Same index but this time for scan 1
    
    """
    # Polar grid resolution is used as nearest neighbour distance to estimate
    # overlaping scanning area
    dr = min(np.diff(np.unique(mg0[0].flatten())))/2
    dp = min(np.diff(np.unique(mg0[1].flatten())))/2   
    # Translation of grids
    r0, p0 = translationpolargrid(mg0,-d/2)
    r1, p1 = translationpolargrid(mg1,d/2) 
    # Overlapping points
    r_o_0, p_o_0, i_o_0 = nearestpoint((r0,p0),(r1,p1),dr,dp)
    r_o_1, p_o_1, i_o_1 = nearestpoint((r1,p1),(r0,p0),dr,dp)
    # Cartesian trees from overlapping points of each scan
    pos0 = np.c_[r_o_0*np.cos(p_o_0),r_o_0*np.sin(p_o_0)]    
    tree_0 = KDTree(pos0)
    pos1 = np.c_[r_o_1*np.cos(p_o_1),r_o_1*np.sin(p_o_1)]    
    tree_1 = KDTree(pos1)  
    # Intersection points, first iteration will find pair of points
    ind0,dist0 = tree_0.query_radius(tree_1.data, r=3*dr/2,return_distance = True,sort_results=True)
    ind1,dist1 = tree_1.query_radius(tree_0.data, r=3*dr/2,return_distance = True,sort_results=True) 
    ind00 = []
    ind01 = []
    for i,j in zip(ind0,range(len(ind0))):
        if i.size > 0:
            #indices
            ind00.append(np.asscalar(i[0]))
            ind01.append(j)           
    ind10 = []
    ind11 = []
    for i,j in zip(ind1,range(len(ind1))):
        if i.size > 0:
            #indices
            ind11.append(np.asscalar(i[0]))
            ind10.append(j)      
    # Center of grafity of near-intersection points
    posg0=np.c_[0.5*(pos0[:,0][ind00]+pos1[:,0][ind01]),0.5*(pos0[:,1][ind00]+pos1[:,1][ind01])] 
    posg1=np.c_[0.5*(pos0[:,0][ind10]+pos1[:,0][ind11]),0.5*(pos0[:,1][ind10]+pos1[:,1][ind11])]  
    posg = np.vstack((posg0,posg1))
    unique = [list(t) for t in zip(*list(set(zip(posg[:,0], posg[:,1]))))] 
    posg = np.c_[unique[0],unique[1]]
    tree_g = KDTree(posg)  
    # Intersection points, final iteration 
    # Identification of nearest neighbours to each preestimated intersection point
    indg, distg = tree_g.query_radius(tree_g.data, r=2*dr, return_distance = True, sort_results=True)
    S = sorted(set((tuple(sorted(tuple(i))) for i in indg if len(tuple(i))>1)))
    nonS = [np.asscalar(i) for i in indg if len(tuple(i))==1]
    temp = [set(u) for u in S]
    S = []
    for ti in temp:
        aux = [t for t in temp if t!=ti]
        if not any(ti <= u for u in aux):
            S.append(list(ti))
    aux=np.array([np.mean(posg[list(p),:],axis=0) for p in S])  
    posg = np.vstack((aux,posg[nonS])) 
    tree_g = KDTree(posg)
    # Diastances and labels of neighbours to intersection points
    d0,n0 = tree_g.query(tree_0.data, return_distance = True)
    d1,n1 = tree_g.query(tree_1.data, return_distance = True) 
    dg,ng = tree_g.query(tree_g.data, k = 3, return_distance = True)
    # Correct dimensions!
    d0 = np.squeeze(d0)
    d1 = np.squeeze(d1)
    n0 = np.squeeze(n0)
    n1 = np.squeeze(n1)
    dg = np.squeeze(dg)
    ng = np.squeeze(ng)
    
    # Weights' bandwidth estimation 
    rg = dg[:,1]/2
    n0_bigger = np.unique(n0[(np.max(np.c_[rg[n0],d0],axis=1)-d0 == 0).nonzero()[0]]) 
    n1_bigger = np.unique(n0[(np.max(np.c_[rg[n0],d0],axis=1)-d0 == 0).nonzero()[0]]) 
    #repeated values

#    n0_r = (np.diff(np.sort(n0_bigger))==0).nonzero()[0]  
#    n0_r = np.unique(np.r_[n0_r,n0_r+1])
#    n0_r = np.sort(n0_bigger)[n0_r]
#    n0_r = np.unique(n0_r)
    
    for i in n0_bigger:
        rg[i] = np.max(d0[n0==i])
    for i in n1_bigger:
        rg[i] = np.max([np.max(d1[n1==i]),rg[i]])        
    
    rg=rg*1.01
    
    # Weights estimation

#    w0 = np.squeeze(d0)**-1
#    w1 = np.squeeze(d1)**-1

##   Gaussian        
#    rg=rg/2
#    w0 = 1/(rg[n0]*np.sqrt(2*np.pi))*np.exp(-d0**2/(2*rg[n0]**2))
#    w1 = 1/(rg[n1]*np.sqrt(2*np.pi))*np.exp(-d1**2/(2*rg[n1]**2))

#   Epanechnikov        
    
    w0 = .75*(1-(d0/rg[n0])**2)
    w1 = .75*(1-(d1/rg[n1])**2)


#    w0 = (rg[n0]**2-d0**2)/(d0**2+rg[n0]**2)
#    w1 = (rg[n1]**2-d1**2)/(d1**2+rg[n1]**2)
    # Delaunay triangulation of intersection points
    tri = Triangulation(posg[:,0], posg[:,1])  
    return (tree_g, tri, w0, n0, i_o_0, w1, n1, i_o_1)

# In[Wind field reconstruction]

def wind_field_rec(Lidar0, Lidar1, tree, triangle, d):
    """
    Function to reconstruct horizontal wind field (2D) in Cartesian coordinates, taking advantage of
    Kd-tree structures. This function works with PPI scans that are not synchronous and using 
    equation (12) in [1]. Continuity might be included in this formulation, and uncertainty 
    estimation.

    Input:
    -----
        Lidar_i  - Tuple with (vr_i,r_i,phi_i,w_i,neigh_i,index_i):
            
                        vr_i          - Array with V_LOS of Lidar_i
                        phi_i         - Arrays with polar coordinates of the first scan in local 
                                        frame, non-translated.
                        w_i           - Array with weights of each measurement vr_i dependant on
                                        distance from (r_i, phi_i) to the correponding unstructured 
                                        grid point in triangle.
                        neigh_i       - Array with indexes of the corresponding nearest intersection
                                        point.
                        index_i       - Array with indexes of the original polar grid in local Lidar
                                        coordinates.
        
        tree     - Kd-tree of the unstructured grid of laser beams intersection points.
        
        triangle - Delaunay triangulation with the unstructured grid of laser beams intersection
                   points.
        
        d        - Linear distance between Lidar_i and Lidar_j.
        
    Output:
    ------
        U, V     - Cartesian components of wind speed.
    
    
    [1] Michel Chong and Claudia Campos, Extended Overdetermined Dual-Doppler Formalismin 
        Synthesizing Airborne Doppler Radar Data, 1996, Journal of Atmopspheric and Oceanic Technology
    """
    # Input extraction
    vr0, phi0_old, w0, neigh0, index0 = Lidar0
    vr1, phi1_old, w1, neigh1, index1 = Lidar1 
    vr0 = vr0.values.flatten()[index0]
    vr1 = vr1.values.flatten()[index1]
    phi0_old = phi0_old.flatten()[index0]
    phi1_old = phi1_old.flatten()[index1]
    # Initialization of wind components
    U = np.ones(len(tree.data))
    V = np.ones(len(tree.data))
    U[U==1] = np.nan
    V[V==1] = np.nan
    # Loop over each member of the un-structured grid tree
    for i in range(len(tree.data)):
        # Identification of neighbouring observations to intersection points.
        ind0 = (neigh0==i).nonzero()[0]
        ind1 = (neigh1==i).nonzero()[0]
        # Selection of valid observations only
        ind00 = (~np.isnan(vr0[ind0])).nonzero()[0]
        ind11 = (~np.isnan(vr1[ind1])).nonzero()[0]
        # Corresponding V_LOS
        vr_0 = vr0[ind0][ind00]
        vr_1 = vr1[ind1][ind11]
        # Corresponding weights
        w_0  = w0[ind0][ind00]
        w_1  = w1[ind1][ind11]
        # Transformation to cartesian coordinates
        sin0 = np.sin(phi0_old[ind0][ind00])
        cos0 = np.cos(phi0_old[ind0][ind00]) 
        sin1 = np.sin(phi1_old[ind1][ind11])
        cos1 = np.cos(phi1_old[ind1][ind11])
        # LSQ fitting of all observations in the neighbourhood, only if it has valid observations 
        # from at least two Lidar's
        if (w_0.size > 0) and (w_1.size > 0):
            # Coefficients of linear equation system, including weights
            beta_i = np.r_[sin0*w_0,sin1*w_1]
            alpha_i = np.r_[cos0*w_0,cos1*w_1]
            V_i = np.r_[vr_0*w_0,vr_1*w_1]
            # Components in matrix of coefficients
            S11 = np.nansum(alpha_i**2)
            S12 = np.nansum(alpha_i*beta_i)
            S22 = np.nansum(beta_i**2)
            # V_LOS solution vector
            V11 = np.nansum(alpha_i*V_i)
            V22 = np.nansum(beta_i*V_i)
            # Coeffiecient matrix
            a = np.array([[S11,S12], [S12,S22]])
            b = np.array([V11,V22])
            # Linear system solution
            x = np.linalg.solve(a, b)
            U[i] = x[0]
            V[i] = x[1]
        else:
            # Grid points lacking information from one or both Lidars are assumed as NaN.
            U[i], V[i] = np.nan, np.nan
    return (U, V)

# In[]   
def data_interp_triang(U,V,x,y,dt): 
    """
    Function to interpolate wind speed in grid points with missing information from both or one Lidar.
    Kd-tree is again used to do the regression and interpolation, this time neighbours are defined 
    in space and time, the latter assuming a constant wind speed and trajectory in successive scans.

    Input:
    -----
        U, V          - Lists of arrays representing the wind field in cartesian coordinates in each 
                        grid point of a triangulation represented by coordinates x and y. Each array
                        in the list represent  one scan.
                       
        x, y          - Arrays with cartesian coordinates of the un-structured grid.
        
        dt            - Time step between scans.
        
    Output:
    ------
        U_int,V_int   - List of arrays with interpolated wind speed field. Each array in the list
                        represent  one scan.
       
    """  
    # Initialization of the kdtree regressor. The number of neighbours, n_neighbours is set equal to
    # the number of corners and midpoints of a cube
    neigh = KNeighborsRegressor(n_neighbors=26,weights='distance',algorithm='auto', leaf_size=30,
                                n_jobs=1) 
    # Initialization of output
    U_int = []
    V_int = []
    it = range(len(U))
    # Loop over the list elements
    for scan in it:
        print(scan)
        # Initial and final scans are interpolated only in one direction in time, orwards and 
        # backwards, respectively.
        if scan == it[0]:
            # Temporal coordinate to spatial, assuming constant speed between scans.
            xj = x-U[scan+1]*dt
            yj = y-V[scan+1]*dt
            # Input for kd-regressor
            X = np.c_[np.r_[x,xj],np.r_[y,yj]]
            U_aux = U[scan].copy()
            V_aux = V[scan].copy()
            # Indexes of missing wind speeds in current and next scan
            ind0 = np.isnan(np.r_[U[scan],U[scan+1]])
            # Indexes in current scan
            ind1 = np.isnan(U[scan])
            # Check if there are any missing wind speed in the current scan and if there are enough
            # neighbours for interpolation.
            if (sum(ind1)>0) & (sum(~ind0)>26):
                # Regressor is defined for the actual scan
                neigh.fit(X[~ind0,:], np.r_[U[scan],U[scan+1]][~ind0])
                # Interpolation is carried out
                U_aux[ind1] = neigh.predict(np.c_[x,y][ind1,:])
                # Same for V component
                neigh.fit(X[~ind0,:], np.r_[V[scan],V[scan+1]][~ind0])
                V_aux[ind1] = neigh.predict(np.c_[x,y][ind1,:])
                U_int.append(U_aux)
                V_int.append(V_aux)
            # If there are not enough neighbours.    
            if ~(sum(~ind0)>26):
                U_int.append([])
                V_int.append([])                
            # If there is not missing data, go to the next scan   . 
            if ~(sum(ind1)>0):
                U_int.append(U_aux)
                V_int.append(V_aux)                
        # Same as before, this time when the last scan is reached.
        if scan == it[-1]:
            # Temporal coordinate to spatial, assuming constant speed between scans, this time going
            # backwards.
            xj = x+U[scan-1]*dt
            yj = y+V[scan-1]*dt 
            X = np.c_[np.r_[xj,x],np.r_[yj,y]]       
            U_aux = U[scan].copy()
            V_aux = V[scan].copy()
            ind0 = np.isnan(np.r_[U[scan-1],U[scan]])
            ind1 = np.isnan(U[scan])        
            if (sum(ind1)>0) & (sum(~ind0)>26):               
                neigh.fit(X[~ind0,:], np.r_[U[scan-1],U[scan]][~ind0])              
                U_aux[ind1] = neigh.predict(np.c_[x,y][ind1,:])                
                neigh.fit(X[~ind0,:], np.r_[V[scan-1],V[scan]][~ind0])               
                V_aux[ind1] = neigh.predict(np.c_[x,y][ind1,:])              
                U_int.append(U_aux)
                V_int.append(V_aux)           
            if ~(sum(~ind0)>26):
                U_int.append([])
                V_int.append([])                              
            if ~(sum(ind1)>0):
                U_int.append(U_aux)
                V_int.append(V_aux) 
        # All the rest, interpolated backwards and forwards in time.        
        else:
            # Temporal coordinate to spatial, forwards.
            xj = x+U[scan-1]*dt
            yj = y+V[scan-1]*dt
            # Temporal coordinate to spatial, backwards.
            xk = x-U[scan+1]*dt
            yk = y-V[scan+1]*dt
            X = np.c_[np.r_[xj,x,xk],np.r_[yj,y,yk]]
            U_aux = U[scan].copy()
            V_aux = V[scan].copy()
            ind0 = np.isnan(np.r_[U[scan-1],U[scan],U[scan+1]])
            ind1 = np.isnan(U[scan])
            if (sum(ind1)>0) & (sum(~ind0)>26):
                neigh.fit(X[~ind0,:], np.r_[U[scan-1],U[scan],U[scan+1]][~ind0])
                U_aux[ind1] = neigh.predict(np.c_[x,y][ind1,:])
                neigh.fit(X[~ind0,:], np.r_[V[scan-1],V[scan],V[scan+1]][~ind0])
                V_aux[ind1] = neigh.predict(np.c_[x,y][ind1,:])
                U_int.append(U_aux)
                V_int.append(V_aux)   
            if ~(sum(~ind0)>26):
                U_int.append([])
                V_int.append([])                  
            if ~(sum(ind1)>0):
                U_int.append(U_aux)
                V_int.append(V_aux) 
    return (U_int,V_int)

# In[Rapid reconstruction]

def dir_rec_rapid(V_a,V_b,a,b):
    Sa = np.sin(a)/np.sin(a-b)
    Sb = np.sin(b)/np.sin(a-b)
    Ca = np.cos(a)/np.sin(a-b)
    Cb = np.cos(b)/np.sin(a-b)
    U = -(Sb*V_a-Sa*V_b)
    V = -(-Cb*V_a+Ca*V_b)
    return (U,V)


# In[]
    
def direct_wf_rec(Lidar0, Lidar1, tri, d, angle = 'azim', r = 'range_gate',v_los = 'ws', scan = 'scan', N_grid = 512,interp = True):
    """
    Function to reconstruct horizontal wind field (2D) in Cartesian coordinates.
    
    Input:
    -----
        Lidar_i  - DataFrame with range_gate, azim/elev angle, line-of-sight wind speed, filtered
        
        tri      - Delaunay triangulation with the unstructured grid of laser beams intersection
                   points.
        
        d        - Linear distance between Lidar_i and Lidar_j.
        
    Output:
    ------
        U, V     - Cartesian components of wind speed.
    
    """
    # Grid creation
    scans0 = set(np.r_[np.unique(Lidar0[scan].values),np.unique(Lidar0[scan].values)])
    scans1 = set(np.r_[np.unique(Lidar1[scan].values),np.unique(Lidar1[scan].values)])
    #print(scans1)

    if scans0 < scans1:
        scans = scans0
    elif scans1 < scans0:
        scans = scans1
    else:
        scans = scans1
    U_out = []
    V_out = []
    scan_out = []
   
    phi_0 = np.pi-np.radians(Lidar0.loc[Lidar0[scan]==min(scans)][angle].unique())
    phi_1 = np.pi-np.radians(Lidar1.loc[Lidar1[scan]==min(scans)][angle].unique())

    r_0 = np.unique(Lidar0.loc[Lidar0[scan]==min(scans)][r].values)
    r_1 = np.unique(Lidar1.loc[Lidar1[scan]==min(scans)][r].values)

    r_g_0, phi_g_0 = np.meshgrid(r_0,phi_0)
    r_g_1, phi_g_1 = np.meshgrid(r_1,phi_1)
    
    
    r_t_0, phi_t_0 = translationpolargrid((r_g_0, phi_g_0),d/2)
    r_t_1, phi_t_1 = translationpolargrid((r_g_1, phi_g_1),-d/2)
    
#   Area lim
    area_lim = np.max(r_0)*np.max(np.diff(phi_0))*np.max(np.diff(r_0))

    x0 = r_t_0*np.cos(phi_t_0)
    y0 = r_t_0*np.sin(phi_t_0)  
    x1 = r_t_1*np.cos(phi_t_1)
    y1 = r_t_1*np.sin(phi_t_1)

    x = np.linspace(np.min(np.r_[x0.flatten(),x1.flatten()]),
                    np.max(np.r_[x0.flatten(),x1.flatten()]), N_grid)
    y = np.linspace(np.min(np.r_[y0.flatten(),y1.flatten()]),
                    np.max(np.r_[y0.flatten(),y1.flatten()]), N_grid)

    grd = np.meshgrid(x,y)
    
    # From Cartesian coord. to polar in global grid
    r_tri_s = np.sqrt(grd[0]**2 + grd[1]**2)
    phi_tri_s = np.arctan2(grd[1],grd[0])
    _, phi_tri_1_s = translationpolargrid((r_tri_s, phi_tri_s),-d/2)
    _, phi_tri_0_s = translationpolargrid((r_tri_s, phi_tri_s),d/2)
    
    # Mask of overlaping domain
    
    mask = np.reshape(tri.get_trifinder()(grd[0].flatten(),grd[1].flatten()),
                        grd[0].shape) == -1

    n, m = grd[0].shape
    
    
    x_i = grd[0][~mask]
    y_i = grd[1][~mask]
    
    v_sq_0 = x_i*np.nan
    v_sq_1 = y_i*np.nan
    
    r_i = np.sqrt(x_i**2 + y_i**2)
    phi_i = np.arctan2(y_i,x_i)

    r_i_0, phi_i_0 = translationpolargrid((r_i, phi_i),-d/2)
    r_i_1, phi_i_1 = translationpolargrid((r_i, phi_i), d/2)
      
    
    
    for i, scan_n in enumerate(scans):

        v_los_0 = Lidar0[v_los].loc[Lidar0[scan]==scan_n].values
        v_los_1 = Lidar1[v_los].loc[Lidar1[scan]==scan_n].values
        
        ind0 = ~np.isnan(v_los_0.flatten())
        ind1 = ~np.isnan(v_los_1.flatten())
        
        frac0 = np.sum(ind0)/len(v_los_0.flatten())
        frac1 = np.sum(ind1)/len(v_los_1.flatten())
        
        print(np.min([frac0,frac1]))
        
        
        if np.min([frac0,frac1])>.3:
            U = np.zeros((n,m))
            V = np.zeros((n,m))       
            U[mask] = np.nan
            V[mask] = np.nan
            scan_out.append(scan_n)
            x_0 = x0.flatten()[ind0]
            y_0 = y0.flatten()[ind0]
            x_1 = x1.flatten()[ind1]
            y_1 = y1.flatten()[ind1]
    #        
    #        plt.figure()
    #        plt.triplot(tri_t)
#            print(1)
#            tri_t0 = Triangulation(x_0,y_0)
#            maskt = TriAnalyzer(tri_t0).circle_ratios(rescale=False)<.1#get_flat_tri_mask(.05)
#            maska = areatriangles(tri_t0)> np.mean(areatriangles(tri_t0)) + 2*np.std(areatriangles(tri_t0))
#            mask0 = maskt | maska
#            tri_t0.set_mask(mask0)# = Triangulation(tri_t0.x,tri_t0.y,triangles=tri_t0.triangles[~mask0])
#            
#            if np.min([frac0,frac1])<.6:
#                plt.figure()
#                plt.triplot(tri_t0,color='red')
#            
#            pts_in_0 = tri_t0.get_trifinder()(x_i,y_i) !=-1
#            tri_t0.set_mask(None)
#            
#            print(2)
#            tri_t1 = Triangulation(x_1,y_1)
#            maskt=TriAnalyzer(tri_t1).circle_ratios(rescale=False)<.1#get_flat_tri_mask(.05) 
#            maska = areatriangles(tri_t1)> np.mean(areatriangles(tri_t1)) + 2*np.std(areatriangles(tri_t1))
#            mask1 = maskt | maska
#            tri_t1.set_mask(mask1)# = Triangulation(tri_t1.x,tri_t1.y,triangles=tri_t1.triangles[~mask1])
#            
#            if np.min([frac0,frac1])<.6:
#                plt.figure()
#                plt.triplot(tri_t1,color='red')            
#            
#            pts_in_1 = tri_t1.get_trifinder()(x_i,y_i) !=-1
#            tri_t1.set_mask(None)
                        
###################################################################################################################            
                        
            trid1 = Delaunay(np.c_[x_1,y_1])
            areas1 = areatriangles(trid1, delaunay = True)
            
            maskt = circleratios(trid1)<.05
            maska = areas1> area_lim
            mask1 = maskt | maska
            
            triangle_ind1 = np.arange(0,len(trid1.simplices))
            
            indtr1 = np.isin(trid1.find_simplex(np.c_[x_i,y_i]),triangle_ind1[~mask1])
            
            v_sq_1[indtr1] = sp.interpolate.griddata(np.c_[x_1,y_1],
                  v_los_1.flatten()[ind1], (x_i[indtr1],y_i[indtr1]), method='cubic')            
           
            trid0 = Delaunay(np.c_[x_0,y_0])
            areas0 = areatriangles(trid0, delaunay = True)
            
            maskt = circleratios(trid0)<.05
            maska = areas0> area_lim#np.mean(areas0) + np.std(areas0)
            mask0 = maskt | maska
            
            triangle_ind0 = np.arange(0,len(trid0.simplices))
            
            indtr0 = np.isin(trid0.find_simplex(np.c_[x_i,y_i]),triangle_ind0[~mask0])
            
            v_sq_0[indtr0] = sp.interpolate.griddata(np.c_[x_0,y_0],
                  v_los_0.flatten()[ind0], (x_i[indtr0],y_i[indtr0]), method='cubic')
            
            
           
###################################################################################################################            
    #        tri_t0 = Triangulation(tri_t.x,tri_t.y,triangles=tri_t.triangles[~maskt])
    #        plt.triplot(tri_t0,color='red')
    #        
    #        tri_t1 = Triangulation(tri_t.x,tri_t.y,triangles=tri_t.triangles[~(maskt | maska)])
              
    #        tri_t = Triangulation(x_1,y_1)
    #        plt.figure()
    #        plt.triplot(tri_t)
    #    
    #        maskt=TriAnalyzer(tri_t).circle_ratios(rescale=False)<.1#get_flat_tri_mask(.05)
    #        
    #        maska = areatriangles(tri_t)> np.mean(areatriangles(tri_t)) + 2*np.std(areatriangles(tri_t))
    #        
    #        tri_t0 = Triangulation(tri_t.x,tri_t.y,triangles=tri_t.triangles[~maskt])
    #        plt.triplot(tri_t0,color='red')
    #        
    #        tri_t1 = Triangulation(tri_t.x,tri_t.y,triangles=tri_t.triangles[~(maskt | maska)])
    #        plt.triplot(tri_t1,color='green')
    #        
    #        plt.figure()
    #        plt.plot(np.sort(areatriangles(tri_t)))
    #        plt.plot([1,len(np.sort(areatriangles(tri_t)))],[np.mean(areatriangles(tri_t)) + np.std(areatriangles(tri_t)),np.mean(areatriangles(tri_t)) + np.std(areatriangles(tri_t))])
    #        plt.plot([1,len(np.sort(areatriangles(tri_t)))],[np.mean(areatriangles(tri_t)) + 2*np.std(areatriangles(tri_t)),np.mean(areatriangles(tri_t)) + 2*np.std(areatriangles(tri_t))])
            #plt.triplot(tri_t1)
    #        plt.scatter(x_i[pts_in_1], y_i[pts_in_1])
            
    #        plt.figure()
    #        plt.plot(np.sort(areatriangles(tri_t1)))
    #        plt.plot([1,len(np.sort(areatriangles(tri_t1)))],[np.mean(areatriangles(tri_t1)) + np.std(areatriangles(tri_t1)),np.mean(areatriangles(tri_t1)) + np.std(areatriangles(tri_t1))])
    #        plt.plot([1,len(np.sort(areatriangles(tri_t1)))],[np.mean(areatriangles(tri_t1)) + 3*np.std(areatriangles(tri_t1)),np.mean(areatriangles(tri_t1)) + 3*np.std(areatriangles(tri_t1))])
    #        plt.yscale('log')
    #        plt.xscale('log')
            
            #plt.scatter( np.c_[x_1,y_1][tri_t1.triangles][:,:,0].flatten(),np.c_[x_1,y_1][tri_t1.triangles][:,:,1].flatten(),s=10,color='k')
            #print(CubicTriInterpolator(tri_t1, v_los_1.flatten()[ind1]))
#            print(3)
#            v_sq_1[pts_in_1] = CubicTriInterpolator(tri_t1, v_los_1.flatten()[ind1], kind='geom')(x_i[pts_in_1], y_i[pts_in_1]).data
#            v_sq_0[pts_in_0] = CubicTriInterpolator(tri_t0, v_los_0.flatten()[ind0], kind='geom')(x_i[pts_in_0], y_i[pts_in_0]).data
        
        
    #        v_sq_0 = sp.interpolate.griddata(np.c_[x_0,y_0],
    #                        v_los_0.flatten()[ind0], (x_i, y_i), method='cubic')
    #        
    #        v_sq_1 = sp.interpolate.griddata(np.c_[x_1,y_1],
    #                        v_los_1.flatten()[ind1], (x_i, y_i), method='cubic')
    
    #    
#            U[~mask] = v_sq_0
#            plt.figure()
#            plt.contourf(grd[0],grd[1],U[:,:],100,cmap='jet')
#            plt.colorbar()
    #        
    #        U[~mask,i] = v_sq_1
    #        plt.figure()
    #        plt.contourf(grd[0],grd[1],U[:,:,0],100,cmap='jet')
    #        plt.colorbar()
#            
#            vel_s = np.array([T_i(a,b,np.array([v_lv,v_ls])) for a,b,v_lv,v_ls in
#                              zip(phi_i_0,phi_i_1,v_sq_0,v_sq_1)])
    
    
            u,v = dir_rec_rapid(v_sq_1.flatten(),v_sq_0.flatten(), phi_i_1.flatten(),phi_i_0.flatten())
                
            #print(vel_s[:,0].shape,U[~mask,i].shape,v_sq_0.shape)
            U[~mask] = u#vel_s[:,0]
            
#            plt.figure()
#            plt.contourf(grd[0],grd[1],U,100,cmap='jet')
#            plt.colorbar()
            
            V[~mask] = v#vel_s[:,1]      
            U_out.append(U)
            V_out.append(V)
   
    return (U_out,V_out,grd,scan_out)

# In[]

def areatriangles(tri, delaunay = False):
    """ integrate scattered data, given a triangulation
    zsum, areasum = sumtriangles( xy, z, triangles )
    In:
        xy: npt, dim data points in 2d, 3d ...
        z: npt data values at the points, scalars or vectors
        tri: ntri, dim+1 indices of triangles or simplexes, as from
                   http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html
    Out:
        zsum: sum over all triangles of (area * z at midpoint).
            Thus z at a point where 5 triangles meet
            enters the sum 5 times, each weighted by that triangle's area / 3.
        area: the area or volume of the convex hull of the data points.
            For points over the unit square, zsum outside the hull is 0,
            so zsum / areasum would compensate for that.
            Or, make sure that the corners of the square or cube are in xy.
    """
    # z concave or convex => under or overestimates
    #ind = ~np.isnan(z)
    if delaunay:
        xy = tri.points
        triangles = tri.simplices
    else:
        xy = np.c_[tri.x,tri.y]
        triangles = tri.triangles
        
    npt, dim = xy.shape
    ntri, dim1 = triangles.shape
    #assert npt == len(z), "shape mismatch: xy %s z %s" % (xy.shape, z.shape)
    #assert dim1 == dim+1, "triangles ? %s" % triangles.shape
    area = np.zeros( ntri )
    #areasum = 0
    dimfac = np.prod( np.arange( 1, dim+1 ))
    for i, tri in enumerate(triangles):
        corners = xy[tri]
        t = corners[1:] - corners[0]
        if dim == 2:
            area[i] = abs( t[0,0] * t[1,1] - t[0,1] * t[1,0] ) / 2
        else:
            area[i] = abs( np.linalg.det( t )) / dimfac  # v slow
        #aux = area * np.nanmean(z[tri],axis=0)
        #if ~np.isnan(aux):
        #    zsum += aux
        #    areasum += area
    return area

# In[]
    
def circleratios(tri):
        """
        Returns a measure of the triangulation triangles flatness.

        The ratio of the incircle radius over the circumcircle radius is a
        widely used indicator of a triangle flatness.
        It is always ``<= 0.5`` and ``== 0.5`` only for equilateral
        triangles. Circle ratios below 0.01 denote very flat triangles.

        To avoid unduly low values due to a difference of scale between the 2
        axis, the triangular mesh can first be rescaled to fit inside a unit
        square with :attr:`scale_factors` (Only if *rescale* is True, which is
        its default value).

        Parameters
        ----------
        rescale : boolean, optional
            If True, a rescaling will be internally performed (based on
            :attr:`scale_factors`, so that the (unmasked) triangles fit
            exactly inside a unit square mesh. Default is True.

        Returns
        -------
        circle_ratios : masked array
            Ratio of the incircle radius over the
            circumcircle radius, for each 'rescaled' triangle of the
            encapsulated triangulation.
            Values corresponding to masked triangles are masked out.

        """
        # Coords rescaling
#        if rescale:
#            (kx, ky) = self.scale_factors
#        else:
        #(kx, ky) = (1.0, 1.0)
#        pts = np.vstack([self._triangulation.x*kx,
#                         self._triangulation.y*ky]).T
        
        pts = tri.points
        tri_pts = pts[tri.simplices.copy()]
        # Computes the 3 side lengths
        a = tri_pts[:, 1, :] - tri_pts[:, 0, :]
        b = tri_pts[:, 2, :] - tri_pts[:, 1, :]
        c = tri_pts[:, 0, :] - tri_pts[:, 2, :]
        a = np.sqrt(a[:, 0]**2 + a[:, 1]**2)
        b = np.sqrt(b[:, 0]**2 + b[:, 1]**2)
        c = np.sqrt(c[:, 0]**2 + c[:, 1]**2)
        # circumcircle and incircle radii
        s = (a+b+c)*0.5
        prod = s*(a+b-s)*(a+c-s)*(b+c-s)
        # We have to deal with flat triangles with infinite circum_radius
        bool_flat = (prod == 0.)
        if np.any(bool_flat):
            # Pathologic flow
            ntri = tri_pts.shape[0]
            circum_radius = np.empty(ntri, dtype=np.float64)
            circum_radius[bool_flat] = np.inf
            abc = a*b*c
            circum_radius[~bool_flat] = abc[~bool_flat] / (
                4.0*np.sqrt(prod[~bool_flat]))
        else:
            # Normal optimized flow
            circum_radius = (a*b*c) / (4.0*np.sqrt(prod))
        in_radius = (a*b*c) / (4.0*circum_radius*s)
        circle_ratio = in_radius/circum_radius
        #mask = self._triangulation.mask
        #if mask is None:
        return circle_ratio
        #else:
        #    return np.ma.array(circle_ratio, mask=mask)


# In[]
        
# In[]
    
#def direct_wf_rec(Lidar0, Lidar1, tri, d, angle = 'azim', r = 'range_gate',v_los = 'ws', scan = 'scan', N_grid = 512,interp = True):
#    """
#    Function to reconstruct horizontal wind field (2D) in Cartesian coordinates.
#    
#    Input:
#    -----
#        Lidar_i  - DataFrame with range_gate, azim/elev angle, line-of-sight wind speed, filtered
#        
#        tri      - Delaunay triangulation with the unstructured grid of laser beams intersection
#                   points.
#        
#        d        - Linear distance between Lidar_i and Lidar_j.
#        
#    Output:
#    ------
#        U, V     - Cartesian components of wind speed.
#    
#    """
#    # Grid creation
#    scans0 = set(np.r_[np.unique(Lidar0[scan].values),np.unique(Lidar0[scan].values)])
#    scans1 = set(np.r_[np.unique(Lidar1[scan].values),np.unique(Lidar1[scan].values)])
#    #print(scans1)
#
#    if scans0 < scans1:
#        scans = scans0
#    elif scans1 < scans0:
#        scans = scans1
#    else:
#        scans = scans1
#    U_out = []
#    V_out = []
#    scan_out = []
#   
##    p0 = df_0.azim.unique()
##    r0 = np.array(df_0.iloc[(df_0.azim==
##                                   min(p0)).nonzero()[0][0]].range_gate)
##    
##    r_g0, p_g0 = np.meshgrid(r0, np.pi-np.radians(p0)) # meshgrid
##    
##    r_g_t0, p_g_t0 = translationpolargrid((r_g0, p_g0), d/2)     
#        
#
#    
##    phi_0 = np.pi-np.radians(np.unique(Lidar0.loc[Lidar0[scan]==min(scans)][angle].values))
##    phi_1 = np.pi-np.radians(np.unique(Lidar1.loc[Lidar1[scan]==min(scans)][angle].values))
#    
#    phi_0 = np.pi-np.radians(Lidar0.loc[Lidar0[scan]==min(scans)][angle].unique())
#    phi_1 = np.pi-np.radians(Lidar1.loc[Lidar1[scan]==min(scans)][angle].unique())
#
#    r_0 = np.unique(Lidar0.loc[Lidar0[scan]==min(scans)][r].values)
#    r_1 = np.unique(Lidar1.loc[Lidar1[scan]==min(scans)][r].values)
#
#    r_g_0, phi_g_0 = np.meshgrid(r_0,phi_0)
#    r_g_1, phi_g_1 = np.meshgrid(r_1,phi_1)
#    
#    
#    r_t_0, phi_t_0 = translationpolargrid((r_g_0, phi_g_0),d/2)
#    r_t_1, phi_t_1 = translationpolargrid((r_g_1, phi_g_1),-d/2)
#    
##   Area lim
#    area_lim = np.max(r_0)*np.max(np.diff(phi_0))*np.max(np.diff(r_0))
#        
##    fig, ax = plt.subplots()
##    ax.set_aspect('equal')
##    ax.use_sticky_edges = False
##    ax.margins(0.07)
##    ax.triplot(tri,lw=2,color='grey',alpha=0.5)
##    im=ax.contourf(r_t_0*np.cos(phi_t_0),r_t_0*np.sin(phi_t_0),Lidar0.ws.loc[Lidar0.scan==1100].values,100,cmap='jet')
##    fig.colorbar(im)
##
##    fig, ax = plt.subplots()
##    ax.set_aspect('equal')
##    ax.use_sticky_edges = False
##    ax.margins(0.07)
##    ax.triplot(tri,lw=2,color='grey',alpha=0.5)
##    im=ax.contourf(r_t_1*np.cos(phi_t_1),r_t_1*np.sin(phi_t_1),Lidar1.ws.loc[Lidar1.scan==1100].values,100,cmap='jet')
##    fig.colorbar(im)
##    
##    
##    fig, ax = plt.subplots()
##    ax.set_aspect('equal')
##    ax.use_sticky_edges = False
##    ax.margins(0.07)
##    ax.triplot(tri,lw=2,color='grey',alpha=0.5)
##    im=ax.contourf(r_t_0*np.cos(phi_t_0),r_t_0*np.sin(phi_t_0),phi_g_0*180/np.pi,100,cmap='jet')
##    fig.colorbar(im)
##
##
##    fig, ax = plt.subplots()
##    ax.set_aspect('equal')
##    ax.use_sticky_edges = False
##    ax.margins(0.07)
##    ax.triplot(tri,lw=2,color='grey',alpha=0.5)
##    im=ax.contourf(r_t_1*np.cos(phi_t_1),r_t_1*np.sin(phi_t_1),phi_g_1*180/np.pi,100,cmap='jet')
##    fig.colorbar(im)
#
#    x0 = r_t_0*np.cos(phi_t_0)
#    y0 = r_t_0*np.sin(phi_t_0)  
#    x1 = r_t_1*np.cos(phi_t_1)
#    y1 = r_t_1*np.sin(phi_t_1)
#
#    x = np.linspace(np.min(np.r_[x0.flatten(),x1.flatten()]),
#                    np.max(np.r_[x0.flatten(),x1.flatten()]), N_grid)
#    y = np.linspace(np.min(np.r_[y0.flatten(),y1.flatten()]),
#                    np.max(np.r_[y0.flatten(),y1.flatten()]), N_grid)
#
#    grd = np.meshgrid(x,y)
#    
#    # From Cartesian coord. to polar in global grid
#    r_tri_s = np.sqrt(grd[0]**2 + grd[1]**2)
#    phi_tri_s = np.arctan2(grd[1],grd[0])
#    _, phi_tri_1_s = translationpolargrid((r_tri_s, phi_tri_s),-d/2)
#    _, phi_tri_0_s = translationpolargrid((r_tri_s, phi_tri_s),d/2)
#    
#    # Mask of overlaping domain
#    
#    
#
#    mask = np.reshape(tri.get_trifinder()(grd[0].flatten(),grd[1].flatten()),
#                        grd[0].shape) == -1
#
#    # Reconstruction function
#
##    T_i = lambda a_i,b_i,v_los: np.linalg.solve(np.array([[np.cos(a_i),
##                                np.sin(a_i)],[np.cos(b_i),np.sin(b_i)]]),v_los)
#
#    n, m = grd[0].shape
#    
#    print(n,m,len(scans),n*m*len(scans))
#    
#    
#    x_i = grd[0][~mask]
#    y_i = grd[1][~mask]
#    
#    v_sq_0 = x_i*np.nan
#    v_sq_1 = y_i*np.nan
#    
#    r_i = np.sqrt(x_i**2 + y_i**2)
#    phi_i = np.arctan2(y_i,x_i)
#
#    r_i_0, phi_i_0 = translationpolargrid((r_i, phi_i),-d/2)
#    r_i_1, phi_i_1 = translationpolargrid((r_i, phi_i), d/2)
#      
#    
#    
#    for i, scan_n in enumerate(scans):
#        print(i,scan_n)
#        v_los_0 = Lidar0[v_los].loc[Lidar0[scan]==scan_n].values
#        v_los_1 = Lidar1[v_los].loc[Lidar1[scan]==scan_n].values
#        
#        ind0 = ~np.isnan(v_los_0.flatten())
#        ind1 = ~np.isnan(v_los_1.flatten())
#        
#        frac0 = np.sum(ind0)/len(v_los_0.flatten())
#        frac1 = np.sum(ind1)/len(v_los_1.flatten())
#        
#        #print(np.min([frac0,frac1]))
#        
#        
#        if np.min([frac0,frac1])>.3:
#            U = np.zeros((n,m))
#            V = np.zeros((n,m))       
#            U[mask] = np.nan
#            V[mask] = np.nan
#            scan_out.append(scan_n)
#            x_0 = x0.flatten()[ind0]
#            y_0 = y0.flatten()[ind0]
#            x_1 = x1.flatten()[ind1]
#            y_1 = y1.flatten()[ind1]
#    #        
#    #        plt.figure()
#    #        plt.triplot(tri_t)
##            print(1)
##            tri_t0 = Triangulation(x_0,y_0)
##            maskt = TriAnalyzer(tri_t0).circle_ratios(rescale=False)<.1#get_flat_tri_mask(.05)
##            maska = areatriangles(tri_t0)> np.mean(areatriangles(tri_t0)) + 2*np.std(areatriangles(tri_t0))
##            mask0 = maskt | maska
##            tri_t0.set_mask(mask0)# = Triangulation(tri_t0.x,tri_t0.y,triangles=tri_t0.triangles[~mask0])
##            
##            if np.min([frac0,frac1])<.6:
##                plt.figure()
##                plt.triplot(tri_t0,color='red')
##            
##            pts_in_0 = tri_t0.get_trifinder()(x_i,y_i) !=-1
##            tri_t0.set_mask(None)
##            
##            print(2)
##            tri_t1 = Triangulation(x_1,y_1)
##            maskt=TriAnalyzer(tri_t1).circle_ratios(rescale=False)<.1#get_flat_tri_mask(.05) 
##            maska = areatriangles(tri_t1)> np.mean(areatriangles(tri_t1)) + 2*np.std(areatriangles(tri_t1))
##            mask1 = maskt | maska
##            tri_t1.set_mask(mask1)# = Triangulation(tri_t1.x,tri_t1.y,triangles=tri_t1.triangles[~mask1])
##            
##            if np.min([frac0,frac1])<.6:
##                plt.figure()
##                plt.triplot(tri_t1,color='red')            
##            
##            pts_in_1 = tri_t1.get_trifinder()(x_i,y_i) !=-1
##            tri_t1.set_mask(None)
#                        
####################################################################################################################            
#                        
#            trid1 = Delaunay(np.c_[x_1,y_1])
#            areas1 = areatriangles(trid1, delaunay = True)
#            
#            maskt = circleratios(trid1)<.05
#            maska = areas1> area_lim
#            mask1 = maskt | maska
#            
#            triangle_ind1 = np.arange(0,len(trid1.simplices))
#            
#            indtr1 = np.isin(trid1.find_simplex(np.c_[x_i,y_i]),triangle_ind1[~mask1])
#            
#            v_sq_1[indtr1] = sp.interpolate.griddata(np.c_[x_1,y_1],
#                  v_los_1.flatten()[ind1], (x_i[indtr1],y_i[indtr1]), method='cubic')            
#           
#            trid0 = Delaunay(np.c_[x_0,y_0])
#            areas0 = areatriangles(trid0, delaunay = True)
#            
#            maskt = circleratios(trid0)<.05
#            maska = areas0> area_lim#np.mean(areas0) + np.std(areas0)
#            mask0 = maskt | maska
#            
#            triangle_ind0 = np.arange(0,len(trid0.simplices))
#            
#            indtr0 = np.isin(trid0.find_simplex(np.c_[x_i,y_i]),triangle_ind0[~mask0])
#            
#            v_sq_0[indtr0] = sp.interpolate.griddata(np.c_[x_0,y_0],
#                  v_los_0.flatten()[ind0], (x_i[indtr0],y_i[indtr0]), method='cubic')
#            
#            
#           
####################################################################################################################            
#    #        tri_t0 = Triangulation(tri_t.x,tri_t.y,triangles=tri_t.triangles[~maskt])
#    #        plt.triplot(tri_t0,color='red')
#    #        
#    #        tri_t1 = Triangulation(tri_t.x,tri_t.y,triangles=tri_t.triangles[~(maskt | maska)])
#              
#    #        tri_t = Triangulation(x_1,y_1)
#    #        plt.figure()
#    #        plt.triplot(tri_t)
#    #    
#    #        maskt=TriAnalyzer(tri_t).circle_ratios(rescale=False)<.1#get_flat_tri_mask(.05)
#    #        
#    #        maska = areatriangles(tri_t)> np.mean(areatriangles(tri_t)) + 2*np.std(areatriangles(tri_t))
#    #        
#    #        tri_t0 = Triangulation(tri_t.x,tri_t.y,triangles=tri_t.triangles[~maskt])
#    #        plt.triplot(tri_t0,color='red')
#    #        
#    #        tri_t1 = Triangulation(tri_t.x,tri_t.y,triangles=tri_t.triangles[~(maskt | maska)])
#    #        plt.triplot(tri_t1,color='green')
#    #        
#    #        plt.figure()
#    #        plt.plot(np.sort(areatriangles(tri_t)))
#    #        plt.plot([1,len(np.sort(areatriangles(tri_t)))],[np.mean(areatriangles(tri_t)) + np.std(areatriangles(tri_t)),np.mean(areatriangles(tri_t)) + np.std(areatriangles(tri_t))])
#    #        plt.plot([1,len(np.sort(areatriangles(tri_t)))],[np.mean(areatriangles(tri_t)) + 2*np.std(areatriangles(tri_t)),np.mean(areatriangles(tri_t)) + 2*np.std(areatriangles(tri_t))])
#            #plt.triplot(tri_t1)
#    #        plt.scatter(x_i[pts_in_1], y_i[pts_in_1])
#            
#    #        plt.figure()
#    #        plt.plot(np.sort(areatriangles(tri_t1)))
#    #        plt.plot([1,len(np.sort(areatriangles(tri_t1)))],[np.mean(areatriangles(tri_t1)) + np.std(areatriangles(tri_t1)),np.mean(areatriangles(tri_t1)) + np.std(areatriangles(tri_t1))])
#    #        plt.plot([1,len(np.sort(areatriangles(tri_t1)))],[np.mean(areatriangles(tri_t1)) + 3*np.std(areatriangles(tri_t1)),np.mean(areatriangles(tri_t1)) + 3*np.std(areatriangles(tri_t1))])
#    #        plt.yscale('log')
#    #        plt.xscale('log')
#            
#            #plt.scatter( np.c_[x_1,y_1][tri_t1.triangles][:,:,0].flatten(),np.c_[x_1,y_1][tri_t1.triangles][:,:,1].flatten(),s=10,color='k')
#            #print(CubicTriInterpolator(tri_t1, v_los_1.flatten()[ind1]))
##            print(3)
##            v_sq_1[pts_in_1] = CubicTriInterpolator(tri_t1, v_los_1.flatten()[ind1], kind='geom')(x_i[pts_in_1], y_i[pts_in_1]).data
##            v_sq_0[pts_in_0] = CubicTriInterpolator(tri_t0, v_los_0.flatten()[ind0], kind='geom')(x_i[pts_in_0], y_i[pts_in_0]).data
#        
#        
#    #        v_sq_0 = sp.interpolate.griddata(np.c_[x_0,y_0],
#    #                        v_los_0.flatten()[ind0], (x_i, y_i), method='cubic')
#    #        
#    #        v_sq_1 = sp.interpolate.griddata(np.c_[x_1,y_1],
#    #                        v_los_1.flatten()[ind1], (x_i, y_i), method='cubic')
#    
#    #    
##            U[~mask] = v_sq_0
##            plt.figure()
##            plt.contourf(grd[0],grd[1],U[:,:],100,cmap='jet')
##            plt.colorbar()
#    #        
#    #        U[~mask,i] = v_sq_1
#    #        plt.figure()
#    #        plt.contourf(grd[0],grd[1],U[:,:,0],100,cmap='jet')
#    #        plt.colorbar()
##            
##            vel_s = np.array([T_i(a,b,np.array([v_lv,v_ls])) for a,b,v_lv,v_ls in
##                              zip(phi_i_0,phi_i_1,v_sq_0,v_sq_1)])
#    
#    
#            u,v = dir_rec_rapid(v_sq_1.flatten(),v_sq_0.flatten(), phi_i_1.flatten(),phi_i_0.flatten())
#                
#            #print(vel_s[:,0].shape,U[~mask,i].shape,v_sq_0.shape)
#            U[~mask] = u#vel_s[:,0]
#            
##            plt.figure()
##            plt.contourf(grd[0],grd[1],U,100,cmap='jet')
##            plt.colorbar()
#            
#            V[~mask] = v#vel_s[:,1]      
#            U_out.append(U)
#            V_out.append(V)
#   
#    return (U_out,V_out,grd,scan_out)