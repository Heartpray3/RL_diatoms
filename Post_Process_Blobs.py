# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from __future__ import division, print_function
import os
import numpy as np
import matplotlib.pylab as plt
import cv2
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from utils import load_config, abs_path

#%% Input 

a = 0.183228708092682 # blob radius

plot_blobs_all_step = 0

save_movie_blobs = 1
save_movie_without_blobs = 0

config = load_config()
Nblobs_vec = [config.Nblobs]
Nrods_vec = [config.Nrods] #Do not vary with a np.arrange
phase_shift_vec = [config.phase_shift]


root_name = 'bacillaria_'
# Path = '/home/ely/Documents/internship/RigidMultiblobsWall-master-JLD/multi_bodies/examples/Optim/'
Path = abs_path(config.output_directory)

Nstep = 80
dt = 0.0025

#%% Function 

def quaternion_rotation_matrix(q0,q1,q2,q3):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    q0,q1,q2,q3
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
    return rot_matrix

#%% Code

Nr = len(Nrods_vec)    
Nps = len(phase_shift_vec)
Nb = len(Nblobs_vec)

L = Nrods_vec[0]*2*a

#####Vertex file#####

os.chdir(Path)
         
for nb in range(Nb):
    Nblobs = Nblobs_vec[nb]
    suffix_Nrods = str(Nblobs) +  '_blobs_'
    filename1 = root_name + suffix_Nrods
    
    f  = open(root_name + str(Nblobs) + '_blobs.vertex')
    data1 = f.read()
    data1 = data1.split()
    
    vertex = []
    blobs = []
    for i in range(len(data1)):
        if data1[i]!=data1[0]:
            vertex.append(data1[i])
            
    blobs.append(data1[0])
    blobs = np.asarray(blobs)
    blobs = blobs.astype(float)
    blobs = int(blobs[0])
    
    vertex = np.asarray(vertex)
    vertex = vertex.astype(float)
    vertex = np.reshape(vertex,(-1,3))
    
    
    N_phase_shift_vec = np.size(phase_shift_vec)
    position_com_all_ps = np.zeros((N_phase_shift_vec,Nstep,3))
    ind1 = 0

# ######## Vitesse #########

    velocity = np.zeros((Nb,Nps,40))
    Nrods = Nrods_vec[0]
    
    for h in range(Nps):
        phase_shift = phase_shift_vec[h]
        psint,psfloat = str(phase_shift).split(".")
        psint = psint[0]
        psfloat = psfloat[0]
        os.chdir(Path+str(filename1)+str(Nrods)+'_rods_phase_shift_'+str(psint)+'_'+str(psfloat)+'pi')
        
        for i in range(40,80,1):
            
            f  = open(str(filename1)+str(Nrods)+'_rods_phase_shift_'+str(psint)+'_'+str(psfloat)+'pi.Velocity_COM_step_'+str(i),'r')
            data = f.read()
            data = data.split()
            data = np.asarray(data)
            data = data.astype(float)
            
            data[0] = data[0] * 40 * dt / L 
            data[2] = data[2] * 40 * dt / L 

            velocity[nb,h,i-40] = np.sqrt(data[0]**2+data[2]**2)
            

#####Boucle pour plusieurs cas#####

    for g in range(Nr):
        Nrods = Nrods_vec[g]
        L = Nrods*2*a
        N_lambda_vec= []
        for h in range(Nps):
            phase_shift = phase_shift_vec[h]
            N_lambda = phase_shift/2
            N_lambda_vec.append(N_lambda)
            
            #####Config file#####  
                    
            psint,psfloat = str(phase_shift).split(".")
            psint = psint[0]
            psfloat = psfloat[0]
            os.chdir(Path+str(filename1)+str(Nrods)+'_rods_phase_shift_'+str(psint)+'_'+str(psfloat)+'pi')
            f  = open('run_'+str(filename1)+str(Nrods)+'_rods_phase_shift_'+str(psint)+'_'+str(psfloat)+'pi.'+str(filename1)+str(Nrods)+'_rods.config','r')
            data2 = f.read()
            data2 = data2.split()
                    
            config = []
            for i in range(len(data2)):
                if data2[i]!=str(Nrods):
                    config.append(data2[i])
            config = np.asarray(config)
            config = config.astype(float)
            config = np.reshape(config,(-1,7))
                    
            position_com = np.zeros((Nstep,3))
            pos_all_blobs = np.zeros((blobs*Nrods,Nstep,3))
            
            for i in range(Nstep): #loop over time
                    
                #####Center of mass##### 
                        
                config1 = 0
                config2 = 0
                config3 = 0  
                        
                blobs_act = 0 
                
                for j in range(i*Nrods,(i+1)*Nrods): 
                    config1 += config[j,0]
                    config2 += config[j,1]
                    config3 += config[j,2]
                             
                config1 = config1 / Nrods
                config2 = config2 / Nrods
                config3 = config3 / Nrods               
                        
                position_com[i,0] = config1
                position_com[i,1] = config2
                position_com[i,2] = config3
                
                position_com_all_ps[ind1,:,:] = position_com[:,:]
            
                ######Blob position##### 
            
                for j in range(i*Nrods,(i+1)*Nrods):
                    
                    #Rotation matrix for each blobs
                    R1 = quaternion_rotation_matrix(config[j,3],config[j,4],config[j,5],config[j,6]) 
                    U1 = np.array((config[j,0],config[j,1],config[j,2]))
                    
                    # Position of Nblob on N
                    b1_pos = np.repeat(U1,1) + (R1 @ vertex.T).T
                    
                    for k in range(blobs):
                    
                        pos_all_blobs[blobs_act,i,:] = b1_pos[k,:] 
                        blobs_act += 1
            
                                
    #%% Post-Processing
    
            xmin = np.min(pos_all_blobs[:,:,0])
            xmax = np.max(pos_all_blobs[:,:,0])
            zmin = np.min(pos_all_blobs[:,:,2])
            zmax = np.max(pos_all_blobs[:,:,2])
            
            maxi = max(abs(xmin),xmax,abs(zmin),zmax)
                
            xmin = -maxi
            xmax = maxi
            zmin = -maxi
            zmax = maxi
            
            theta = np.linspace( 0 , 2 * np.pi , 100)
            
            if plot_blobs_all_step == 1 : 
                for i in range(Nstep): #loop over time
            
                    plt.figure('N = ' + str(Nrods_vec[g]) + ', phi = ' + str(N_lambda_vec[h]) + ' et Step = ' + str(i),figsize=(12.8,9.6))
                    plt.xlabel('x/L')
                    plt.ylabel('z/L')
                    plt.xlim(((xmin-3*a)/L,(xmax+3*a)/L))
                    plt.ylim(((zmin-3*a)/L,(zmax+3*a)/L))
                    for j in range(blobs*Nrods): 
                        x = a * np.cos(theta)/L + pos_all_blobs[j,i,0]/L
                        z = a * np.sin(theta)/L + pos_all_blobs[j,i,2]/L
                        plt.plot(x,z,'#21918c')
                        plt.fill_between(x,z,facecolor='#21918c')        
                    plt.plot(position_com[0:i,0]/L,position_com[0:i,2]/L,'r') 
                    plt.show()
                    plt.pause(0.25)
                    plt.close()
                
        if save_movie_blobs == 1 :
            
            norm = plt.Normalize(velocity[g,h,:].min(), velocity[g,h,:].max())
            
            for i in range(Nstep): #loop over time
                plt.figure('Test pour N = ' + str(Nrods_vec[g]) + ' et N_lambda = ' + str(N_lambda_vec[h]) + ' Step = ' + str(i),figsize=(12.8,9.6))
                plt.xlabel('x/L')
                plt.ylabel('z/L')
                plt.xlim(((xmin-3*a)/L,(xmax+3*a)/L))
                plt.ylim(((zmin-3*a)/L,(zmax+3*a)/L))
                
                for j in range(blobs*Nrods): 
                    x = a * np.cos(theta)/L + pos_all_blobs[j,i,0]/L
                    z = a * np.sin(theta)/L + pos_all_blobs[j,i,2]/L
                    plt.plot(x,z,'#d2aa0f',zorder=-2)
                    plt.fill_between(x,z,facecolor='#d2aa0f')              
                points = np.array([position_com[0:i,0]/L, position_com[0:i,2]/L]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                
                # Create a continuous norm to map from data points to colors
                # lc = LineCollection(segments, cmap='plasma', norm=norm)
                # lc.set_array(velocity[g,h,:])
                # lc.set_linewidth(2)
                # plt.gca().add_collection(lc)
                # plt.colorbar(cm.ScalarMappable(cmap = 'plasma', norm=norm))
                
                plt.savefig('N = ' + str(Nrods_vec[g]) + ' et N_lambda = ' + str(N_lambda_vec[h]) + ' Step = ' + str(i) +'.pdf')
                plt.close()
            
        if save_movie_without_blobs == 1 :
            
            xmin2 = np.min(position_com[:,0])
            xmax2 = np.max(position_com[:,0])
            zmin2 = np.min(position_com[:,2])
            zmax2 = np.max(position_com[:,2])
            
            maxi = max(abs(xmin2),xmax2,abs(zmin2),zmax2)
            
            xmin2 = -maxi
            xmax2 = maxi
            zmin2 = -maxi
            zmax2 = maxi
            
            norm = plt.Normalize(velocity[g,h,:].min(), velocity[g,h,:].max())
            
            for i in range(Nstep): #loop over time
                plt.figure('N = ' + str(Nrods_vec[g]) + ' et N_lambda = ' + str(N_lambda_vec[h]) + ' Step = ' + str(i),figsize=(12.8,9.6))
                plt.xlabel('x/L')
                plt.ylabel('z/L')
                plt.xlim(((xmin2-3*a)/L,(xmax2+3*a)/L))
                plt.ylim(((zmin2-3*a)/L,(zmax2+3*a)/L))
                points = np.array([position_com[0:i,0]/L, position_com[0:i,2]/L]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                # Create a continuous norm to map from data points to colors
                lc = LineCollection(segments, cmap='plasma', norm=norm)
                lc.set_array(velocity[g,h,:])
                lc.set_linewidth(2)
                plt.gca().add_collection(lc)
                plt.colorbar(cm.ScalarMappable(cmap = 'plasma', norm=norm))
                plt.savefig('Trajectory COM for N = ' + str(Nrods_vec[g]) + ' et N_lambda = ' + str(N_lambda_vec[h]) + ' Step = ' + str(i) +'.jpg',format='jpg')
                plt.close()
   

        
        
        
