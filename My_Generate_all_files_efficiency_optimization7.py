#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 16:00:01 2025

@author: stefani

Copie du code de Julien
Adapté de la v6

v7 - an and bn coefficients are given as inputs as strings separated by /.

"""

from __future__ import division, print_function
import os
import numpy as np
import math
import shutil
import argparse
import shutil
from utils import get_sim_folder

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


def update_dat(input_file, output_file, updates):
    lines = open(input_file).readlines()
    max_len = max((len(l.split()[0]) for l in lines if l.strip() and not l.strip().startswith('#')), default=0)

    with open(output_file, 'w') as f:
        for line in lines:
            if not line.strip() or line.lstrip().startswith('#'):
                f.write(line)
                continue
            key, *vals = line.split()
            new_vals = updates.get(key, vals)
            f.write(f"{key.ljust(max_len + 1)}{' '.join(new_vals)}\n")


def main(output_directory, Nblobs, phase_shift, Nrods, an, bn, nmodes, dt, Nstep, freq):
    an = [float(i) for i in an.replace('*','-').split('/')[:nmodes]]
    bn = [float(i) for i in bn.replace('*','-').split('/')[:nmodes]]

    a = 0.183228708092682  
    
    # dt = 0.00125
    # Nstep = 160
    # freq = 10
    
    ##Generate vextex files 
    X_coef = 7.4209799e-02
    X_step = 2 * X_coef
    
    Nconst_per_rod = 2
    root_name = 'bacillaria_'
    
    
    
    #%% Code Generate_const_clone_list_vertex_file_and_execute_Blobs
    
    suffix_Nrods = str(Nblobs) +  '_blobs_'
    filename1 = root_name + suffix_Nrods
    L_rods = Nblobs * 0.81 * a  
    L = 0.5 * 0.9 * L_rods
    
    suffix_Nrods = str(Nrods) +  '_rods'
    filename = filename1 + suffix_Nrods
    
    # suffix_phase_shift = '_phase_shift_' +  str(phase_shift).replace('.', '_') + 'pi'

    # suffix_fourier = '_A_'
    # for i in range(nmodes):
    #     Aint, Afloat = str(an[i]).split('.')
    #     suffix_fourier += f'{Aint}_{Afloat}_'
    # suffix_fourier += 'B_'
    # for i in range(nmodes):
    #     Bint, Bfloat = str(bn[i]).split('.')
    #     suffix_fourier += f'{Bint}_{Bfloat}_'
    # suffix_fourier += f'n_{nmodes}'
    
    foldername = filename #+ suffix_phase_shift #+ suffix_fourier

    output_folder = get_sim_folder(foldername, Nrods, Nblobs)
    if os.path.exists(output_folder):
        print("Folder {} already exists!".format(output_folder))
        response = input("Do you want to restart the simulation ? (y/n) : ").strip().lower()
        if response == 'y':
            shutil.rmtree(output_folder)
            print("Simulation restarted.")
        else:
            print("Abort simulation.")
            return ()

    os.makedirs(output_folder)
    
    input_directory = '/home/ely/Documents/internship/RigidMultiblobsWall-master-JLD/multi_bodies/examples/RL_diatoms/'
    # input_directory = os.path.join(os.path.dirname(__file__), output_folder)
    ## Generate clones files 
    filename_clones = filename +  '.clones'
    
    pos = np.zeros((Nrods,3))
    quat = np.zeros((Nrods,4))
    
    for n in range(Nrods):
        pos[n,2] = (n-1)*2*a
        quat[n,0] = 1;
    mean_pos = np.mean(pos,axis=0)
    
    for n in range(Nrods):
        pos[n,:] = pos[n,:] - mean_pos
    
    to_save = np.concatenate((pos, quat),axis=1)
    
    fid= open(output_folder + '/' + filename_clones,'w')
    fid.write(str(Nrods)+ '\n')
    np.savetxt(fid,to_save)
    fid.close()
     
    ## Generate Vertex files 
    
    with open(os.path.join(output_directory, f"{Nblobs}_Blobs",  root_name + str(Nblobs) +  '_blobs.vertex'), "w") as f:
        f.write(str(Nblobs) + "\n")
        Start = -((Nblobs/2-1)*X_step + X_coef)
        for i in range(Nblobs):
            X = Start + i*X_step
            f.write("{:.7e} {:.7e} {:.7e}\n".format(X, 0, 0))
            
    ## Generate list vertex files 
    filename_vertex = root_name + str(Nblobs) +  '_blobs.vertex'
    
    filename_list_vertex = filename +  '.list_vertex'
    fid= open(output_folder + '/' + filename_list_vertex,'w')
    
    for n in range(Nrods):
        fid.write(os.path.join(output_directory, f"{Nblobs}_Blobs" , filename_vertex) + '\n')
    fid.close()
    
    ## Constraints
    Nconst = (Nrods-1)*Nconst_per_rod
    filename_const = filename +  '.const'
    fid= open(output_folder + '/' + filename_const, 'w')
    fid.write(str(Nrods) + '\n')
    fid.write(str(Nconst) + '\n')
    
    sixzeros = '0 0 0 0 0 0'
    for n in range(Nrods-1): 
        ps = phase_shift*n*math.pi/(Nrods-1)

        if Nconst_per_rod >= 1:
            pos1z = ' 0 '
            pos2z = ' 0 '
            pos1y = str(a) + ' '
            pos2y = str(-a) + ' '
            vel1z = ' 0 '
            vel2z = ' 0 '
            vel1y = ' 0 '
            vel2y = ' 0 '
 
            pos1x = ''
            pos2x = ''
            vel1x = '0'
            vel2x = '0'
		      
            for i in range(nmodes):
                pos1x += f'+{str(L*an[i])}*cos({str(2*math.pi*freq*(i+1))}*t+{str(ps)})+{str(L*bn[i])}*sin({str(2*math.pi*freq*(i+1))}*t+{str(ps)})'
                pos2x += f'+{str(-L*an[i])}*cos({str(2*math.pi*freq*(i+1))}*t+{str(ps)})+{str(-L*bn[i])}*sin({str(2*math.pi*freq*(i+1))}*t+{str(ps)})'
                vel1x += f'+{str(-L*an[i]*2*math.pi*freq*(i+1))}*sin({str(2*math.pi*freq*(i+1))}*t+{str(ps)})+{str(L*bn[i]*2*math.pi*freq*(i+1))}*cos({str(2*math.pi*freq*(i+1))}*t+{str(ps)})'
                vel2x += f'+{str(L*an[i]*2*math.pi*freq*(i+1))}*sin({str(2*math.pi*freq*(i+1))}*t+{str(ps)})+{str(-L*bn[i]*2*math.pi*freq*(i+1))}*cos({str(2*math.pi*freq*(i+1))}*t+{str(ps)})'
				  		
            c1 = str(n) + ' '\
                + str(n+1) + ' '\
                + sixzeros + ' '\
                + pos1x \
                + pos1z \
                + pos1y \
                + pos2x \
                + pos2z \
                + pos2y \
                + vel1x \
                + vel1z \
                + vel1y \
                + vel2x \
                + vel2z \
                + vel2y
            fid.write(c1 + '\n')
		
        if Nconst_per_rod >= 2:
            offset = 0.1
		    # Location of second is arbitrary as long as it does not coincide with link 1
            c2 = str(n) +  ' '\
			+ str(n+1) +  ' '\
			+ sixzeros + ' '\
			+ str(1+offset) +  '*(' + pos1x + ')' \
			+ pos1z \
			+ str((1+offset)) + '*' + pos1y \
			+ str((1-offset))+ '*(' + pos2x + ')'\
			+ pos2z \
			+ str((1-offset)) +  '*' + pos2y \
			+ str((1+offset)) + '*(' + vel1x + ')'\
			+ vel1z \
			+ vel1y \
			+ str((1-offset)) + '*(' + vel2x + ')'\
			+ vel2z \
			+ vel2y
            
            fid.write(c2 + '\n')
    
    fid.close()
    
    ## Modify input file accordingly
    # os.chdir(input_directory)
    filename_input = 'inputfile_bacillaria'
    input_path = os.path.join(input_directory, filename_input + '.dat')

    # Valeurs à mettre à jour
    updates = {
        'dt': [str(dt)],
        'n_steps': [str(Nstep)],
        'output_name': ['run_' + foldername],
        'articulated': [filename_list_vertex, filename_clones, filename_const]
    }

    # Chemin de sortie
    os.chdir(os.path.join(output_directory, output_folder))
    filename_input_local = f"{filename_input}_{suffix_Nrods}.dat"

    # Écriture du nouveau fichier .dat
    update_dat(input_path, filename_input_local, updates)
    
    my_command = f'python3 ../../../multi_bodies_bacillaria1.py --input-file {filename_input_local}' #+ ' --print-residual'
    os.system(my_command)
    # os.chdir('../')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Traite des fichiers avec optimisation d'efficacité.")
    parser.add_argument("--output_directory", type=str, required=True, help= "Fichier de sortie") 
    parser.add_argument("--Nblobs", type=int, required=True, help="Nombre de blobs")
    parser.add_argument("--phase_shift", type=float, required=True, help="Décalage de phase")
    parser.add_argument("--Nrods", type=int, required=True, help="Nombre de rods")
    parser.add_argument("--an", type=str, required=True, help = "coeffs an")
    parser.add_argument("--bn", type=str, required=True, help = "coeffs bn")
    parser.add_argument("--n", type=int, required=True, help = "number of Fourier modes in decomposition")
    parser.add_argument("--dt", type=float, default= 0.00125, help = "Timestep")
    parser.add_argument("--Nstep", type=int, default = 160, help = "Nombre d'étapes")
    parser.add_argument("--freq", type=float, default = 10, help = "fréquence du sinus dans le coulissement")
    
    args = parser.parse_args()
    
    main(args.output_directory, args.Nblobs, args.phase_shift, args.Nrods, args.an, args.bn, args.n, args.dt, args.Nstep, args.freq)
