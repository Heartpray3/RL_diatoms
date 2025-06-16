#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 9 13:00:01 2025

@author: Ely Cheikh Abass

La structure de base est inspire du code de Julien/Stefanie


Code d'apprentissage par renforcement / simulation
"""

from __future__ import division, print_function
import os
import numpy as np
import math
import argparse
import shutil
from utils import get_sim_folder, ColonyState, Action
from typing import List


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


def write_move_const(
    output_folder: str,
    file_name: str,
    Nrods: int,
    moving_rod: int,      # indice [0‥Nrods-1]
    direction: int,       # +1 ou -1
    a: float,             # rayon → déplacement cible = 2a
    dt: float,            # pas de temps
    Nconst_per_rod: int = 2,
    offset: float = 0.1
) -> str:
    """
    Écrit un fichier .const basé sur un déplacement linéaire de ±2a
    pour le bâtonnet `moving_rod`, pendant dt. Respecte le format de contraintes complet.
    """

    const_path = os.path.join(output_folder, file_name)
    with open(const_path, "w") as fid:
        fid.write(f"{Nrods}\n")
        fid.write(f"{(Nrods - 1) * Nconst_per_rod}\n")

        sixzeros = '0 0 0 0 0 0'

        for n in range(Nrods - 1):
            pos1z, pos2z = ' 0 ', ' 0 '
            pos1y, pos2y = f"{a} ", f"{-a} "
            vel1z = vel2z = vel1y = vel2y = ' 0 '

            pos1x = pos2x = '0'
            vel1x = vel2x = '0'
            expression = a / (2 * dt) * direction
            if n == moving_rod:
                pos1x = f"{expression}*t"
                pos2x = f"{-expression}*t"
                vel1x = f"{expression}"
                vel2x = f"{-expression}"
            elif n + 1 == moving_rod:
                pos1x = f"{-expression}*t"
                pos2x = f"{expression}*t"
                vel1x = f"{-expression}"
                vel2x = f"{expression}"

            c1 = (
                f"{n} {n + 1} {sixzeros} "
                f"{pos1x} {pos1z}{pos1y}"
                f"{pos2x} {pos2z}{pos2y}"
                f"{vel1x} {vel1z}{vel1y}"
                f"{vel2x} {vel2z}{vel2y}"
            )
            fid.write(c1 + '\n')

            if Nconst_per_rod >= 2:
                c2 = (
                    f"{n} {n + 1} {sixzeros} "
                    f"{1 + offset}*({pos1x}) {pos1z}{(1 + offset) * a} "
                    f"{1 - offset}*({pos2x}) {pos2z}{(1 - offset) * -a} "
                    f"{1 + offset}*({vel1x}) {vel1z}{vel1y} "
                    f"{1 - offset}*({vel2x}) {vel2z}{vel2y}"
                )
                fid.write(c2 + '\n')

    return const_path


class DiatomEnv:
    def __init__(self, n_rods: int, n_blobs: int, a: float, dt: float):
        self.n_rods = n_rods
        self.n_blobs = n_blobs
        self.a = a
        self.dt = dt
        self.state = None
        self.reset()

    def get_available_actions(self, state: ColonyState):
        available_actions: List[Action] = []
        for n_gap, gap in enumerate(state.gaps):
            if abs(gap) >= self.n_blobs - 1:
                available_actions.append(Action(n_gap, -int(math.copysign(1, gap))))
                continue
            available_actions.append(Action(n_gap, +1))
            available_actions.append(Action(n_gap, -1))
        return available_actions

    def reset(self) -> ColonyState:
        self.state = ColonyState((0,)*(self.n_rods - 1))
        return self.state

    def step(self):
        # TODO: get les actions
        # TODO: choisir une action
        # TODO: appliquer l'action generer le fichier const pr cette action
        # TODO: lancer la simulation
        # TODO: utiliser les informations du .config genere et les sauvegarder dans un fichier clones et un autre .config au fur et a mesure
        # TODO: Calculer la vitesse instantan'ee et le reward
        pass


def main(input_directory, output_directory, Nblobs, Nrods, dt, Nstep, freq):

    a = 0.183228708092682
    
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

    ## Generate clones files
    filename_clones = filename +  '.clones'
    
    pos = np.zeros((Nrods,3))
    quat = np.zeros((Nrods,4))
    
    for n in range(Nrods):
        pos[n,2] = (n-1)*2*a
        quat[n,0] = 1
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
    filename_const = filename +  '.const'
    write_move_const(output_folder,
                     filename_const,
                     Nrods=Nrods,
                     moving_rod=0,
                     direction=-1,
                     a=a,
                     dt=dt
                     )
    
    ## Modify input file accordingly
    filename_input = 'inputfile_bacillaria'
    input_path = os.path.join(input_directory, filename_input + '.dat')

    # Valeurs à mettre à jour
    updates = {
        'dt': [str(dt)],
        'n_steps': [str(1)], # RL feedback
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
