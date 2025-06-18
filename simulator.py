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
from utils import get_sim_folder
from typing import List
from sim_env import DiatomEnv

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

def main(input_directory, output_directory, Nblobs, Nrods, dt, Nstep):
    a = 0.183228708092682  # blob radius
    DiatomEnv(input_directory, output_directory, Nrods, Nblobs, dt, a)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Traite des fichiers avec optimisation d'efficacité.")
    parser.add_argument("--input_directory", type=str, required=True, help= "Fichier de d'entree")
    parser.add_argument("--output_directory", type=str, required=True, help= "Fichier de sortie")
    parser.add_argument("--Nblobs", type=int, required=True, help="Nombre de blobs")
    parser.add_argument("--Nrods", type=int, required=True, help="Nombre de rods")
    parser.add_argument("--dt", type=float, default= 0.00125, help = "Timestep")
    parser.add_argument("--Nstep", type=int, default = 160, help = "Nombre d'étapes")
    parser.add_argument("--freq", type=float, default = 10, help = "fréquence du sinus dans le coulissement")
    
    args = parser.parse_args()
    
    main(args.input_directory, args.output_directory, args.Nblobs, args.Nrods, args.dt, args.Nstep)
