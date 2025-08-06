#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 9 13:00:01 2025

@author: Ely Cheikh Abass

Script utilitaire pour copier les dossiers de simulation vers un serveur distant.
Facilite la gestion des résultats de simulation sur des systèmes distribués.
"""

import paramiko
from scp import SCPClient
import os
from sim_env import RewardMethod

# Remote base path
remote_base_path = "/moogwai/usr/ely/RigidMultiblobsWall-master-JLD/multi_bodies/examples/RL_diatoms"
remote_folders = []
nb_blobs = [2, 5, 10]
tested_params = [(200, 1000), (40, 5000)]

# Build remote folder paths
for nb_steps, nb_epoch in tested_params:
    for blobs in nb_blobs:
        for method, angle in [(RewardMethod.FORWARD_PROGRESS, 90), (RewardMethod.FORWARD_PROGRESS, 0),
                                      (RewardMethod.CIRCULAR_ZONES, 0)]:

            folder_name = f"ppo_3r_{blobs}b_ep_{nb_epoch}_step_{nb_steps}_meth_{method.value}_ang_{angle}"
            full_path = os.path.join(remote_base_path, folder_name)
            remote_folders.append(full_path)

# Local path to save the folders
local_base_path = '../training_results'

# SSH credentials
host = 'gibi.polytechnique.fr'
port = 22
username = 'ely'
password = 'pc,01:ma'

# Ensure local base folder exists
os.makedirs(local_base_path, exist_ok=True)

# Set up SSH and SCP
ssh = paramiko.SSHClient()
ssh.load_system_host_keys()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(hostname=host, port=port, username=username, password=password)

scp = SCPClient(ssh.get_transport())

# Download each remote folder to local
for remote_folder in remote_folders:
    folder_name = os.path.basename(remote_folder)
    # local_target = os.path.join(local_base_path, folder_name)
    # os.makedirs(local_target, exist_ok=True)

    # This gets the entire folder recursively
    try:
        scp.get(remote_folder, local_path=local_base_path, recursive=True)
        print(f"Downloaded: {remote_folder} -> {local_base_path}")
    except Exception as e:
        print(f"Failed to download {remote_folder}: {e}")

# Clean up
scp.close()
ssh.close()
print("All folders downloaded.")