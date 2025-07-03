import paramiko
from scp import SCPClient
import os

# Remote base path
remote_base_path = "/moogwai/usr/ely/RigidMultiblobsWall-master-JLD/multi_bodies/examples/RL_diatoms"
remote_folders = []

# Build remote folder paths
for i in range(2):
    for j in range(10):
        folder_name = f"0.{j}_{'x_mvt' if i == 0 else 'z_mvt'}"
        full_path = os.path.join(remote_base_path, folder_name)
        remote_folders.append(full_path)

# Local path to save the folders
local_base_path = './training_results'

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
