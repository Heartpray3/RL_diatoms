from __future__ import division, print_function
import os
import numpy as np
import math as m
import shutil
import argparse


    
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


    
def main(Nblobs, phase_shift, Nrods):

    a = 0.183228708092682   
    freq = 10
    
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
    
    print('Nrods = ', Nrods)
    print('phase_shift = ', phase_shift)
    suffix_phase_shift = '_phase_shift_' +  str(phase_shift).replace('.', '_') + 'pi'
    foldername = filename + suffix_phase_shift
    
    if not os.path.exists(foldername):
           os.makedirs(foldername)
    
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
    
    fid= open(foldername + '/' + filename_clones,'w')
    fid.write(str(Nrods)+ '\n')
    np.savetxt(fid,to_save)
    fid.close()
     
    ## Generate Vertex files 
    
    with open(root_name + str(Nblobs) +  '_blobs.vertex', "w") as f:
        f.write(str(Nblobs) + "\n")
        Start = -((Nblobs/2-1)*X_step + X_coef)
        for i in range(Nblobs):
            X = Start + i*X_step
            f.write("{:.7e} {:.7e} {:.7e}\n".format(X, 0, 0))
            
    ## Generate list vertex files 
    filename_vertex = '../' + root_name + str(Nblobs) +  '_blobs.vertex'
    
    filename_list_vertex = filename +  '.list_vertex'
    fid= open(foldername + '/' + filename_list_vertex,'w')
    
    for n in range(Nrods):
        fid.write(filename_vertex + '\n')
    fid.close()
    
    ## Constraints
    Nconst = (Nrods-1)*Nconst_per_rod
    filename_const = filename +  '.const'
    fid= open(foldername + '/' + filename_const, 'w')
    fid.write(str(Nrods) + '\n')
    fid.write(str(Nconst) + '\n')
    
    sixzeros = '0 0 0 0 0 0'
    for n in range(Nrods-1):
    
        ps = phase_shift*n*m.pi/(Nrods-1) 
    
        if Nconst_per_rod >= 1:
          c1 = str(n) +  ' '\
            + str(n+1) +  ' '\
            + sixzeros + ' '\
            + str(L) +  '*sin(' +  str(2*m.pi*freq) + '*t+' +  str(ps) +  ') '\
            +  ' 0 '\
            + str(a) +  ' '\
            + str(-L) + '*sin(' +  str(2*m.pi*freq) + '*t+' +  str(ps) +  ') '\
            + ' 0 '\
            + str(-a) + ' '\
            + str(L*2*m.pi*freq) +  '*cos(' + str(2*m.pi*freq) + '*t+' + str(ps) + ') '\
            + ' 0 '\
            + ' 0 '\
            + str(-L*2*m.pi*freq) +  '*cos('+ str(2*m.pi*freq) + '*t+' + str(ps) + ') '\
            + ' 0 '\
            + ' 0 '
          fid.write(c1 + '\n')
    
        if Nconst_per_rod >= 2:
          offset = 0.1
          # Location of second is arbitrary as long as it does not coincide with link 1
          c2 = str(n) +  ' '\
            + str(n+1) +  ' '\
            + sixzeros + ' '\
            + str((1+offset)*L) +  '*sin(' +  str(2*m.pi*freq) + '*t+' +  str(ps) +  ') '\
            + ' 0 '\
            + str((1+offset)*a) +  ' '\
            + str(-(1-offset)*L) +  '*sin(' +  str(2*m.pi*freq) + '*t+' +  str(ps) +  ') '\
            + ' 0 '\
            + str(-(1-offset)*a) +  ' '\
            + str((1+offset)*L*2*m.pi*freq) +  '*cos(' + str(2*m.pi*freq) + '*t+' + str(ps) + ') '\
            + ' 0 '\
            + ' 0 '\
            + str(-(1-offset)*L*2*m.pi*freq) +  '*cos(' + str(2*m.pi*freq) + '*t+' + str(ps) + ') '\
            + ' 0 '\
            + ' 0 '
    
          fid.write(c2 + '\n')
    
    fid.close()
    
    ## Modify input file accordingly
    filename_input = 'inputfile_bacillaria'
    fid = open(filename_input +  '.dat','r')
    C=fid.readlines()
    fid.close()
    
    Line_output = C[34]
    Line_output = Line_output.split()
    Line_output[1] = 'run_' +  foldername
    Line_output = ' '.join(Line_output)
    C[34] = Line_output
    
    Line_const = C[39]
    Line_const = Line_const.split()
    Line_const[1] =  filename_list_vertex
    Line_const[2] =  filename_clones
    Line_const[3] =  filename_const
    Line_const = ' '.join(Line_const)
    C[39] = Line_const
    
    
    filename_input_local = filename_input +  '_' +  suffix_Nrods + suffix_phase_shift +  '.dat'
    fid = open(foldername + '/' + filename_input_local, 'w')           
    fid.writelines(C) 
    fid.close()
    
    os.chdir(foldername)
    my_command = 'python3 ../multi_bodies_bacillaria.py --input-file ' + filename_input_local #+ ' --print-residual'
    os.system(my_command)
    # os.chdir('../')
    
    #%% Code post-process 
    
    Nstep = 80 #Extract from input files
    
    print(' ')
    print(' ')
    print('Post Process Nblobs : ' + str(Nblobs) + ' Nrods ' + str(Nrods) + '  phase_shift : ' + str(phase_shift))
    Constraint = []
    Velocity_step = []
    Velocity5 = np.zeros((Nstep,Nrods,6))
    Velocity_COM = np.zeros((Nstep,6))
    for i in range(Nstep):
        if len(str(i)) == 1 :
            Nstep_str = '00' + str(i)
        if len(str(i)) == 2 :
            Nstep_str = '0' + str(i)
        if len(str(i)) == 3 :
            Nstep_str = str(i)
        
        psint,psfloat = str(phase_shift).split(".")
        psint = psint[0]
        psfloat = psfloat[0]
    
        f  = open('constraint_power.00000'+Nstep_str,'r')
        data2 = f.read()
        Constraint.append(data2)
        
        #Permet de lire les vitesses 
        f = open('Velocities.00000'+Nstep_str+'.dat','r')
        Velocity_step = f.read().splitlines()
        for j, values_str in enumerate(Velocity_step):
            values_float = np.fromstring(values_str, sep=' ')
            Velocity5[i,j,:] = values_float
        Velocity_COM[i,:] = sum(Velocity5[i,:,:])/Nrods
        
        with open('bacillaria_'+str(Nblobs)+'_blobs_'+str(Nrods)+'_rods_phase_shift_'+str(psint)+'_'+str(psfloat)+'pi.Velocity_COM_step_'+str(i),'w') as f:
            f.write(str(Velocity_COM[i,0]) + ' ' + str(Velocity_COM[i,1]) + ' ' + str(Velocity_COM[i,2]) + ' ' + str(Velocity_COM[i,3]) + ' ' + str(Velocity_COM[i,4]) + ' ' + str(Velocity_COM[i,5]))
    
    Constraint = np.asarray(Constraint)
    Constraint = Constraint.astype(float)
    with open('bacillaria_'+str(Nblobs)+'_blobs_'+str(Nrods)+'_rods_phase_shift_'+str(psint)+'_'+str(psfloat)+'pi.constraint_power','w') as f:
        for line in Constraint:
          f.write(str(line))
          f.write('\n')
    
    #%% Code vertex in files 
    
    From_step = 40
    To_step = Nstep
    
    print(' ')
    print(' ')
    
    suffix_Nrods = str(Nblobs) +  '_blobs_'
    filename1 = root_name + suffix_Nrods
    
    os.chdir('../') 
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
    
    
    position_com_all_ps = np.zeros((1,Nstep,3))
    ind1 = 0
    N_lambda_vec= []
    
    print('Vertex File : Nblobs '+str(Nblobs)+' Nrods '+ str(Nrods) + ' phase_shift ' + str(phase_shift))
    N_lambda = round(phase_shift/2, 4)
    N_lambda_vec.append(N_lambda)
    
    #####Config file#####  
           
    psint,psfloat = str(phase_shift).split(".")
    psint = psint[0]
    psfloat = psfloat[0]
    
    
    os.chdir(foldername) 
    f  = open('run_'+str(root_name)+str(Nblobs)+'_blobs_'+str(Nrods)+'_rods_phase_shift_'+str(psint)+'_'+str(psfloat)+'pi.bacillaria_'+str(Nblobs)+'_blobs_'+str(Nrods)+'_rods.config','r')
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
            
                pos_all_blobs[blobs_act,i,:] = b1_pos[k,:] - position_com[i,:]
                blobs_act += 1
    
    
    for i in range(From_step,To_step):
        
        filename_list_vertex = str(root_name)+str(Nblobs)+'_blobs_'+str(Nrods)+'_rods_phase_shift_'+str(psint)+'_'+str(psfloat)+'pi_Nstep_'+str(i)+'.vertex'
        with open(filename_list_vertex,'w') as fid:
            fid.write(str(blobs*Nrods) + '\n')
            for j in range(blobs*Nrods):
                fid.write(str(pos_all_blobs[j,i,0]) + ' ' + str(pos_all_blobs[j,i,1]) + ' ' + str(pos_all_blobs[j,i,2]) + ' ' + '\n') 
        fid.close()
    
    #%% Code Generate_file_and_execute 
    
    print(' ')
    print(' ')
    
    file = '../inputfile_body_mobility.dat'
    mydir = os.getcwd()
    shutil.copy(file, mydir)
    
    file = '../Cylinder_N_14_Lg_1_9295_Rg_0_18323.clones'
    mydir = os.getcwd()
    shutil.copy(file, mydir)
    
    print('Resistance : Nblobs '+str(Nblobs)+' Nrods '+ str(Nrods) + ' phase_shift ' + str(phase_shift))
    
    for nstep in range(From_step,To_step):
        
    
        ## Modify input file accordingly
        filename_input = 'inputfile_body_mobility'
        fid = open(filename_input +  '.dat','r')
        C=fid.readlines()
        fid.close()
        
        # Output_file 
        Line_output = C[15]
        Line_output = Line_output.split()
        Line_output[1] = 'run_' +  foldername + '_Nstep_' + str(nstep)
        Line_output = ' '.join(Line_output)
        C[15] = Line_output
    
        # Velocity_file 
        Line_output = C[18]
        Line_output = Line_output.split()
        Line_output[1] = 'bacillaria_' + str(Nblobs) + '_blobs_' + str(Nrods) + '_rods' + suffix_phase_shift + '.Velocity_COM_step_'+str(nstep)
        Line_output = ' '.join(Line_output)
        C[18] = Line_output
        
        # Structure 
        Line_output = C[25]
        Line_output = Line_output.split()
        Line_output[1] = 'bacillaria_' + str(Nblobs) + '_blobs_' + str(Nrods) + '_rods'+suffix_phase_shift+'_Nstep_'+str(nstep)+'.vertex'
        Line_output = ' '.join(Line_output)
        C[25] = Line_output
    
        filename_input_local = filename_input +  '_' +  suffix_Nrods + suffix_phase_shift + '_Nstep_' + str(nstep) + '.dat'
        fid = open(filename_input_local, 'w')           
        fid.writelines(C) 
        fid.close()
    
        my_command = 'python3 ../multi_bodies_utilities.py --input-file ' + filename_input_local #+ ' --print-residual'
        os.system(my_command)
    
    
    #%% Code Effeciency
    
    
    #Contraint power 
    f  = open('bacillaria_'+str(Nblobs)+'_blobs_'+str(Nrods)+'_rods_phase_shift_'+str(psint)+'_'+str(psfloat)+'pi.constraint_power','r')
    data = f.read()
    data = data.split()
        
    constraint = []
    for i in range(Nstep):
        constraint.append(data[i])
    constraint = np.asarray(data)
    constraint = constraint.astype(float)
    constraint = constraint[(To_step - From_step):Nstep] 
    constraint = np.sum(constraint)/(To_step - From_step)
    power_swim_arr = constraint
     
    print(' ')
    print(' ')
    print('power_swim_arr')
    print(power_swim_arr)
    
    #Power mean 

    Velocity_X = []
    Velocity_Z = []
    R_FU_mean = np.zeros((3,3))
    V_tilde = np.zeros((3))
    for i in range(From_step,To_step):
        
        #Force
        f = open('bacillaria_'+str(Nblobs)+'_blobs_'+str(Nrods)+'_rods_phase_shift_'+str(psint)+'_'+str(psfloat)+'pi.Velocity_COM_step_'+str(i))
        data4 = f.read()
        data4 = data4.split()
        velocity = np.asarray(data4)
        velocity = velocity.astype(float)
                    
        Velocity_X.append(velocity[0])
        Velocity_Z.append(velocity[2])
                    
        #Matrice R_FU
        f = open('run_bacillaria_'+str(Nblobs)+'_blobs_'+str(Nrods)+'_rods_phase_shift_'+str(psint)+'_'+str(psfloat)+'pi_Nstep_'+str(i)+'.RFU.dat')
        data5 = f.readlines()[0:]
        data5 = [row.split() for row in data5]
        data5 = [[float(element) for element in row] for row in data5]
        arr = np.asarray(data5)
        R_FU_mean += arr
    
    V_tilde[0]= np.mean(Velocity_X)
    V_tilde[2]= np.mean(Velocity_Z)
    print(' ')
    print('V_tilde')
    print(V_tilde)
    
    R_FU = R_FU_mean/(To_step - From_step)
    print(' ')
    print('R_FU')
    print(R_FU)
    
    F_mean = R_FU @ V_tilde
    print(' ')
    print('F_mean')
    print(F_mean)
    
    power_pull_arr = np.dot(F_mean,V_tilde)
    print(' ')
    print('power_pull_arr')
    print(power_pull_arr)
    
    efficiency = power_pull_arr/power_swim_arr
                
    print(' ')
    print(' ')
    print('Effciency = ', efficiency)
    print(' ')
    print(' ')

    fichier = open('Efficiency_'+str(Nblobs)+'_blobs_'+str(Nrods)+'_rods_phase_shift_'+str(psint)+'_'+str(psfloat)+'pi.txt', 'a')
    fichier.write(str(efficiency))
    fichier.close()
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Traite des fichiers avec optimisation d'efficacité.")
    parser.add_argument("--Nblobs", type=int, required=True, help="Nombre de blobs")
    parser.add_argument("--phase_shift", type=float, required=True, help="Décalage de phase")
    parser.add_argument("--Nrods", type=int, required=True, help="Nombre de rods")
    
    args = parser.parse_args()
    
    main(args.Nblobs, args.phase_shift, args.Nrods)
