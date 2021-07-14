# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 21:04:20 2018

@author: jodie
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 20:45:54 2018

@author: jodie
"""

from numba import cuda, int16, float32, from_dtype, jit
import numba as nb
import numpy as np
from Samplegeneration import *
from DEMgraphics import *
import GPUDEMcnstnt as GPUDEM
import ColliosiondetectionregularCnstnt as CD
import importlib

def runDEMGPU(mass,stiffnessC,stiffnessS,dt,pulloutV,iterations,distance,sample,mod,KtB):
    importlib.reload(GPUDEM)
    importlib.reload(CD)
    Strngth=np.zeros((11,11,2))
    Strngth[:,:,0]=1e9
    Strngth[1,2,0]=1e5
    Strngth[2,1,0]=1e5
    Strngth[:,:,1]=2
    rad=[sample[0],sample[0],sample[0],sample[0],sample[0],sample[0],sample[0],sample[0],sample[0],sample[0],sample[0]]

    poss=[0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2]
    dmp=[0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7]
    mss=[mass,mass,mass,mass,mass,mass,mass,mass,mass,mass,mass]
    Stiff=[stiffnessC,stiffnessC,stiffnessS,stiffnessS,stiffnessS,stiffnessC,stiffnessC,stiffnessC,stiffnessC,stiffnessC,stiffnessC]
    vl=[[0,0,0,0]]*11
    vl[0]=[0,0,0,0]
    vl[1]=[1,0,0,0]
    vl[2]=[1,0,0,0]
#    vl[2]=[0,-pulloutV[0],0,0]
    vl[3]=[0,pulloutV[0],pulloutV[1],0]
    # vl[2]=[0,pulloutV[0],pulloutV[1],0]
#    vl[1]=[0,0,0,0]
    rt=[[1,0,0,0]]*11
#    rt[3]=[0,0,0,0]
    Nx=(pulloutV[0])/((pulloutV[0]**2+pulloutV[1]**2)**0.5)
    Ny=(pulloutV[1])/((pulloutV[0]**2+pulloutV[1]**2)**0.5)
    Prt,hashdata,hashcreationdata=sample_genhccp2(sample[0],sample[1],sample[2],sample[3],sample[4],sample[5])
    Prtcopy=np.copy(Prt)
    Bprp=Bnd_propgenList(rad,Stiff,poss,Strngth,mod,KtB)
    Pprp=Prt_propgenList(rad,mss,vl,dmp,rt,Bprp)
    dTT=np.array([dt],dtype=np.float64)
    
    CD.makeglobalsCollision(Pprp,hashdata,hashcreationdata)
    GPUDEM.makeglobalsDem(Pprp,Bprp,dTT)
    d_Fx, d_Fy, d_Fz, d_Mx, d_My, d_Mz,d_P=sendDATA(Prt,Pprp)
    
    GC=cuda.to_device(np.array([0], dtype=np.int32))
    threads=32
    blocksPrt=(len(Prt)//threads)+1
    print(blocksPrt,threads)
    print('getting size')
    CD.collisionDetectSize[blocksPrt,threads](d_P,GC)
    HostGC= GC.copy_to_host()
    d_Bnd=sendBNDS(HostGC)
    print(HostGC,'number of bonds')
    d_Csample=cuda.to_device(np.array([0,0,0,0,0,0,0,0,0,0], dtype=np.float64))
    d_CCsample=cuda.to_device(np.array([0,Nx,Ny,0], dtype=np.float64))
    
    dGC=cuda.to_device(np.array([0], dtype=np.int32))
    dF=cuda.to_device(np.array([10000,0.6], dtype=np.float64))
    
#    print(1/0)
    CD.collisionDetect[blocksPrt,threads](d_P,d_Bnd,dGC)

    Fnx=[] #0
    slip=[] #1
    U=[]#2
    number=[] #3
    totaldamage=[]#4
    broken=[]#5
    ForceX=[]
#    Fnx,slip,U,number,totaldamage,broken,ForceX
    T=0
    
#    Bnds= d_Bnd.copy_to_host()
    balllist=objgen(Prt,Pprp)
#    linelist=Linegen(Bnds,Prt,Pprp)

    threads=16
    blocksBnds=int((HostGC/threads))+1
    blocksPrt=(len(Prt)//threads)+1
    print(threads,blocksBnds,blocksPrt)
    flag=1
    dxL=[]
    dx=0
    stepsize=dt*(pulloutV[0]**2+pulloutV[1]**2)**0.5
    DMG=0.7
    print("launching")
    c=0
    Flags=[]
    damaged=0
    maxx=0
    minx=100
    maxy=0
    miny=100
    Shear=[]
    Damagecheck=0
    stepforward=0
    stepbackward=0
    for P in Prt:
        if P['x']>maxx:
            maxx=P['x']
        if P['y']>maxy:
            maxy=P['y']
        if P['x']<minx:
            minx=P['x']
        if P['y']<miny:
            miny=P['y']
    for i in range (0,iterations):
        for subi in range(0,10):
            dx+=stepsize
            c+=1
            T+=1
            if flag==1:
                GPUDEM.ContactHandler[blocksBnds,threads](d_P,d_Bnd,d_Fx,d_Fy,d_Fz,d_Mx,d_My,d_Mz)
                GPUDEM.Integrator[blocksPrt,threads](d_P,d_Fx,d_Fy,d_Fz,d_Mx,d_My,d_Mz)
                stepforward+=1
            if flag==0:
                GPUDEM.ContactHandlerNoDMG[blocksBnds,threads](d_P,d_Bnd,d_Fx,d_Fy,d_Fz,d_Mx,d_My,d_Mz)
                GPUDEM.IntegratorLocked[blocksPrt,threads](d_P,d_Fx,d_Fy,d_Fz,d_Mx,d_My,d_Mz)
                stepbackward+=1
                
        GPUDEM.SampleBnds[blocksBnds,threads](d_Bnd,d_Csample,d_P)
        C=d_Csample.copy_to_host()
        GPUDEM.ContactHandlerNoDMG[blocksBnds,threads](d_P,d_Bnd,d_Fx,d_Fy,d_Fz,d_Mx,d_My,d_Mz)
        GPUDEM.SamplePrt[blocksPrt,threads](d_CCsample,d_P,d_Fx,d_Fy)
        CC=d_CCsample.copy_to_host()
        if flag==1:
            Flags.append(flag)
            Fnx.append(C[0]) #0
            slip.append(C[1]) #1
            U.append(C[2])#2
            number.append(C[3]) #3
            totaldamage.append(C[4])#4
            broken.append(C[5])#5
            ForceX.append(CC[0])

        if flag==1:
            GPUDEM.Integrator[blocksPrt,threads](d_P,d_Fx,d_Fy,d_Fz,d_Mx,d_My,d_Mz)
            stepforward+=1
        if flag==0:
            GPUDEM.IntegratorLocked[blocksPrt,threads](d_P,d_Fx,d_Fy,d_Fz,d_Mx,d_My,d_Mz)
            stepbackward+=1


        
        if C[4]/C[3]>=DMG:
            DMG+=0.02
#            print(DMG)
            flag = 0
            c=0
            
            if DMG==1.2:
                break
        if c>10000 and flag==0:
            flag=1
            c=0

        dxL.append(dx)
        

        if stepbackward*3>=stepforward:
            flag=1
            stepforward=0
            stepbackward=0
            

        if C[4]/C[3]>0:
            kt=1e9
            Prt= d_P.copy_to_host()
            Bnds= d_Bnd.copy_to_host()
            for b in Bnds:
                if b["tYpe"]==12:
                    p1=Prt[b['P1']]
                    p2=Prt[b['P2']]
                    Cox=(p1["x"]+p2["x"])/2
                    S=abs(b["Usx"]*kt*b["D"])
                    Shear.append([S,Cox,C[4]/C[3],stiffnessC,stiffnessS,pulloutV[0]])
            # break
        GPUDEM.Reset[1,1](d_Csample)
        GPUDEM.ResetCC[1,1](d_CCsample)

        
    Prt= d_P.copy_to_host()
    Bnds= d_Bnd.copy_to_host()

    return Bprp,Fnx,slip,U,number,totaldamage,broken,ForceX,dxL,Prt,Prtcopy,Bnds,Flags,Shear

def sendDATA(Prt,Pprop):
    
    Particle = np.dtype([
    # sphere (x, y, z) coordinates
    ('x', 'f8'),  ('y', 'f8'), ('z', 'f8'), 
    # sphere Type 
    ('tYpe', 'u1'),
    ('hash', 'u4'),
    # sphere (Vx, Vy, Vz) velocity 
    ('Vx', 'f8'),  ('Vy', 'f8'), ('Vz', 'f8'),
    # sphere (Vx, Vy, Vz) velocity 
    ('ax', 'f8'),  ('ay', 'f8'), ('az', 'f8'),
    #angular velocity
    ('Wx', 'f8'),  ('Wy', 'f8'), ('Wz', 'f8'),
    ], align=True) 
    Particle_nb = from_dtype(Particle)

    d_P=cuda.device_array(Prt.shape[0], dtype=Particle_nb)
    
    d_Fx=cuda.to_device(np.zeros(Prt.shape[0], dtype=np.float64))
    d_Fy=cuda.to_device(np.zeros(Prt.shape[0], dtype=np.float64))
    d_Fz=cuda.to_device(np.zeros(Prt.shape[0], dtype=np.float64))      
    d_Mx=cuda.to_device(np.zeros(Prt.shape[0], dtype=np.float64))
    d_My=cuda.to_device(np.zeros(Prt.shape[0], dtype=np.float64))
    d_Mz=cuda.to_device(np.zeros(Prt.shape[0], dtype=np.float64))

    cuda.to_device(Prt,to=d_P)
    return d_Fx, d_Fy, d_Fz, d_Mx, d_My, d_Mz,d_P 

def sendBNDS(number):
    Bond = np.dtype([
    # Bond (d0, d1, d2) values
    # Bond Damage 
    ('D', 'f8'),
    ('L', 'f8'),
    ('X0', 'f8'),
    # Particles in bond 
    ('P1', 'i4'), ('P2', 'i4'),
    # Bond max strain 
    #Normal Vector
    ('Nx', 'f8'),  ('Ny', 'f8'), ('Nz', 'f8'),
    #Bond Type
    ('tYpe','u1'),
    #Normal strain 
    ('Unx', 'f8'),  ('Uny', 'f8'), ('Unz', 'f8'),
    ('Usx', 'f8'),  ('Usy', 'f8'), ('Usz', 'f8'),
    ('Wnx', 'f8'),  ('Wny', 'f8'), ('Wnz', 'f8'),
    ('Wsx', 'f8'),  ('Wsy', 'f8'), ('Wsz', 'f8'),
    ('MaxK','f8'),], align=True)
    Bond_nb = from_dtype(Bond)
    Bnds=np.empty(int(number),dtype=Bond_nb)
    d_Bnd=cuda.device_array(Bnds.shape[0], dtype=Bond_nb)
    cuda.to_device(Bnds,to=d_Bnd)
    return d_Bnd