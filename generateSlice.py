# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 14:57:22 2019

@author: JHodges
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from collections import defaultdict
import pandas as pd

import pyfdstools as fds
import os
import glob
import struct
import scipy.ndimage as scim

def readNextTime(f, size):
    (NX, NY, NZ) = (size[1], size[3], size[5])
    _ = np.frombuffer(f.read(8), dtype=np.float32)
    time = np.frombuffer(f.read(4), dtype=np.float32)
    _ = np.frombuffer(f.read(8), dtype=np.float32)
    try:
        data = np.frombuffer(f.read((NX+1)*(NY+1)*(NZ+1)*4), 
                             dtype=np.float32)
    except:
        data = False
    return time, data

def readSLCFheader(f):
    data = f.read(142)
    header = data[:110]
    size = struct.unpack('>iiiiii', data[115:139])
    tmp = header.split(b'\x1e')
    quantity = tmp[1].decode('utf-8').replace('\x00','').strip(' ')
    shortName = tmp[3].decode('utf-8').replace('\x00','').strip(' ')
    units = tmp[5].decode('utf-8').replace('\x00','').strip(' ')
    
    return quantity, shortName, units, size

def writeSLCFheader(f, quantity, shortName, units, size):
    sz = struct.pack('>%0.0fi'%(len(size)), *size)
    qty = str.encode("{:<30}".format(quantity))
    sn = str.encode("{:<30}".format(shortName))
    un = str.encode("{:<30}".format(units))
    f.write(b'\x1e\x00\x00\x00')
    f.write(qty)
    f.write(b'\x1e\x00\x00\x00\x1e\x00\x00\x00')
    f.write(sn)
    f.write(b'\x1e\x00\x00\x00\x1e\x00\x00\x00')
    f.write(un)
    f.write(b'\x1e\x00\x00\x00\x18')
    f.write(sz)
    f.write(b'\x00\x00\x00')

def writeTime(f, time, data):
    f.write(b'\x00\x00\x00\x00\x00\x00\x00\x00')
    t = time.tobytes()
    f.write(t)
    f.write(b'\x00\x00\x00\x00\x00\x00\x00\x00')
    d = data.tobytes()
    f.write(d)


def buildP3Ddata(file, compData, quantities, time, C=3, visibilityMax=30):
    comps = list(file.comps.keys())
    datas = dict()
    for i, c in enumerate(comps):
        comp = file.comps[c]
        height = comp['HEIGHT']
        depth = comp['DEPTH']
        width = comp['WIDTH']
        origin = comp['ORIGIN']
        cells = comp['GRID']
        (nx, ny, nz) = (cells[0], cells[1], cells[2])
        (x0, y0, z0) = (origin[0], origin[1], origin[2])
        (x1, y1, z1) = (x0+width, y0+depth, z0+height)
        #XB = [x0, x1, y0, y1, z0, z1]
        #xaxis = np.linspace(x0, x1, nx)
        #yaxis = np.linspace(y0, y1, ny)
        zaxis = np.linspace(z0, z1, nz)

        times = compData['Time']
        tInd = np.argmin(abs(times-time))
        
        suffix = "_%d"%(i+1)
        
        HGL = compData['HGT%s'%(suffix)]
        ULT = compData['ULT%s'%(suffix)]
        ULO2 = compData['ULO2%s'%(suffix)]
        ULCO2 = compData['ULCO2%s'%(suffix)]
        ULCO = compData['ULCO%s'%(suffix)]
        ULOD = compData['ULCO%s'%(suffix)]
        
        LLT = compData['LLT%s'%(suffix)]
        LLO2 = compData['LLO2%s'%(suffix)]
        LLCO2 = compData['LLCO2%s'%(suffix)]
        LLCO = compData['LLCO%s'%(suffix)]
        LLOD = compData['LLCO%s'%(suffix)]
        
        zInd = np.argmin(abs(zaxis-HGL[tInd]))
        
        data = np.zeros((nx, ny, nz, 5))
        for j in range(0, 5):
            if quantities[j] == 'TEMPERATURE':
                data[:, :, :zInd, j] = LLT[tInd]
                data[:, :, zInd:, j] = ULT[tInd]
            elif quantities[j] == 'OXYGEN':
                data[:, :, :zInd, j] = LLO2[tInd]
                data[:, :, zInd:, j] = ULO2[tInd]
            elif quantities[j] == 'CARBON MONOXIDE':
                data[:, :, :zInd, j] = LLCO[tInd]
                data[:, :, zInd:, j] = ULCO[tInd]
            elif quantities[j] == 'CARBON DIOXIDE':
                data[:, :, :zInd, j] = LLCO2[tInd]
                data[:, :, zInd:, j] = ULCO2[tInd]
            elif quantities[j] == 'VISIBILITY':
                if LLOD[tInd] > 0:
                    data[:, :, :zInd, j] = min([C/(2.3*LLOD[tInd]), visibilityMax])
                else:
                    data[:, :, :zInd, j] = visibilityMax
                if ULOD[tInd] > 0:
                    data[:, :, zInd:, j] = min([C/(2.3*ULOD[tInd]), visibilityMax])
                else:
                    data[:, :, zInd:, j] = visibilityMax
        datas[comp] = data
    return datas

if __name__ == "__main__":
    
    resultDir = "directory with resutls files"
    chid = "chid"
    
    fdsFileName = "%s%s.fds"%(resultDir, chid)
    smvFile = "%s%s.smv"%(resultDir, chid)
    devcFile = "%s%s_devc.csv"%(resultDir, chid)
    
    slcfFiles = glob.glob('%s%s*.sf'%(resultDir, chid))
    
    quantities, slcfFiles, dimensions, meshes = fds.readSLCFquantities(chid, resultDir)
    
    meshDict = defaultdict(bool)
    for qty, file, dim, mesh in zip(quantities, slcfFiles, dimensions, meshes):
        if meshDict[mesh] is False: meshDict[mesh] = defaultdict(bool)
        (NX, NY, NZ) = (dim[1] - dim[0], dim[3] - dim[2], dim[5] - dim[4])
        if ((NX > 0 and NY > 0) and (NZ > 0)):
            meshDict[mesh][qty] = file
    
    
    with open(smvFile, 'r') as f3:
        smvText = f3.readlines()
    #for slcfFile in slcfFiles:
    meshes = list(meshDict.keys())
    for mesh in meshes:
        iintFile = meshDict[mesh]['INTEGRATED INTENSITY']
        f = open(iintFile, 'rb')
        quantity, shortName, units, size = readSLCFheader(f)
        sliceNum = float(iintFile.split('_')[-1].replace('.sf',''))
        meshNum = float(iintFile.split('_')[-2])
        (NX, NY, NZ) = (size[1], size[3], size[5])
        shape = (NX+1, NY+1, NZ+1)
        if NZ > 40:
            NZ2 = 8
        else:
            NZ2 = 4
        size2 = [0, NX, 0, NY, NZ2, NZ2]
        
        tempFile = meshDict[mesh]['TEMPERATURE']
        f1 = open(tempFile, 'rb')
        quantity1, shortName1, units1, size = readSLCFheader(f1)
        
        quantity2 = 'RADIATIVE HEAT FLUX GAS'
        shortName2 = 'RHFG'
        units2 = 'kW/m2'
        rhfgFileOut = iintFile.split('_')
        rhfgFileOut[-1] = '%.0f.sf'%(sliceNum+1000)
        rhfgFileOut = '_'.join(rhfgFileOut)
        f2 = open(rhfgFileOut, 'wb')
        writeSLCFheader(f2, quantity2, shortName2, units2, size)
        
        quantity3 = 'PCB TEMPERATURE'
        shortName3 = 'PCBT'
        units3 = 'C'
        rho = 1900
        cp = 1.2
        delta = 1.6/1000
        emi = 0.9
        threshold = 65.0
        TambK = 20 + 273.15
        sigma = 5.67*10**-11
        
        pcbtFileOut = iintFile.split('_')
        pcbtFileOut[-1] = '%.0f.sf'%(sliceNum+2000)
        pcbtFileOut = '_'.join(pcbtFileOut)
        f3 = open(pcbtFileOut, 'wb')
        writeSLCFheader(f3, quantity3, shortName3, units3, size)
        
        quantity4 = 'FAILURE TIME'
        shortName4 = 'FTIME'
        units4 = 'min'
        ftimeFileOut = iintFile.split('_')
        ftimeFileOut[-1] = '%.0f.sf'%(sliceNum+3000)
        ftimeFileOut = '_'.join(ftimeFileOut)
        f4 = open(ftimeFileOut, 'wb')
        writeSLCFheader(f4, quantity4, shortName4, units4, size2)
        
        pcbTempK = np.zeros((NX+1)*(NY+1)*(NZ+1), dtype=np.float32) + 20.0 + 273.15
        failureTimes = np.zeros((NX+1)*(NY+1)*(NZ+1), dtype=np.float32) + 3600
        iint_time, iint_data = readNextTime(f, size)
        temp_time, temp_data = readNextTime(f1, size)
        time = iint_time
        oldTime = 0
        while len(time) > 0:
            if (len(iint_time) > 0 and len(temp_time) > 0):
                temp_dataK = temp_data + 273.15
                
                qrad_amb = iint_data - 4*(sigma)*(TambK)**4
                h = 1.31 * (temp_dataK - TambK) ** (1/3) / 1000
                qnet_amb = emi * qrad_amb + h * (temp_dataK - TambK)
                
                qnet = qnet_amb - emi * sigma * (pcbTempK**4 - TambK**4) - h * (pcbTempK - TambK)
                
                dt = time-oldTime
                pcbTempK = pcbTempK + qnet*dt/(rho*cp*delta)
                #data22 = np.reshape(data2, shape, order='F')
                #data22 = scim.filters.median_filter(data22, 3)
                #data3 = np.reshape(data22, (data2.shape[0],), order='F')
                writeTime(f2, time, qrad_amb)
                
                pcbTemp = pcbTempK.copy() - 273.15
                
                mask = pcbTemp > threshold
                pcbTemp[mask] = threshold
                writeTime(f3, time, pcbTemp)
                
                mask2 = failureTimes > time[0]
                failureTimes[np.logical_and(mask, mask2)] = time[0]
                #mask2 = failureTimes[mask] > time[0]
                #failureTimes[mask][mask2] = time[0]
                
                failureTimes2 = np.reshape(failureTimes, shape, order='F')
                if NZ > 40:
                    N_MX = int(2.85/0.15)
                else:
                    N_MX = int(2.7/0.30)
                failureT = np.min(failureTimes2[:, :, :N_MX], axis=2)
                #assert False, "Stopped"
                #failureT = np.min(failureTimes2, axis=2)
                for i in range(0, failureTimes2.shape[2]):
                    failureTimes2[:, :, i] = failureT[:, :]
                failureT2 = np.reshape(failureTimes2, (shape[0]*shape[1]*shape[2],), order='F')
                failureT3 = np.reshape(failureT, (shape[0]*shape[1]), order='F')
                writeTime(f4, time, failureT3/60)
            oldTime = time[0]
            iint_data_old = iint_data.copy()
            temp_data_old = temp_data.copy()
            iint_time, iint_data = readNextTime(f, size)
            temp_time, temp_data = readNextTime(f1, size)
            time = iint_time
            
        while oldTime < 3600:
            exptrapolation_dt = dt[0]
            temp_dataK = temp_data_old + 273.15
            time = oldTime + exptrapolation_dt
            
            qrad_amb = iint_data_old - 4*(sigma)*(TambK)**4
            h = 1.31 * (temp_dataK - TambK) ** (1/3) / 1000
            qnet_amb = emi * qrad_amb + h * (temp_dataK - TambK)
            
            qnet = qnet_amb - emi * sigma * (pcbTempK**4 - TambK**4) - h * (pcbTempK - TambK)
            
            pcbTempK = pcbTempK + qnet*exptrapolation_dt/(rho*cp*delta)
            #writeTime(f2, time, qrad_amb)
            pcbTemp = pcbTempK.copy() - 273.15
            
            mask = pcbTemp > threshold
            pcbTemp[mask] = threshold
            #writeTime(f3, time, pcbTemp)
            
            mask2 = failureTimes > time
            failureTimes[np.logical_and(mask, mask2)] = time
            
            failureTimes2 = np.reshape(failureTimes, shape, order='F')
            failureT = np.min(failureTimes2, axis=2)
            for i in range(0, failureTimes2.shape[2]):
                failureTimes2[:, :, i] = failureT[:, :]
            failureT2 = np.reshape(failureTimes2, (shape[0]*shape[1]*shape[2],), order='F')
            failureT3 = np.reshape(failureT, (shape[0]*shape[1]), order='F')
            writeTime(f4, time, failureT3/60)
            oldTime = time
            
        f1.close()
        f2.close()
        f3.close()
        f4.close()
        
        
        smvText.append('SLCC     %0.0f # STRUCTURED &     0    %0.0f     0    %0.0f     0    %0.0f !      %0.0f\n'%(meshNum, NX, NY, NZ, meshNum))
        smvText.append(' %s\n'%(rhfgFileOut.split(os.sep)[-1]))
        smvText.append(' %s\n'%(quantity2))
        smvText.append(' %s\n'%(shortName2))
        smvText.append(' %s\n'%(units2))
        
        smvText.append('SLCC     %0.0f # STRUCTURED &     0    %0.0f     0    %0.0f     0    %0.0f !      %0.0f\n'%(meshNum, NX, NY, NZ, meshNum))
        smvText.append(' %s\n'%(pcbtFileOut.split(os.sep)[-1]))
        smvText.append(' %s\n'%(quantity3))
        smvText.append(' %s\n'%(shortName3))
        smvText.append(' %s\n'%(units3))
        
        
        smvText.append('SLCC    %0.0f # STRUCTURED &     0    %0.0f     0    %0.0f    %0.0f    %0.0f !     %0.0f\n'%(meshNum, NX, NY, NZ2, NZ2, meshNum))
        smvText.append(' %s\n'%(ftimeFileOut.split(os.sep)[-1]))
        smvText.append(' %s\n'%(quantity4))
        smvText.append(' %s\n'%(shortName4))
        smvText.append(' %s\n'%(units4))
            
        f.close()
        
    with open(smvFile.replace('.smv','_mod.smv'), 'w') as f:
        f.writelines(smvText)
    
    slcfFile = "%s%s_0001_01.sf"%(resultDir, chid)
    
    