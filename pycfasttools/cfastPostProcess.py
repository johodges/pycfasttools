import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import glob
import pandas as pd

def importCfastSheet(file, headerrow=0, skiprows=4):
    with open(file, 'r') as f:
        data = f.read()
    data = data.split('\n')
    header = data[headerrow].split(',')
    data = pd.read_csv(file, header=None, names=header, skiprows=4)
    return data

def check_plane(pt, box):
    if (pt[0] >= box[1]) or (box[0] >= pt[0]):
        return False
    if (pt[1] >= box[3]) or (box[2] >= pt[1]):
        return False
    return True

def readCfastData(file, tstart=30, tend=600):
    #name = file.split(os.sep)[-1]
    columns=['l','w','h',
         'DSE','DST','DSZi','DSZn','DSMl','DSMu','DSTl','DSTu','DSName',
         'DNE','DNT','DNZi','DNZn','DNMl','DNMu','DNTl','DNTu','DNName',
         'DWE','DWT','DWZi','DWZn','DWMl','DWMu','DWTl','DWTu','DWName',
         'DEE','DET','DEZi','DEZn','DEMl','DEMu','DETl','DETu','DEName']
    data = pd.DataFrame(columns=columns, dtype=object)
    
    comps = file.comps
    vents = file.vents
    
    zoneData = importCfastSheet("%s_zone.csv"%(file.file.replace('.in','')), headerrow=1, skiprows=2)
    compData = importCfastSheet("%s_compartments.csv"%(file.file.replace('.in','')), headerrow=0, skiprows=4)
    
    ventData = importCfastZone(zoneData, tstart=tstart, tend=tend)
    defaultSide = defaultdict(bool)
    defaultSide['mdotL'] = 0.0
    defaultSide['mdotU'] = 0.0
    defaultSide['tL'] = 20.0
    defaultSide['tU'] = 20.0
    defaultSide['zInt'] = 0.0
    defaultSide['zN'] = 0.0
    defaultSide['C_ID'] = 'SOLID'
    defaultSide['ID'] = False
    defaultSide['D_WIDTH'] = -1
    defaultSide['D_HEIGHT'] = -1
    defaultSide['D_POSITION'] = -1
    
    for key in list(comps.keys()):
        for key2 in list(comps[key]['sides'].keys()):
            file.comps[key]['sides'][key2] = defaultSide.copy()
        height = comps[key]['HEIGHT']
    
    for key in list(vents.keys()):
        vent = vents[key]
        ventName = vent['VENTNAME']
        if ventData[ventName]:
            #print(ventData[ventName])
            file.vents[key]['DATA'] = ventData[ventName]
            compIDs = vent['COMP_IDS']
            if compIDs[0] != 'OUTSIDE':
                cwidth = file.comps[compIDs[0]]['WIDTH']
                cdepth = file.comps[compIDs[0]]['DEPTH']
                height = file.comps[compIDs[0]]['HEIGHT']
            else:
                cwidth = file.comps[compIDs[1]]['WIDTH']
                cdepth = file.comps[compIDs[1]]['DEPTH']
                height = file.comps[compIDs[1]]['HEIGHT']
            
            if ventData[ventName]['TYPE'] == 'MECHANICAL':
                width = vent['AREAS'][0]/height
                mVent_cx = vent['OFFSETS'][0] + width/2
                mVent_cy = vent['OFFSETS'][1] + width/2
                
                ycheck = check_plane([mVent_cx, vent['OFFSETS'][1]], [0, cwidth, 0, cdepth])
                xcheck = check_plane([vent['OFFSETS'][0], mVent_cy], [0, cwidth, 0, cdepth])
                # NOTE: DEFINED OPPOSITE SIDE BECAUSE OUTSIDE IS FIRST AND IT GETS SWAPPED LATER
                if (xcheck) and (not ycheck) and (vent['OFFSETS'][1] == 0): vent['FACE'] = 'REAR' # parallel to x-axis
                    #if (vent['OFFSETS'][1] > 0): vent['FACE'] = 'FRONT'
                elif (not xcheck) and (ycheck) and (vent['OFFSETS'][0] == 0): vent['FACE'] = 'RIGHT' # parallel to y-axis
                    #if (vent['OFFSETS'][0] > 0): vent['FACE'] = 'LEFT'
                else:
                    if vent['OFFSETS'][0] == cwidth: vent['FACE'] = 'LEFT'
                    elif vent['OFFSETS'][1] == cdepth: vent['FACE'] = 'FRONT'
                    elif (width == cwidth) and (vent['OFFSETS'][1] == 0): vent['FACE'] = 'REAR'
                    elif (width == cwidth) and (vent['OFFSETS'][1] == cdepth): vent['FACE'] = 'FRONT'
                    elif (width == cdepth) and (vent['OFFSETS'][0] == 0): vent['FACE'] = 'RIGHT'
                    elif (width == cdepth) and (vent['OFFSETS'][0] == cwidth): vent['FACE'] = 'LEFT'
                    elif (vent['OFFSETS'][1] == 0) and (vent['OFFSETS'][0] != 0): vent['FACE'] = 'REAR'
                    elif (vent['OFFSETS'][0] == 0) and (vent['OFFSETS'][1] != 0): vent['FACE'] = 'REAR'
                    else:
                        print(xcheck, ycheck, width, height, mVent_cx, mVent_cy, vent['OFFSETS'], cwidth, cdepth)
            else:
                width = file.vents[key]['WIDTH']
            if width == 0.6: widthID = 0
            elif width == 1.0: widthID = 1
            elif width == 6.0: widthID = 2
            else: widthID = 3
            face = vent['FACE']
            if face == 'REAR': fSide = 'north'
            if face == 'FRONT': fSide = 'south'
            if face == 'LEFT': fSide = 'west'
            if face == 'RIGHT': fSide = 'east'
            #print(vent, cwidth, cdepth)
            '''
            if (fSide == 'south') or (fSide == 'north'):
                sWidth = cwidth
            else:
                sWidth = cdepth
            '''
            if vent['OFFSET'] is not False:
                vOff = vent['OFFSET']
            elif (fSide == 'south') or (fSide == 'north'):
                vOff = vent['OFFSETS'][0]-width/2
            else:
                vOff = vent['OFFSETS'][1]-width/2
            if compIDs[0] != 'OUTSIDE':
                c1origin = file.comps[compIDs[0]]['ORIGIN']
            if compIDs[1] != 'OUTSIDE':
                c2origin = file.comps[compIDs[1]]['ORIGIN']
            if compIDs[0] != 'OUTSIDE':
                c1xs = [c1origin[0], c1origin[0]+file.comps[compIDs[0]]['WIDTH']]
                c1ys = [c1origin[1], c1origin[1]+file.comps[compIDs[0]]['DEPTH']]
            else:
                c1xs = [c2origin[0], c2origin[0]+file.comps[compIDs[1]]['WIDTH']]
                c1ys = [c2origin[1], c2origin[1]+file.comps[compIDs[1]]['DEPTH']]
            if compIDs[1] != 'OUTSIDE':
                c2xs = [c2origin[0], c2origin[0]+file.comps[compIDs[1]]['WIDTH']]
                c2ys = [c2origin[1], c2origin[1]+file.comps[compIDs[1]]['DEPTH']]
            else:
                c2xs = [c1origin[0], c1origin[0]+file.comps[compIDs[0]]['WIDTH']]
                c2ys = [c1origin[1], c1origin[1]+file.comps[compIDs[0]]['DEPTH']]
            if (fSide == 'north') or (fSide == 'south'):
                c1cen = (c1xs[1] + c1xs[0])/2
                Dpos = c1xs[0] + vOff + (width)/2
                c2cen = (c2xs[1] + c2xs[0])/2
            else:
                c1cen = (c1ys[1] + c1ys[0])/2
                Dpos = c1ys[0] + vOff + (width)/2
                c2cen = (c2ys[1] + c2ys[0])/2     
            if (abs(c1cen - Dpos) < 0.1): c1pos = 1
            if (c1cen - Dpos >= 0.1):
                if (fSide == 'south') or (fSide == 'east'): c1pos = 2
                elif (fSide == 'north') or (fSide == 'west'): c1pos = 0
            elif (c1cen - Dpos <= -0.1):
                if (fSide == 'south') or (fSide == 'east'): c1pos = 0
                elif (fSide == 'north') or (fSide == 'west'): c1pos = 2
            if fSide == 'south': sSide = 'north'
            if fSide == 'north': sSide = 'south'
            if fSide == 'west': sSide = 'east'
            if fSide == 'east': sSide = 'west'
            if (abs(c2cen - Dpos) < 0.1): c2pos = 1
            if (c2cen - Dpos >= 0.1):
                if (sSide == 'south') or (sSide == 'east'): c2pos = 2
                elif (sSide == 'north') or (sSide == 'west'): c2pos = 0
            elif (c2cen - Dpos <= -0.1):
                if (sSide == 'south') or (sSide == 'east'): c2pos = 0
                elif (sSide == 'north') or (sSide == 'west'): c2pos = 2
            #print(ventName, compIDs, c1cen, Dpos, c2cen, c1pos, c2pos, c1ys, vOff, width/2)
            if ventData[ventName]['TYPE'] == 'WALL':
                if compIDs[0] != 'OUTSIDE':
                    vdat = ventData[ventName].copy()
                    vdat['C_ID'] = compIDs[1]
                    vdat['D_POSITION'] = c1pos
                    vdat['D_WIDTH'] = widthID
                    vdat['D_HEIGHT'] = vent['TOP']-vent['BOTTOM']
                    file.comps[compIDs[0]]['sides'][fSide] = vdat
                if compIDs[1] != 'OUTSIDE':
                    vdat = ventData[ventName].copy()
                    vdat['mdotU'] = vdat['mdotU']*-1
                    vdat['mdotL'] = vdat['mdotL']*-1
                    vdat['C_ID'] = compIDs[0]
                    vdat['D_POSITION'] = c2pos
                    vdat['D_WIDTH'] = widthID
                    vdat['D_HEIGHT'] = vent['TOP']-vent['BOTTOM']
                    file.comps[compIDs[1]]['sides'][sSide] = vdat
            elif ventData[ventName]['TYPE'] == 'MECHANICAL':
                width = vent['AREAS'][0]/height
                #print(width, height)
                if compIDs[0] == 'OUTSIDE':
                    vdat = ventData[ventName].copy()
                    vdat['C_ID'] = compIDs[0]
                    vdat['D_POSITION'] = c2pos
                    vdat['D_WIDTH'] = widthID
                    vdat['D_HEIGHT'] = height
                    file.comps[compIDs[1]]['sides'][sSide] = vdat
                elif compIDs[1] == 'OUTSIDE':
                    vdat = ventData[ventName].copy()
                    vdat['C_ID'] = compIDs[1]
                    vdat['D_POSITION'] = c1pos
                    vdat['D_WIDTH'] = widthID
                    vdat['D_HEIGHT'] = height
                    file.comps[compIDs[0]]['sides'][fSide] = vdat
                #print(vdat)
        else:
            print(key, ventName, "FAILURE")
    
    return zoneData, ventData, compData

def importCfastZone(data, tstart=30, tend=500, debug=False):
    zs = np.linspace(0,2.0,21)
    hventCounter = 0
    mventCounter = 0
    for key in list(data.keys()):
        if 'HVENT_' in key:
            hventCounter += 1
        if 'MVENT_' in key:
            mventCounter += 1
    ventCounter = hventCounter + mventCounter
    times = data['Time']
    tind1 = np.min(np.where(times >= tstart)[0])
    tind2 = np.max(np.where(times <= tend)[0])
    Tss = np.zeros((zs.shape[0], ventCounter, times.shape[0]))
    Fss = np.zeros((zs.shape[0], ventCounter, times.shape[0]))
    mdots = np.zeros((9, ventCounter, times.shape[0]))
    #print(mdots.shape)
    for k in range(0,times.shape[0]):
        for i in range(1,ventCounter+1):
            if i <= hventCounter:
                namespace = 'HSLAB'
                numspace = '%0.0f'%(i)
            else:
                namespace = 'MSLAB'
                numspace = '%0.0f'%(i-hventCounter)
            slabs = data['%s_%s'%(namespace, numspace)]

            z = []
            Ts = []
            Fs = []
            eps = 1e-5
            if i <= hventCounter:
                for j in range(1, int(slabs.values[k]+1)):
                    zb = data['%sYB_%s_%0.0f'%(namespace, numspace, j)].values[k]
                    zt = data['%sYT_%s_%0.0f'%(namespace, numspace, j)].values[k]
                    T = data['%sT_%s_%0.0f'%(namespace, numspace, j)].values[k]
                    F = data['%sF_%s_%0.0f'%(namespace, numspace, j)].values[k]
                    z.append(np.round(zb, decimals=4)+eps)
                    Ts.append(T)
                    Fs.append(F/(zt-zb))
                    z.append(np.round(zt, decimals=4)-eps)
                    Ts.append(T)
                    Fs.append(F/(zt-zb))
                    mdots[j-1, i-1, k] = F
                '''
                zbs = []
                zts = []
                Ts = []
                Fs = []
                z = [data['%sYB_%s_%0.0f'%(namespace, numspace, 1)].values[k]]
                Ts = [max(data['%sT_%s_%0.0f'%(namespace, numspace, 1)].values[k],293.15)]
                Fs = [data['%sF_%s_%0.0f'%(namespace, numspace, 1)].values[k]]
                for j in range(1, int(slabs.values[k]+1)):
                    zb = data['%sYB_%s_%0.0f'%(namespace, numspace, j)].values[k]
                    zt = data['%sYT_%s_%0.0f'%(namespace, numspace, j)].values[k]
                    T = data['%sT_%s_%0.0f'%(namespace, numspace, j)].values[k]
                    F = data['%sF_%s_%0.0f'%(namespace, numspace, j)].values[k]
                    z.append(zt)
                    zbs.append(zb)
                    zts.append(zt)
                    Ts.append(T)
                    Fs.append(F/(zt-zb))
                    mdots[j-1, i-1, k] = F
                '''
                if (k == times.shape[0]-1) and (debug):
                    print('%s_%s'%(namespace, numspace), z, Fs)
                Ts = np.interp(zs, z, Ts)
                Fs = np.interp(zs, z, Fs)
                if (k == times.shape[0]-1) and (debug):
                    print('%s_%s'%(namespace, numspace), zs, Fs)
                Tss[:,i-1,k] = Ts.copy()
                Fss[:,i-1,k] = Fs.copy()
            else:
                z = zs.copy()
                Ts = np.zeros_like(zs)
                Fs = np.zeros_like(zs)
                for j in range(1, int(slabs.values[k]+1)):
                    zb = data['%sYB_%s_%0.0f'%(namespace, numspace, j)].values[k]
                    zt = data['%sYT_%s_%0.0f'%(namespace, numspace, j)].values[k]
                    T = data['%sT_%s_%0.0f'%(namespace, numspace, j)].values[k]
                    F = data['%sF_%s_%0.0f'%(namespace, numspace, j)].values[k]
                    if abs(zb-zt) > 0:
                        Ts = np.zeros_like(zs) + np.nanmean(T)
                        Fs = np.zeros_like(zs) + np.nanmean(F)/(zs.max()-zs.min())
                        mdots[j-1, i-1, k] = np.nanmean(F)
                Tss[:,i-1,k] = Ts.copy()
                Fss[:,i-1,k] = Fs.copy()
                if (k == times.shape[0]-1) and (debug):
                    print('%s_%s'%(namespace, numspace), zs, Fs, Ts)
    
    ventNames = []
    for i in range(1,ventCounter+1):
        if i <= hventCounter:
            ventNames.append('HVENT_%0.0f'%(i))
        else:
            ventNames.append('MVENT_%0.0f'%(i-hventCounter))
    Ts = np.nanmean(Tss[:,:,tind1:tind2], axis=2)
    Fs = np.nanmean(Fss[:,:,tind1:tind2], axis=2)
    mdots = np.nanmean(mdots[:,:,tind1:tind2], axis=2)
    
    tLs = np.zeros((Ts.shape[1],))
    tUs = np.zeros((Ts.shape[1],))
    zInts = np.zeros((Ts.shape[1],))
    zNs = np.zeros((Ts.shape[1],))
    mdotUs = np.zeros((Ts.shape[1],))
    mdotLs = np.zeros((Ts.shape[1],))
    
    #print(Fs)
    fs = 16
    lw = 3
    for i in range(0, Ts.shape[1]):
        
        if debug:
            
            if i+1 <= hventCounter:
                namespace = 'HSLAB'
                numspace = '%0.0f'%(i+1)
            else:
                namespace = 'MSLAB'
                numspace = '%0.0f'%(i-hventCounter+1)
            plt.figure(figsize=(10,6))
            plt.subplot(1,2,1)
            plt.plot(Ts[:,i]-273.15, zs, linewidth=lw)
            plt.xlabel('Temperature ($\mathrm{^{\circ}C}$)', fontsize=fs)
            plt.ylabel('Height (m)', fontsize=fs)
            plt.tick_params(labelsize=fs)
            plt.subplot(1,2,2)
            plt.xlabel('$\mathrm{\dot{m}/z}$ (kg/s-m)', fontsize=fs)
            plt.plot(Fs[:,i], zs, linewidth=lw)
            plt.suptitle('%s_%s'%(namespace, numspace), fontsize=fs)
            plt.tick_params(labelsize=fs)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        
        zInt, tL, tU = getTwoZoneT(zs[::-1], Ts[::-1,i]-273.15)
        tLs[i] = tL
        tUs[i] = tU
        zInts[i] = zInt
        try:
            doorH = zs[abs(Fs[:,i]) > 0].max()
        except:
            doorH = zs.max()
        zN = getNeutralPlane(zs[::-1], Fs[::-1,i], doorH)
        zNs[i] = zN
        
        mdot = mdots[:,i]
        mdotin = np.sum(mdot[mdot > 0])
        mdotout = np.sum(mdot[mdot < 0])
        if np.median(np.where(mdot[mdot > 0])[0]) > np.median(np.where(mdot[mdot < 0])[0]):
            mdotUs[i] = mdotin
            mdotLs[i] = mdotout
        else:
            mdotUs[i] = mdotout
            mdotLs[i] = mdotin
    ventCounter = 0
    ventData = defaultdict(bool)
    for tL, tU, zInt, zN, mdotU, mdotL, vN in zip(tLs, tUs, zInts, zNs, mdotUs, mdotLs, ventNames):
        ventCounter += 1
        vent = defaultdict(bool)
        vent['ID'] = vN
        if 'MVENT' in vN: vent['TYPE'] = 'MECHANICAL'
        if 'HVENT' in vN: vent['TYPE'] = 'WALL'
        (vent['tL'], vent['tU']) = (tL, tU)
        (vent['zInt'], vent['zN']) = (zInt, zN)
        (vent['mdotU'], vent['mdotL']) = (mdotU, mdotL)
        ventData[vent['ID']] = vent
    
    return ventData

def getTwoZoneT(dataZ, dataT):
    dataT = dataT+273
    tL = dataT[-1]
    H = dataZ[0]
    I1 = np.trapz(dataT[::-1],dataZ[::-1])
    I2 = np.trapz(1/dataT[::-1],dataZ[::-1])
    if (I1+I2*tL**2-2*tL*H) == 0:
        zInt = 0.0
    else:
        zInt = tL*(I1*I2-H**2)/(I1+I2*tL**2-2*tL*H)
    zInt = np.nanmax([zInt, 0.0])
    
    zU = np.linspace(zInt,H,num=50)
    tU_tmp = np.interp(zU,dataZ[::-1],dataT[::-1])
    tU = np.trapz(tU_tmp,zU)/(H-zInt)
    
    tU = tU - 273
    tL = tL - 273
    return zInt, tL, tU

def getNeutralPlane(dataZ,dataV,doorH):
    dataV_tmp = dataV.copy()
    #dataV_tmp[dataZ>doorH-0.1] = 9001
    dataV_tmp[dataZ<0.1] = 9001
    dataVSign = np.sign(dataV)
    dataVDiff = np.diff(dataVSign)/2
    ind = np.where(dataVDiff != 0)[0]
    if len(ind) == 0:
        zN = 0
    else:
        ind = ind[-1]
        ind1 = max(ind-2, 0)
        ind2 = min(ind+2, dataZ.shape[0])
        extractDataV = dataV[ind1:ind2]
        extractDataZ = dataZ[ind1:ind2]
        
        try:
            flip = True if extractDataV[0] > extractDataV[-1] else False
        except IndexError:
            return 0
        zN = np.interp([0],extractDataV[::-1],extractDataZ[::-1])[0] if flip else np.interp([0],extractDataV,extractDataZ)[0]
        zN = min(zN, dataZ.max())
        zN = max(zN, dataZ.min())
    return zN
