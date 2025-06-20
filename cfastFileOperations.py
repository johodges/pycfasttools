#-----------------------------------------------------------------------
# Copyright (C) 2020, All rights reserved
#
# Jonathan L. Hodges
#
#-----------------------------------------------------------------------
#=======================================================================
# 
# DESCRIPTION:
# This software is part of a python library to assist in developing and
# analyzing simulation results from Consolidated Model of Fire and Smoke
# Transport (CFAST).
# CFAST is an open source software package developed by NIST. The source
# code is available at: https://github.com/firemodels/cfast
#
# EXAMPLES:
# See the examples subroutine for example operation.
#
#=======================================================================
# # IMPORTS
#=======================================================================

from collections import defaultdict
import numpy as np

def dictMerge(a, b, path=None):
    "merges b into a"
    if path is None: path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                dictMerge(a[key], b[key], path + [str(key)])
            elif a[key] == b[key]:
                pass
            else:
                a[key] = b[key]
        else:
            a[key] = b[key]
    return a

class cfastFileOperations(object):
    def __init__(self):
        self.head = defaultdict(bool)
        self.time = defaultdict(bool)
        self.init = defaultdict(bool)
        self.misc = defaultdict(bool)
        self.matls = defaultdict(bool)
        self.comps = defaultdict(bool)
        self.vents = defaultdict(bool)
        self.chems = defaultdict(bool)
        self.tabls = defaultdict(bool)
        self.fires = defaultdict(bool)
        self.slcfs = defaultdict(bool)
        self.devcs = defaultdict(bool)
        
        self.compUnknownCounter = 0
        self.ventUnknownCounter = 0
        self.hventCounter = 1
        self.mventCounter = 1
        self.slcfCounter = 0
    
    def addHEAD(self, title, version=7300):
        self.head['TITLE'] = title
        self.head['VERSION'] = "%0.0f"%(version)
    
    def addTIME(self, simTime, printTime=5, smokeviewTime=5, spreadsheetTime=5):
        self.time['SIMULATION'] = simTime
        self.time['PRINT'] = printTime
        self.time['SMOKEVIEW'] = smokeviewTime
        self.time['SPREADSHEET'] = spreadsheetTime
    
    def addINIT(self, pressure=101325, relativeHumidity=40, interiorTemperature=20, exteriorTemperature=20):
        self.init['PRESSURE'] = pressure
        self.init['RELATIVE_HUMIDITY'] = relativeHumidity
        self.init['INTERIOR_TEMPERATURE'] = interiorTemperature
        self.init['EXTERIOR_TEMPERATURE'] = exteriorTemperature
        
    def addMISC(self, adiabatic='FALSE', MAX_TIME_STEP=0.1, LOWER_OXYGEN_LIMIT=0.127):
        self.misc['ADIABATIC'] = adiabatic
        self.misc['MAX_TIME_STEP'] = MAX_TIME_STEP
        self.misc['LOWER_OXYGEN_LIMIT'] = LOWER_OXYGEN_LIMIT
        
    def addCOMP(self, cid, x, y, z, dx, dy, dz, GRID=[50, 50, 50],
                CEILING_MATL_ID='OFF', WALL_MATL_ID='OFF', FLOOR_MATL_ID='OFF',
                SHAFT=None):
        comp = defaultdict(bool)
        comp['ID'] = cid
        comp['ORIGIN'] = [x, y, z]
        comp['DEPTH'] = dx
        comp['WIDTH'] = dy
        comp['HEIGHT'] = dz
        comp['GRID'] = GRID
        comp['CEILING_MATL_ID'] = CEILING_MATL_ID
        comp['WALL_MATL_ID'] = WALL_MATL_ID
        comp['FLOOR_MATL_ID'] = FLOOR_MATL_ID
        comp['sides'] = defaultdict(bool)
        comp['sides']['east'] = defaultdict(bool)
        comp['sides']['west'] = defaultdict(bool)
        comp['sides']['north'] = defaultdict(bool)
        comp['sides']['south'] = defaultdict(bool)
        if SHAFT is not None:
            comp['SHAFT'] = SHAFT
        self.comps[cid] = comp
    
    def addVENT(self, vid, vtype, comp1, comp2, face, offset, bottom, width, top, areas=None):
        vent = defaultdict(bool)
        vent['ID'] = vid
        vent['TYPE'] = vtype
        vent['COMP_IDS'] = [comp1, comp2]
        vent['FACE'] = face
        vent['OFFSET'] = offset
        vent['BOTTOM'] = bottom
        vent['TOP'] = top
        if vtype == 'WALL':
            vent['VENTNAME'] = 'HVENT_%0.0f'%(self.hventCounter)
            vent['WIDTH'] = width
            self.hventCounter += 1
        elif vtype == 'MECHANICAL':
            vent['VENTNAME'] = 'MVENT_%0.0f'%(self.mventCounter)
            vent['AREAS'] = areas
            self.mventCounter += 1
        self.vents[vid] = vent
        
    def addMVENT(self, vid, vtype, comp1, comp2, areas, heights, flow, cutoffs, offsets, filterTime, filterEfficiency, orientations):
        vent = defaultdict(bool)
        vent['ID'] = vid
        vent['TYPE'] = vtype
        vent['COMP_IDS'] = [comp1, comp2]
        vent['AREAS'] = areas
        vent['HEIGHTS'] = heights
        vent['FLOW'] = flow
        vent['CUTOFFS'] = cutoffs
        vent['OFFSETS'] = offsets
        vent['FILTER_TIME'] = filterTime
        vent['FILTER_EFFICIENCY'] = filterEfficiency
        vent['ORIENTATIONS'] = orientations
        self.vents[vid] = vent
    
    def addFIRE(self, mID, cID, fID, LOCATION):
        fire = defaultdict(bool)
        fire['ID'] = mID
        fire['COMP_ID'] = cID
        fire['FIRE_ID'] = fID
        fire['LOCATION'] = LOCATION
        self.fires[mID] = fire        
    
    def addCHEM(self, fID, c, ch, h, n, o, hc, xr):
        chem = defaultdict(bool)
        chem['ID'] = fID
        chem['CARBON'] = c
        chem['CHLORINE'] = ch
        chem['HYDROGEN'] = h
        chem['OXYGEN'] = o
        chem['HEAT_OF_COMBUSTION'] = hc
        chem['RADIATIVE_FRACTION'] = xr
        self.chems[fID] = chem
    
    def addTABL(self, fID, labels, datas):
        tabl = defaultdict(bool)
        tabl['LABELS'] = labels
        tabl['DATA'] = datas
        tabl['ID'] = fID
        self.tabls[fID] = tabl
    
    def addMATL(self, ID, MATERIAL, CONDUCTIVITY, DENSITY, SPECIFIC_HEAT, THICKNESS, EMISSIVITY=0.9):
        matl = defaultdict(bool)
        matl['ID'] = ID
        matl['MATERIAL'] = MATERIAL
        matl['CONDUCTIVITY'] = CONDUCTIVITY
        matl['DENSITY'] = DENSITY
        matl['SPECIFIC_HEAT'] = SPECIFIC_HEAT
        matl['THICKNESS'] = THICKNESS
        matl['EMISSIVITY'] = EMISSIVITY
        self.matls[matl['ID']] = matl
    
    def importFile(self, file=None, text=None, textList=None):
        if file != None:
            with open(file, 'r') as f:
                textCFAST = f.read()
        elif text != None:
            textCFAST = text
        elif textList != None:
            textCFAST = '\n'.join(textList)
        else:
            print("Either file, text, or textList must be specified.")
            return ''
        lines = self.makeCFASTLines(textCFAST)
        keys = self.makeCFASTkeys(lines)
        self.lines = lines
        self.keys = keys
        self.parseCFASTKeys(keys)
        self.file = file
        self.text = textCFAST
        
    def makeCFASTLines(self, text):
        lines = [x.split('/')[0] for x in text.split("&")[1:]]
        for j in range(0, len(lines)):
            line2 = str(lines[j])
            line2 = line2.replace(' =','=').replace('= ','=').replace('\n',',').replace(' ,',',').replace(', ',',') #.replace(' ',',')
            line2 = line2.replace(',,',',').replace(',,',',').replace(',,',',').replace(',,',',').replace(',,',',')
            line2 = "%s,"%(line2) if line2[-1] != ',' else line2
            
            tmp = line2.split("'")
            if len(tmp) > 0:
                line3 = '%s'%(tmp[0])
                if 'TABL' in line2 and False:
                    print(line2)
                else:
                    fields = []
                    for i in range(0, len(tmp)):
                        if i % 2 == 0:
                            field = tmp[i].replace(' ', ',').replace(',,',',').replace(',,',',').replace(',,',',').replace(',,',',').replace(',,',',')
                            if field != ',':
                                fields.append(field)
                        else:
                            fields.append("'%s'"%(tmp[i]))
                    line3 = '%s'%(fields[0])
                    for i in range(1, len(fields)):
                        line3 = "%s%s,"%(line3, fields[i])
                    line3 = line3.replace('=,','=').replace(',,',',').replace(',,',',').replace(',,',',').replace(',,',',').replace(',,',',')
                    line2 = line3
            lines[j] = line2
        return lines
    
    def makeCFASTkeys(self, lines):
        keys = [self.splitLineIntoKeys(line) for line in lines]
        return keys
    
    def getLineType(self, key):
        lineType = key[0]
        return lineType
    
    def splitLineIntoKeys(self, line2):
        line = line2.replace('\n',',').replace(',,',',').replace(',,',',').replace(',,',',')
        line = line.replace('  ',' ').replace('  ',' ').replace('  ',' ').replace('  ',' ').replace('  ',' ').replace('  ',' ').replace('  ',' ').replace('  ',' ')
        
        keys = line.split(',')
        updatedKeys = [keys[0]]
        txt = ''
        for i in range(1,len(keys)):
            if '=' in keys[i]:
                updatedKeys.append(txt)
                txt = keys[i]
            else:
                txt = ','.join([txt,keys[i]])
        updatedKeys.append(txt)
        while ("" in updatedKeys):
            updatedKeys.remove("")
        return updatedKeys
    
    def parseCFASTKeys(self, keys):
        for key in keys:
            lineType = self.getLineType(key)
            if lineType == 'HEAD': self.parseHEAD(key)
            if lineType == 'TIME': self.parseTIME(key)
            if lineType == 'INIT': self.parseINIT(key)
            if lineType == 'MISC': self.parseMISC(key)
            if lineType == 'MATL': self.parseMATL(key)
            if lineType == 'COMP': self.parseCOMP(key)
            if lineType == 'VENT': self.parseVENT(key)
            if lineType == 'CHEM': self.parseCHEM(key)
            if lineType == 'TABL': self.parseTABL(key)
            if lineType == 'FIRE': self.parseFIRE(key)
            if lineType == 'SLCF': self.parseSLCF(key)
    
    def parseHEAD(self, keys):
        head = defaultdict(bool)
        for key in keys:
            if "VERSION" in key: head['VERSION'] = key.split("VERSION")[1].replace('=','').replace(' ','')
            if "TITLE" in key: head['TITLE'] = key.split("TITLE")[1].split("'")[1]
        self.head = dictMerge(self.head, head)
        
    def parseTIME(self, keys):
        time = defaultdict(bool)
        for key in keys:
            if 'SIMULATION' in key: time['SIMULATION'] = float(key.split('=')[1].split(',')[0])
            if 'PRINT' in key: time['PRINT'] = float(key.split('=')[1].split(',')[0])
            if 'SMOKEVIEW' in key: time['SMOKEVIEW'] = float(key.split('=')[1].split(',')[0])
            if 'SPREADSHEET' in key: time['SPREADSHEET'] = float(key.split('=')[1].split(',')[0])
        self.time = dictMerge(self.time, time)
        
    def parseMISC(self, keys):
        misc = defaultdict(bool)
        for key in keys:
            if 'ADIABATIC' in key: misc['ADIABATIC'] = key.split('.')[1]
            if 'MAX_TIME_STEP' in key: misc['MAX_TIME_STEP'] = float(key.split('=')[1].replace(',',''))
            if 'LOWER_OXYGEN_LIMIT' in key: misc['LOWER_OXYGEN_LIMIT'] = float(key.split('=')[1].replace(',',''))
        self.misc = dictMerge(self.misc, misc)
        
    def parseMATL(self, keys):
        matls = defaultdict(bool)
        for key in keys:
            if 'MATERIAL' in key:
                matls['MATERIAL'] = key.split('=')[1].replace('"','').replace("'",'')
            elif 'CONDUCTIVITY' in key:
                matls['CONDUCTIVITY'] = float(key.split('=')[1].replace(',',''))
            elif 'DENSITY' in key:
                matls['DENSITY'] = float(key.split('=')[1].replace(',',''))
            elif 'SPECIFIC_HEAT' in key:
                matls['SPECIFIC_HEAT'] = float(key.split('=')[1].replace(',',''))
            elif 'THICKNESS' in key:
                matls['THICKNESS'] = float(key.split('=')[1].replace(',',''))
            elif 'EMISSIVITY' in key:
                matls['EMISSIVITY'] = float(key.split('=')[1].replace(',',''))
            elif 'ID' in key:
                matls['ID'] = key.split('=')[1].replace('"','').replace("'", '')
        self.matls[matls['ID']] = matls
        
    
    def parseINIT(self, keys):
        init = defaultdict(bool)
        for key in keys:
            if 'PRESSURE' in key: init['PRESSURE'] = float(key.split('=')[1].split(',')[0])
            if 'RELATIVE_HUMIDITY' in key: init['RELATIVE_HUMIDITY'] = float(key.split('=')[1].split(',')[0])
            if 'INTERIOR_TEMPERATURE' in key: init['INTERIOR_TEMPERATURE'] = float(key.split('=')[1].split(',')[0])
            if 'EXTERIOR_TEMPERATURE' in key: init['EXTERIOR_TEMPERATURE'] = float(key.split('=')[1].split(',')[0])
        self.init = init
    
    def parseCOMP(self, keys):
        comp = defaultdict(bool)
        for key in keys:
            if key[-1] == ',': key = key[:-1]
            if ('WALL_MATL_ID' in key): comp['WALL_MATL_ID'] = key.split('=')[1].split("'")[1]
            elif ('FLOOR_MATL_ID' in key): comp['FLOOR_MATL_ID'] = key.split('=')[1].split("'")[1]
            elif ('CEILING_MATL_ID' in key): comp['CEILING_MATL_ID'] = key.split('=')[1].split("'")[1]
            elif ('GRID' in key): comp['GRID'] = [int(x) for x in key.split('=')[1].split(',')]
            elif ('ORIGIN' in key): comp['ORIGIN'] = [float(x) for x in key.split('=')[1].split(',')]
            elif ('WIDTH' in key): comp['WIDTH'] = float(key.split('=')[1].strip())
            elif ('DEPTH' in key): comp['DEPTH'] = float(key.split('=')[1].strip())
            elif ('HEIGHT' in key): comp['HEIGHT'] = float(key.split('=')[1].strip())
            elif ('ID' in key): comp['ID'] = key.split('=')[1].split("'")[1]
        comp['sides'] = defaultdict(bool)
        comp['sides']['east'] = defaultdict(bool)
        comp['sides']['west'] = defaultdict(bool)
        comp['sides']['north'] = defaultdict(bool)
        comp['sides']['south'] = defaultdict(bool)
        if comp['ID']: self.comps[comp['ID']] = comp
        else:
            self.comps['UNKNOWN-COMP-%04.0f'%(self.compUnknownCounter)] = comp
            self.compUnknownCounter += 1
            
    def parseVENT(self, keys):
        vent = defaultdict(bool)
        for key in keys:
            if ('TYPE' in key): vent['TYPE'] = key.split("'")[1]
            elif ('COMP_IDS' in key):
                tmp = key.split('=')[1].split("'")
                tmp = [tmp[1], tmp[3]]
                vent['COMP_IDS'] = tmp
            elif ('FACE' in key): vent['FACE'] = key.split("'")[1]
            elif ('OFFSETS' in key):
                tmp = key.split('=')[1].split(',')
                while ("" in tmp):
                    tmp.remove("")
                vent['OFFSETS'] = [float(x) for x in tmp]
            elif ('OFFSET' in key): vent['OFFSET'] = float(key.split('=')[1].replace(',',''))
            elif ('BOTTOM' in key): vent['BOTTOM'] = float(key.split('=')[1].replace(',',''))
            elif ('WIDTH' in key): vent['WIDTH'] = float(key.split('=')[1].replace(',',''))
            elif ('AREAS' in key): vent['AREAS'] = [float(x) for x in key.split('=')[1].split(',')]
            elif ('HEIGHTS' in key): vent['HEIGHTS'] = [float(x) for x in key.split('=')[1].split(',')]
            elif ('CUTOFFS' in key): vent['CUTOFFS'] = [float(x) for x in key.split('=')[1].split(',')]
            elif ('FILTER_TIME' in key): vent['FILTER_TIME'] = float(key.split('=')[1].replace(',',''))
            elif ('FILTER_EFFICIENCY' in key): vent['FILTER_EFFICIENCY'] = float(key.split('=')[1].replace(',',''))
            elif ('TOP' in key): vent['TOP'] = float(key.split('=')[1].replace(',',''))
            elif ('ID' in key): vent['ID'] = key.split("'")[1]
        
        if vent['TYPE'] == 'WALL':
            vent['VENTNAME'] = 'HVENT_%0.0f'%(self.hventCounter)
            self.hventCounter += 1
        elif vent['TYPE'] == 'MECHANICAL':
            vent['VENTNAME'] = 'MVENT_%0.0f'%(self.mventCounter)
            self.mventCounter += 1
        
        if vent['ID']: self.vents[vent['ID']] = vent
        else:
            self.vents['UNKNOWN-VENT-%04.0f'%(self.ventUnknownCounter)] = vent
            self.ventUnknownCounter += 1
    
    def parseCHEM(self, keys):
        chem = defaultdict(bool)
        for key in keys:
            if 'CARBON' in key: chem['CARBON'] = float(key.split('=')[1].replace(',',''))
            elif 'CHLORINE' in key: chem['CHLORINE'] = float(key.split('=')[1].replace(',',''))
            elif 'HYDROGEN' in key: chem['HYDROGEN'] = float(key.split('=')[1].replace(',',''))
            elif 'NITROGEN' in key: chem['NITROGEN'] = float(key.split('=')[1].replace(',',''))
            elif 'OXYGEN' in key: chem['OXYGEN'] = float(key.split('=')[1].replace(',',''))
            elif 'HEAT_OF_COMBUSTION' in key: chem['HEAT_OF_COMBUSTION'] = float(key.split('=')[1].replace(',',''))
            elif 'RADIATIVE_FRACTION' in key: chem['RADIATIVE_FRACTION'] = float(key.split('=')[1].replace(',',''))
            elif 'ID' in key: chem['ID'] = key.split("'")[1]
        self.chems[chem['ID']] = chem
        
    def parseTABL(self, keys):
        tabl = defaultdict(bool)
        for key in keys:
            if 'LABELS' in key:
                tmp = key.split('=')[1].split(",")
                while ("" in tmp):
                    tmp.remove("")
                tmp = [x.split("'")[1] for x in tmp]
                tabl['LABELS'] = tmp
            elif 'ID' in key: tabl['ID'] = key.split("'")[1]
            elif 'DATA' in key:
                tmp = key.split('=')[1].split(',')
                while ("" in tmp):
                    tmp.remove("")
                tmp = np.array([float(x) for x in tmp])
                tmp = np.reshape(tmp, (1,tmp.shape[0]))
                tabl['DATA'] = tmp
        if self.tabls[tabl['ID']]:
            if self.tabls[tabl['ID']]['DATA'] is False:
                self.tabls[tabl['ID']]['DATA'] = tabl['DATA']
            else:
                data = self.tabls[tabl['ID']]['DATA']
                data = np.append(data, tabl['DATA'], axis=0)
                self.tabls[tabl['ID']]['DATA'] = data                
        else:
            self.tabls[tabl['ID']] = tabl
    
    def parseFIRE(self, keys):
        fire = defaultdict(bool)
        for key in keys:
            if key[-1] == ',': key = key[:-1]
            if ('COMP_ID' in key): fire['COMP_ID'] = key.split('=')[1].split("'")[1]
            elif ('FIRE_ID' in key): fire['FIRE_ID'] = key.split("'")[1]
            elif ('LOCATION' in key):
                tmp = key.split('=')[1].split(',')
                while ("" in tmp):
                    tmp.remove("")
                fire['LOCATION'] = [float(x) for x in tmp]
            elif ("ID" in key): fire['ID'] = key.split('=')[1].split("'")[1]
        self.fires[fire['ID']] = fire
    
    def parseSLCF(self, keys):
        slcf = defaultdict(bool)
        
        for key in keys:
            if 'COMP_ID' in key:
                slcf['COMP_ID'] = key.split('=')[1].replace('"','').replace("'",'')
            elif 'DOMAIN' in key:
                slcf['DOMAIN'] = key.split('=')[1].replace('"','').replace("'",'')
            elif 'POSITION' in key:
                slcf['POSITION'] = float(key.split('=')[1].replace(',',''))
            elif 'PLANE' in key:
                slcf['PLANE'] = key.split('=')[1].replace('"','').replace("'",'').replace(',','')
            '''
            if 'CARBON' in key: chem['CARBON'] = float(key.split('=')[1].replace(',',''))
            elif 'CHLORINE' in key: chem['CHLORINE'] = float(key.split('=')[1].replace(',',''))
            elif 'HYDROGEN' in key: chem['HYDROGEN'] = float(key.split('=')[1].replace(',',''))
            elif 'NITROGEN' in key: chem['NITROGEN'] = float(key.split('=')[1].replace(',',''))
            elif 'OXYGEN' in key: chem['OXYGEN'] = float(key.split('=')[1].replace(',',''))
            elif 'HEAT_OF_COMBUSTION' in key: chem['HEAT_OF_COMBUSTION'] = float(key.split('=')[1].replace(',',''))
            elif 'RADIATIVE_FRACTION' in key: chem['RADIATIVE_FRACTION'] = float(key.split('=')[1].replace(',',''))
            elif 'ID' in key: chem['ID'] = key.split("'")[1]
            '''
        #self.chems[chem['ID']] = chem
        self.slcfs['SLCF-%03.0f'%(self.slcfCounter)] = slcf
        self.slcfCounter += 1
    
    def buildHEAD(self, text=''):
        text = "%s&HEAD VERSION = %s, TITLE = '%s' /\n"%(text, self.head['VERSION'], self.head['TITLE'])
        return text
    
    def buildTIME(self, text=''):
        text = "%s&TIME SIMULATION = %0.4f PRINT = %0.4f SMOKEVIEW = %0.4f SPREADSHEET = %0.4f /\n"%(text, self.time['SIMULATION'], self.time['PRINT'], self.time['SMOKEVIEW'], self.time['SPREADSHEET'])
        return text
    
    def buildINIT(self, text=''):
        text = "%s&INIT PRESSURE = %0.4f RELATIVE_HUMIDITY = %0.4f"%(text, self.init['PRESSURE'], self.init['RELATIVE_HUMIDITY'])
        text = "%s INTERIOR_TEMPERATURE = %0.4f EXTERIOR_TEMPERATURE = %0.4f /\n"%(text, self.init['INTERIOR_TEMPERATURE'], self.init['EXTERIOR_TEMPERATURE'])
        return text
    
    def buildMISC(self, text=''):
        text = "%s&MISC ADIABATIC = .%s. MAX_TIME_STEP = %0.4f LOWER_OXYGEN_LIMIT = %0.4f /\n"%(text, self.misc['ADIABATIC'], self.misc['MAX_TIME_STEP'], self.misc['LOWER_OXYGEN_LIMIT'])
        return text

    def buildMATLS(self, text=''):
        for key in list(self.matls.keys()):
            text = self.buildMATL(self.matls[key], text=text)
        return text    

    def buildMATL(self, matl, text=''):
        text = "%s&MATL ID = '%s' MATERIAL = '%s' \n"%(text, matl['ID'], matl['MATERIAL'])
        text = "%s    CONDUCTIVITY = %0.4f DENSITY = %0.4f SPECIFIC_HEAT = %0.4f THICKNESS = %0.4f EMISSIVITY = %0.4f /\n"%(text, matl['CONDUCTIVITY'], matl['DENSITY'], matl["SPECIFIC_HEAT"], matl['THICKNESS'], matl['EMISSIVITY'])
        return text
    
    def buildSLCFS(self, text=''):
        for key in list(self.slcfs.keys()):
            text = self.buildSLCF(self.slcfs[key], text=text)
        return text
    
    def buildSLCF(self, slcf, text=''):
        text = "%s&SLCF "%(text)
        if slcf['COMP_ID'] is not False:
            text = "%s COMP_ID = '%s'"%(text, slcf['COMP_ID'])
        if slcf['DOMAIN'] is not False:
            text = "%s DOMAIN = '%s'"%(text, slcf['DOMAIN'])
        if slcf['POSITION'] is not False:
            text = "%s POSITION = %0.4f"%(text, slcf['POSITION'])
        if slcf['PLANE'] is not False:
            text = "%s PLANE = '%s'"%(text, slcf['PLANE'])
        text = "%s /\n"%(text)
        return text
    
    def buildCOMPS(self, text=''):
        for key in list(self.comps.keys()):
            text = self.buildCOMP(self.comps[key], text=text)
        return text
    
    def buildCOMP(self, comp, text=''):
        text = "%s&COMP ID = '%s' DEPTH = %0.4f HEIGHT = %0.4f WIDTH = %0.4f\n"%(text, comp['ID'], comp['DEPTH'], comp['HEIGHT'], comp['WIDTH'])
        text = "%s      CEILING_MATL_ID = '%s' WALL_MATL_ID = '%s' FLOOR_MATL_ID = '%s'\n"%(text, comp['CEILING_MATL_ID'], comp['WALL_MATL_ID'], comp['FLOOR_MATL_ID'])
        text = "%s      ORIGIN = %0.4f, %0.4f, %0.4f"%(text, comp['ORIGIN'][0], comp['ORIGIN'][1], comp['ORIGIN'][2])
        if comp['SHAFT'] is not False:
            if (comp['SHAFT'] == True) or ('TRUE' in comp['SHAFT']):
                text = "%s     SHAFT = .TRUE. "%(text)
        text = "%s GRID = %0.0f, %0.0f, %0.0f /\n"%(text, comp['GRID'][0], comp['GRID'][1], comp['GRID'][2])
        return text
    
    def buildVENTS(self, text=''):
        for key in list(self.vents.keys()):
            text = self.buildVENT(self.vents[key], text=text)
        return text
    
    def buildVENT(self, vent, text=''):
        if vent['TYPE'] == 'WALL':
            text = "%s&VENT ID = '%s' TYPE = '%s' COMP_IDS = '%s', '%s'\n"%(text, vent['ID'], vent['TYPE'], vent['COMP_IDS'][0], vent['COMP_IDS'][1])
            text = "%s      TOP = %0.4f, BOTTOM = %0.4f, WIDTH = %0.4f\n"%(text, vent['TOP'], vent['BOTTOM'], vent['WIDTH'])
            text = "%s      FACE = '%s' OFFSET = %0.4f /\n"%(text, vent['FACE'], vent['OFFSET'])
        elif vent['TYPE'] == 'MECHANICAL':
            text = "%s&VENT ID = '%s' TYPE = '%s' COMP_IDS = '%s', '%s'\n"%(text, vent['ID'], vent['TYPE'], vent['COMP_IDS'][0], vent['COMP_IDS'][1])
            text = "%s      AREAS = %0.4f, %0.4f HEIGHTS = %0.4f, %0.4f ORIENTATIONS = %s, %s"%(text, vent['AREAS'][0], vent['AREAS'][1], vent['HEIGHTS'][0], vent['HEIGHTS'][1], vent['ORIENTATIONS'][0], vent['ORIENTATIONS'][1])
            text = "%s      FLOW = %0.4f CUTOFFS = %0.4f, %0.4f"%(text, vent['FLOW'], vent['CUTOFFS'][0], vent['CUTOFFS'][1])
            text = "%s      OFFSETS = %0.4f, %0.4f"%(text, vent['OFFSETS'][0], vent['OFFSETS'][1])
            text = "%s      FILTER_TIME = %0.4f FILTER_EFFICIENCY = %0.4f /\n"%(text, vent['FILTER_TIME'], vent['FILTER_EFFICIENCY'])
        return text
    
    def buildCHEMS(self, text=''):
        for key in list(self.chems.keys()):
            text = self.buildCHEM(self.chems[key], text=text)
        return text
    
    def buildCHEM(self, chem, text=''):
        text = "%s&CHEM ID = '%s' CARBON = %0.4f CHLORINE = %0.4f"%(text, chem['ID'], chem['CARBON'], chem['CHLORINE'])
        text = "%s HYDROGEN = %0.4f NITROGEN = %0.4f OXYGEN = %0.4f"%(text, chem['HYDROGEN'], chem['NITROGEN'], chem['OXYGEN'])
        text = "%s HEAT_OF_COMBUSTION = %0.4f RADIATIVE_FRACTION = %0.4f /\n"%(text, chem['HEAT_OF_COMBUSTION'], chem['RADIATIVE_FRACTION'])
        return text
    
    def buildTABLS(self, text=''):
        for key in list(self.tabls.keys()):
            text = self.buildTABL(self.tabls[key], text=text)
        return text
    
    def buildTABL(self, tabl, text=''):
        tid = tabl['ID']
        datas = tabl['DATA']
        text = "%s&TABL ID = '%s' LABELS ="%(text, tid)
        for l in tabl['LABELS']:
            text = "%s '%s',"%(text, l)
        text = "%s /\n"%(text)
        for i in range(0, datas.shape[0]):
            data = datas[i, :]
            text = "%s&TABL ID = '%s', DATA ="%(text, tid)
            for d in data:
                text = "%s %0.4f,"%(text, d)
            text = "%s /\n"%(text)
        return text
    
    def buildFIRES(self, text=''):
        for key in list(self.fires.keys()):
            text = self.buildFIRE(self.fires[key], text=text)
        return text
    
    def buildFIRE(self, fire, text=''):
        text = "%s&FIRE ID = '%s' COMP_ID = '%s',"%(text, fire['ID'], fire['COMP_ID'])
        text = "%s FIRE_ID = '%s' LOCATION = %0.4f, %0.4f /\n"%(text, fire['FIRE_ID'], fire['LOCATION'][0], fire['LOCATION'][1])
        return text
    
    def buildDEVCS(self, text=''):
        for key in list(self.devcs.keys()):
            text = self.buildDEVC(self.devcs[key], text=text)
        return text
    
    def buildDEVC(self, devc, text=''):
        text = "%s&DEVC"%(text)
        if devc['ID'] is not False:
            text = "%s ID = '%s'"%(text, devc['ID'])
        if devc['COMP_ID'] is not False:
            text = "%s COMP_ID = '%s'"%(text, devc['COMP_ID'])
        if devc['LOCATION'] is not False:
            x, y, z = devc['LOCATION']
            text = "%s LOCATION = %0.4f, %0.4f, %0.4f"%(text, x, y, z)
        if devc['TYPE'] is not False:
            text = "%s TYPE = '%s'"%(text, devc['TYPE'])
        if devc['SETPOINTS'] is not False:
            text = "%s SETPOINTS = %0.4f, %0.4f"%(text, devc['SETPOINTS'][0], devc['SETPOINTS'][1])
        if devc['SETPOINT'] is not False:
            text = "%s SETPOINT = %0.4f"%(text, devc['SETPOINT'])
        if devc['RTI'] is not False:
            text = "%s RTI = %0.4f"%(text, devc['RTI'])
        if devc['SPRAY_DENSITY'] is not False:
            text = "%s SPRAY_DENSITY = %0.8f"%(text, devc['SPRAY_DENSITY'])
        text = "%s /\n"%(text)
        return text
    
    def __repr__(self):
        text = self.buildHEAD()
        text = self.buildTIME(text)
        text = self.buildINIT(text)
        text = self.buildMISC(text)
        text = self.buildMATLS(text)
        text = self.buildCOMPS(text)
        text = self.buildVENTS(text)
        text = self.buildCHEMS(text)
        text = self.buildTABLS(text)
        text = self.buildFIRES(text)
        text = self.buildSLCFS(text)
        text = self.buildDEVCS(text)
        return text
    
    def __str__(self):
        return self.__repr__()
    
if __name__ == "__main__":
    file = "E://projects//mineFireLearning//cfastSimulations//inputFiles//mineFire_01007_cfast.in"
    cfile = cfastFileOperations()
    cfile.importFile(file=file)
