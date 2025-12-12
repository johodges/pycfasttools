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

import sys

sys.path.append('E:\\projects\\customPythonModules\\')

import cfastTools as cfast
import numpy as np

if __name__ == "__main__":
    
    inputFile = "examples\\baseline.in"
    output_directory = "generated\\"
    
    basename = "VENTLESS_FAST_10MW_NEAR"
    floorAreas = np.linspace(10000, 100000, 10)
    ceilingHeights = np.linspace(4.5, 6.0, 2)
   
    for floorArea in floorAreas:
        for HEIGHT in ceilingHeights:
            file = cfast.cfastFileOperations()
            file.importFile(inputFile)
            
            #assert False, "Stopped"
            
            DEPTH = (floorArea/1)**0.5
            WIDTH = DEPTH*1
           
            file.comps['Comp 1']['DEPTH'] = DEPTH
            file.comps['Comp 1']['WIDTH'] = WIDTH
            file.comps['Comp 1']['HEIGHT'] = HEIGHT
            
            for key in list(file.slcfs.keys()):
                if file.slcfs[key]['PLANE'] == 'X':
                    file.slcfs[key]['POSITION'] = file.comps['Comp 1']['DEPTH']/2
                elif file.slcfs[key]['PLANE'] == 'Y':
                    file.slcfs[key]['POSITION'] = file.comps['Comp 1']['WIDTH']/2
                elif file.slcfs[key]['PLANE'] == 'Z':
                    file.slcfs[key]['POSITION'] = 1.8
            
            newfilename = "%s_%0.0f_%0.0f_%0.1f"%(basename, DEPTH, WIDTH, HEIGHT)
            newfilename = newfilename.replace('.','-')
           
            with open("%s%s.in"%(output_directory, newfilename), 'w') as f:
                txt = file.__repr__()
                f.write(txt)
                
                