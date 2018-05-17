import numpy as np
import pandas as pd
from scipy.stats.mstats import gmean

def maxscore(arr):
    if len(arr)>0:
        return max(arr)
    else:
        return 0
    
def minscore(arr):
    if len(arr)>0:
        return min(arr)
    else:
        return 0
    
def avgscore(arr):
    if len(arr)>0:
        return gmean(arr)
    else:
        return 0
        
        
# Function returns true if the point is within the volume specified by arr.
def CheckBorderTPC(x,y,z,array= [[0,0],[0,0],[0,0]]):
    detectorx   =256.35     # In cm
    detectory   =116.5      # Symmetric around 0     
    detectorz   =1036.8
    if (0+array[0][0]) < x < (detectorx-array[0][1]):
            if (-detectory+array[1][0])< y < (detectory-array[1][1]):
                    if (0+array[2][0]) < z < (detectorz-array[2][1]):
                        return True
    return False


# Return true if the point is in the TPC with a tolerance.
def CheckBorderFixed(x,y,z,tolerance=0):
    arr = [[tolerance,tolerance],[tolerance,tolerance],[tolerance,tolerance]]
    return CheckBorderTPC(x,y,z,arr)


# Formatting
def sciNot(x):
    x=float(x)
    return "{:.1f}".format(x)

def sciNot2(x):
    x=float(x)
    return "{:.2f}".format(x)


# error unweighter
def effErr(teller,noemer):
    return np.sqrt(teller*(1-teller/noemer))/noemer

