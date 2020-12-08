#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 13:21:08 2019

@author: tatumhennig
"""
import numpy as np
import pandas as pd

## ROT rotates and flips a quadrant appropriately.
#  Parameters:
#    Input, integer N, the length of a side of the square.  
#    N must be a power of 2.
#    Input/output, integer X, Y, the coordinates of a point.
#    Input, integer RX, RY, ???
def rot( n, x, y, rx, ry ):
  if ( ry == 0 ):
#  Reflect.
    if ( rx == 1 ):
      x = n - 1 - x
      y = n - 1 - y
#  Flip.
    t = x
    x = y
    y = t

  return x, y


## XY2D converts a 2D Cartesian coordinate to a 1D Hilbert coordinate.
#  Discussion:
#    It is assumed that a square has been divided into an NxN array of cells,
#    where N is a power of 2.
#    Cell (0,0) is in the lower left corner, and (N-1,N-1) in the upper 
#    right corner.
#  Parameters:
#    integer M, the index of the Hilbert curve.
#    The number of cells is N=2^M.
#    0 < M.
#    Input, integer X, Y, the Cartesian coordinates of a cell.
#    0 <= X, Y < N.
#    Output, integer D, the Hilbert coordinate of the cell.
#    0 <= D < N * N.
def xy2d(x,y):
    # order 4
#    m = 4
#    n = 16
    
    # order 6
#    m = 6
#    n = 64
    
    # order 8
#    m = 8
#    n = 256
    
    # order 10
    m = 10      # index of hilbert curve
    n = 1024    # number of boxes (2^m)
    
    # order 12
#    m = 12
#    n = 4096
    
    
    xcopy = x
    ycopy = y
    
    d = 0
    n = 2 ** m
    
    s = ( n // 2 )
    
    while ( 0 < s ):
        if ( 0 <  ( abs ( xcopy ) & s ) ):
          rx = 1
        else:
          rx = 0
        if ( 0 < ( abs ( ycopy ) & s ) ):
          ry = 1
        else:
          ry = 0
        d = d + s * s * ( ( 3 * rx ) ^ ry )
        xcopy, ycopy = rot(s, xcopy, ycopy, rx, ry )
        s = ( s // 2 )
    
    return d

#*****************************************************************************#

# load in phi psi csv
name = 'ram_points_hil2'
data = pd.read_csv(name + '.csv')
#data.set_index('Unnamed: 0',inplace=True) # index is the time frames!

#name = 'hd_42_o10_phipsi_wt_allpH_c6_phipsi'
#data=pd.read_csv(name+'.csv')
del data['Unnamed: 0']

# run this if you want to convert to cos/sin
datacos = data.apply(lambda x: np.cos(x*np.pi/180))
datasin = data.apply(lambda x: np.sin(x*np.pi/180))
datalinear = pd.concat([datacos,datasin],axis=1)

## 1 set of phi psi column names
datalinear.columns = ['phi cos', 'psi cos', 'phi sin', 'psi sin']
new_order = [0, 2, 1, 3]
datalinear = datalinear[datalinear.columns[new_order]]



# 10 sets of phi psi column names
datalinear.columns = ['phi2 cos', 'psi2 cos', 'phi3 cos', 'psi3 cos', 'phi4 cos', 
                       'psi4 cos', 'phi5 cos', 'psi5 cos', 'phi6 cos', 'psi6 cos', 
                       'phi7 cos', 'psi7 cos', 'phi8 cos', 'psi8 cos', 'phi9 cos', 
                       'psi9 cos', 'phi10 cos','psi10 cos', 'phi11 cos', 'psi11 cos', 
                       'phi2 sin', 'psi2 sin', 'phi3 sin', 'psi3 sin', 'phi4 sin',
                       'psi4 sin', 'phi5 sin', 'psi5 sin', 'phi6 sin', 'psi6 sin', 
                       'phi7 sin', 'psi7 sin', 'phi8 sin', 'psi8 sin', 'phi9 sin', 
                       'psi9 sin', 'phi10 sin', 'psi10 sin', 'phi11 sin', 'psi11 sin']
new_order = [0, 20, 1, 21, 2, 22, 3, 23, 4, 24, 5, 25, 6, 26, 7, 27, 8, 28, 9, 
             29, 10, 30, 11, 31, 12, 32, 13, 33, 14, 34, 15, 35, 16, 36, 17, 
             37, 18, 38, 19, 39]
datalinear = datalinear[datalinear.columns[new_order]]
datalinear.to_csv('cs_ram_points.csv')

# transform and round data to integer values into pixel space
#   - adding 180 because our lowest phi/psi value possible is -180 and we
#       want lowest value to be zero.
#   - dividing by 1023 because we are using order 10 (0 to 1023 is 1024 pixels)

    # phi psi transform
transformed_data = data.apply(lambda x: np.round((x+180)/(360/1023),decimals=0))

    # 40->20 transform
data = datalinear
transformed_data = data.apply(lambda x: np.round((x+1)/(2/1023),decimals=1))

    # 40->20->10 transform
xmin = min(data.min())
xmax = max(data.max())
transformed_data = data.apply(lambda x : (x-xmin)*1023/(xmax-xmin))

# round data & create combined dataframe
rounded_data = transformed_data.apply(np.int64)
combined_data = pd.DataFrame(index=rounded_data.index)

# combine phi psi values into one column
for i in [0,2,4,6,8,10,12,14,16,18]:
for i in [0]:
    combined_data['AA'+str(i)]=rounded_data.iloc[:,i:i+2].values.tolist()

# combine cos/sin values into one column
#for i in [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38]:
for i in [0, 2]:
    combined_data['a'+str(i)]=rounded_data.iloc[:,i:i+2].values.tolist()

# convert 2d into 1d
frame_num = 16
col_num = 1
hilbert_data = np.zeros((frame_num, col_num))
for i in range(frame_num):
    for j in range(col_num):
        hilbert_data[i, j] = xy2d(combined_data.iloc[i,j][0],combined_data.iloc[i,j][1])

# add index and column titles to hilbert data
hilbert_data=pd.DataFrame(hilbert_data,index=combined_data.index,columns=combined_data.columns)

# save
hilbert_data.to_csv('ram_points_hil1.csv')


