 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 11:33:26 2019

@author: tatumhennig
"""
## HILBERT ANALYSIS ##
#
#
#
#
#
##### LIBRARIES ###############################################################

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt 
from sklearn import mixture
import matplotlib as mpl
import itertools
from scipy.stats import gaussian_kde
from scipy.stats import multivariate_normal as mvn
from scipy import linalg
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import os
import glob


##### COMBINING PHI PSI CSVS ##################################################
# The following code will combine all the csv files in the working directory 
#  folder into a single csv. I manually have to look at which order they are
#  added into the new csv file. 

extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
#combine all files in the list
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
#export to csv
combined_csv.to_csv( '.csv', index=False, encoding='utf-8-sig')


##### LOADING IN DATA #########################################################
# Loading the desired data from the working directory folder


name = ''
data = pd.read_csv(name + '.csv')
data = data.round(2)
del data['Unnamed: 0']  # SOME of csv files have an extra column with the frame #s


##### TRANSFORMING TO Cos/Sin DATA ############################################
# this takes the phi psi data from the inputted csv file and converts the data
# into the cos/sin of the phi psi space. 

datacos = data.apply(lambda x: np.cos(x*np.pi/180))
datasin = data.apply(lambda x: np.sin(x*np.pi/180))
datalinear = pd.concat([datacos,datasin],axgtis=1)

# rearranging the data so its in correct order

# for Hilbert 20
#datalinear.columns = ['phi2 cos', 'psi2 cos', 'phi3 cos', 'psi3 cos', 'phi4 cos', 
#                       'psi4 cos', 'phi5 cos', 'psi5 cos', 'phi6 cos', 'psi6 cos', 
#                       'phi7 cos', 'psi7 cos', 'phi8 cos', 'psi8 cos', 'phi9 cos', 
#                       'psi9 cos', 'phi10 cos','psi10 cos', 'phi11 cos', 'psi11 cos', 
#                       'phi2 sin', 'psi2 sin', 'phi3 sin', 'psi3 sin', 'phi4 sin',
#                       'psi4 sin', 'phi5 sin', 'psi5 sin', 'phi6 sin', 'psi6 sin', 
#                       'phi7 sin', 'psi7 sin', 'phi8 sin', 'psi8 sin', 'phi9 sin', 
#                       'psi9 sin', 'phi10 sin', 'psi10 sin', 'phi11 sin', 'psi11 sin']
#new_order = [0, 20, 1, 21, 2, 22, 3, 23, 4, 24, 5, 25, 6, 26, 7, 27, 8, 28, 9, 
#             29, 10, 30, 11, 31, 12, 32, 13, 33, 14, 34, 15, 35, 16, 36, 17, 
#             37, 18, 38, 19, 39]

# for Hilbert 10
datalinear.columns = ['phi2cos', 'psi2cos', 'phi2sin', 'psi2sin']
new_order = [0, 2, 1, 3]
datalinear = datalinear[datalinear.columns[new_order]]

# saving this data 
datalinear.to_csv( "4A_cossin.csv", index=False, encoding='utf-8-sig')
    # note: have to set data = datalinear if cos/sin data is desired on rest of analysis

data = datalinear

##### HILBERT TRANSFORM #######################################################
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
    # order 10
    m = 10      # index of hilbert curve
    n = 1024    # number of boxes (2^m)
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
#for i in [0,2]:
    combined_data['AA'+str(i)]=rounded_data.iloc[:,i:i+2].values.tolist()

# combine cos/sin values into one column
for i in [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38]:
#for i in [0, 2, 4, 6, 8, 10]:
    combined_data['a'+str(i)]=rounded_data.iloc[:,i:i+2].values.tolist()

# convert 2d into 1d
frame_num =  # size of csv
col_num =  # number of columns
hilbert_data = np.zeros((frame_num, col_num))
for i in range(frame_num):
    for j in range(col_num):
        hilbert_data[i, j] = xy2d(combined_data.iloc[i,j][0],combined_data.iloc[i,j][1])

# add index and column titles to hilbert data
hilbert_data=pd.DataFrame(hilbert_data,index=combined_data.index,columns=combined_data.columns)

# save
hilbert_data.to_csv('')

##### PERFORM PCA #############################################################

# center and scale data
    # -> changes avg value to zero
    # -> changes std dev to one
scaled_data = preprocessing.scale(data)

pca = PCA()     # creating the PCA object
pca.fit(scaled_data)    # math!
pca_data = pca.transform(scaled_data)   # PCA coordinates

# setting up labels and calculating explained variance
per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)
labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]

# PCA!
pca_df = pd.DataFrame(pca_data, columns=labels)


##### PLOTS ###################################################################

# *** Change names of plots as needed :) 
## Scree Plot
fig = plt.figure()
plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.xticks(rotation=90)
plt.title('2xHilbert Scree Plot - M9')
plt.show()
fig.savefig('scree_' + name + '_hil.png')

## Plot using PC1 and PC2
fig2 = plt.figure()
plt.scatter(pca_df.PC1, pca_df.PC2, s=0.01)
plt.title('Cos/Sin PCA Plot - WT + M2 all Temp, C6')
plt.xlabel('PC1 - {0}%'.format(per_var[0]))
plt.ylabel('PC2 - {0}%'.format(per_var[1])) 
plt.show()
fig2.savefig('PC1_PC2_' + name + '.png')

fig2 = plt.figure()
plt.scatter(data.a0, data.a2, s=0.01)
plt.title('Phi vs Psi - Alanine Tetra-Peptide')
plt.xlabel('2-Ala phi')
plt.ylabel('2-Ala psi') 
plt.show()
fig2.savefig('phi_psi' + name + '.png')

## Plot using PC1, PC2, and PC3
ax = plt.axes(projection = '3d')
ax.scatter3D(pca_df.PC1, pca_df.PC2, pca_df.PC3, s=0.01)
ax.set_title('1xHilbert 3D PCA Plot - Alanine Tetra-Peptide')
ax.set_xlabel('PC1 - {0}%'.format(per_var[0]))
ax.set_ylabel('PC2 - {0}%'.format(per_var[1]))
ax.set_zlabel('PC3 - {0}%'.format(per_var[2]))
plt.savefig('3D_PCA_' + name + '.png')

## Coloring simulations separately
# I manually select simulations in combined csv

fig5 = plt.figure() 
plt.scatter(pca_df.PC1[99966:119952], pca_df.PC2[99966:119952], s=0.01, color='royalblue')
plt.scatter(pca_df.PC1[0:19985], pca_df.PC2[0:19985], s=0.01, color='tomato')
plt.scatter(pca_df.PC1[59982:79967], pca_df.PC2[59982:79967], s=0.01, color='lightseagreen')

#plt.scatter(pca_df.PC1[79967:99965], pca_df.PC2[79967:99965], s=0.01, color='royalblue')
#plt.scatter(pca_df.PC1[39984:59981], pca_df.PC2[39984:59981], s=0.01, color='tomato')
#plt.scatter(pca_df.PC1[19986:39983], pca_df.PC2[19986:39983], s=0.01, color='lightseagreen')

#plt.scatter(pca_df.PC1[99966:119951], pca_df.PC2[99966:119951], s=0.01, color='darkblue')

#plt.scatter(pca_df.PC1[59970:79967], pca_df.PC2[59970:79967], s=0.01, color='orange')
#plt.scatter(pca_df.PC1[39972:59981], pca_df.PC2[39972:59981], s=0.01, color='green')
#plt.scatter(pca_df.PC1[99954:119952], pca_df.PC2[99954:119952], s=0.01, color='purple')
#
#plt.scatter(pca_df.PC1[19986:39971], pca_df.PC2[19986:39971], s=0.01, color='peru')
#plt.scatter(pca_df.PC1[119952:139937], pca_df.PC2[119952:139937], s=0.01, color='r')
#plt.scatter(pca_df.PC1[99966:119951], pca_df.PC2[99966:119951], s=0.01, color='turquoise')
#plt.scatter(pca_df.PC1[0:19985], pca_df.PC2[0:19985], s=0.01, color='slateblue')

plt.title('2xHilbert PCA Plot - M9 all pH, C6')
plt.xlabel('PC1 - {0}%'.format(per_var[0]))
plt.ylabel('PC2 - {0}%'.format(per_var[1])) 
fig5.savefig('scatter_m9_c6_hil.png')



## Density plot in 2D
    # diff simulations
    # I am MANUALLY picking x and y here. I know where my simulations 
    # are located in my combined csv file and I am picking them out here.

x = 19986   # I am MANUALLY picking x and y here. I know where my simulations 
y = 39972   # are located in my combined csv file and I am picking them out here.
levels = np.linspace(0, 1, 50) # levels of density
ax = sns.kdeplot(pca_df.PC1[x:y],pca_df.PC2[x:y],n_levels=levels,
                 shade=True,cmap='terrain_r', shade_lowest=False, cbar=True)
ax.set_title('Phi/Psi Hilbert Density Plot - GrBP5 WT pH 9, C6')
ax.set_xlabel('PC1 - {0}%'.format(per_var[0]))
ax.set_ylabel('PC2 - {0}%'.format(per_var[1]))
ax.set_xlim(-6, 6)
ax.set_ylim(-5, 7)
plt.savefig('s-DensityPCA_hil10_wt_pH9_c6.png')

    # all together now
#levels = np.linspace(0, 0.25, 30)
ax = sns.kdeplot(pca_df.PC1,pca_df.PC2,n_levels=100,
                 shade=True, cmap="terrain_r", shade_lowest=False, cbar=True)
ax.set_title('1xHilbert Density Plot - Alanine Tetra-Peptide')
ax.set_xlabel('PC1 - {0}%'.format(per_var[0]))
ax.set_ylabel('PC2 - {0}%'.format(per_var[1]))
#ax.set_xlim(-4, 4)
#ax.set_ylim(-4, 4)
#plt.savefig('DensityPCA_' + name + '.png')
plt.scatter(-2, 1.5)



## 3D PC frames for EzGif
    # again I am manually selecting my x and y (same as above)
xy = np.vstack([pca_df.PC1[x:y],pca_df.PC2[x:y]])
z = gaussian_kde(xy)(xy)
a = 0
cmap = mpl.cm.gist_ncar_r
norm = mpl.colors.Normalize(vmin=0, vmax=0.72)
for i in range(361) :
    ax = plt.axes(projection = '3d')
    ax.scatter3D(pca_df.PC1[x:y],pca_df.PC2[x:y], z, c=z, cmap=cmap, norm=norm,s=0.01)
    ax.set_title('Title')
    ax.set_xlabel('PC1 - {0}%'.format(per_var[0]))
    ax.set_ylabel('PC2 - {0}%'.format(per_var[1]))
    ax.set_zlabel('Density')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(0, 0.8)
    ax.view_init(azim=x)
    plt.savefig(str(i) + '_3D_gif_sim1.png')
    plt.close()
    a = a + 1
    
## Time Movie for EzGif [pedro has better version :)]
x = 0
for i in range(400) :
    plt.scatter(pca_df.PC1, pca_df.PC2,s=0.01)
    plt.title('Title')
    plt.xlabel('PC1 - {0}%'.format(per_var[0]))
    plt.ylabel('PC2 - {0}%'.format(per_var[1]))
    plt.scatter(pca_df.PC1[x], pca_df.PC2[x], color='r')
    plt.savefig('frame_' + str(i) + name + '.png')
    plt.close()
    x = x + 50
    
# WT pH 3 C6
x = 0
y = 19985

## Probability maps
    # again I am manually selecting my x and y!
xmin = pca_df.PC1[x:y].min()
xmax = pca_df.PC1[x:y].max()
ymin = pca_df.PC2[x:y].min()
ymax = pca_df.PC2[x:y].max()
X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([X.ravel(), Y.ravel()])
values = np.vstack([pca_df.PC1[x:y], pca_df.PC2[x:y]])
kernel = gaussian_kde(values)
Z = np.reshape(kernel(positions).T, X.shape)
Z1 = Z*((xmax-xmin)/100)*((ymax-ymin)/100)

kb = 1.38064852 * 10**-23 # m2 kg s-2 K-1
T = 300 # K

G = kb * T * np.log(Z1/(1-Z1))
S = -kb * Z1 * np.log(Z1)
H = G + T*S

ax = sns.heatmap(Z1, cmap="BuPu")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
plt.title('Probability Map - GrBP5 WT pH 3, C6')
plt.xlabel('PC1 - {0}%'.format(per_var[0]))
plt.ylabel('PC2 - {0}%'.format(per_var[1]))
plt.savefig('prob_heatmap_wt_pH3' + name + '.png')

##### LOADING SCORES ##########################################################
# loading scores & sort by magnitude

# get names for PC1
loading_scores1 = pd.Series(pca.components_[0])
sorted_loading_scores1 = loading_scores1.abs().sort_values(ascending=False)
PC1_ls = sorted_loading_scores1[0:40].index.values
print(loading_scores1[PC1_ls])

# get names for PC2
loading_scores2 = pd.Series(pca.components_[1])
sorted_loading_scores2 = loading_scores2.abs().sort_values(ascending=False)
PC2_ls = sorted_loading_scores2[0:40].index.values
print(loading_scores2[PC2_ls])

# get names for PC3
loading_scores3 = pd.Series(pca.components_[2])
sorted_loading_scores3 = loading_scores3.abs().sort_values(ascending=False)
PC3_ls = sorted_loading_scores3[0:40].index.values
print(loading_scores3[PC3_ls])


##### MEANS ###################################################################
# finding a representative time frame for a certain cluster

# I am manually finding these...
time_frames1 = pca_df.index[(pca_df.PC1<-2.0+0.005) & 
                            (pca_df.PC1>-2.0-0.005) & 
                            (pca_df.PC2<1.5+0.005) & 
                            (pca_df.PC2>1.5-0.005)]


##### RMSD DIFFERENCES BETWEEN REPRESENTATIVE FRAMES ##########################
# finding the rmsd difference between frames

## desired frames (from my manual selection of the cluster representaions)
frames = []
frame_data = pd.DataFrame(data, index=frames, columns=datalinear.columns)

## RMSD
df = pd.DataFrame(index=frames, columns=frames)
for i in range(len(frames)) :
    for j in range(len(frames)) :
        df.iloc[i,j] = ((frame_data.T.iloc[:,i] - frame_data.T.iloc[:,j]) ** 2).mean() ** .5
## save
df.to_csv('rmsdstruc_' + name + '.csv')


## Heat Map of saved csv structures
    ## loading in data
r = pd.read_csv('name.csv')
r.set_index('Unnamed: 0',inplace=True) # set frame num to index
    ## heat map!
r.columns=[1,2,3,4,5,6,7] # changing label to numbered structures
r.index=[1,2,3,4,5,6,7]

ax = sns.heatmap(r, cmap='Blues')
ax.set_title('Title')
ax.set_xlabel('Structures')
ax.set_ylabel('Structures')
plt.savefig('rmsd_heatmap' + name + '.png')

##### BI PLOTS ################################################################


plt.scatter(pca_df.PC1[119952:139937], pca_df.PC2[119952:139937], s=0.01, color='orange')
plt.scatter(pca_df.PC1[59982:79967], pca_df.PC2[59982:79967], s=0.01, color='green')
plt.scatter(pca_df.PC1[39996:59981], pca_df.PC2[39996:59981], s=0.01, color='purple')
plt.scatter(pca_df.PC1[99966:119951], pca_df.PC2[99966:119951], s=0.01, color='darkblue')

#plt.scatter(pca_df.PC1[19986:39971], pca_df.PC2[19986:39971], s=0.01, color='peru')
#plt.scatter(pca_df.PC1[119952:139937], pca_df.PC2[119952:139937], s=0.01, color='r')
#plt.scatter(pca_df.PC1[99966:119951], pca_df.PC2[99966:119951], s=0.01, color='turquoise')
#plt.scatter(pca_df.PC1[0:19985], pca_df.PC2[0:19985], s=0.01, color='slateblue')
plt.title('Cos/Sin BiPlot - WT all Temp, C6')
plt.xlabel('PC1 - {0}%'.format(per_var[0]))
plt.ylabel('PC2 - {0}%'.format(per_var[1])) 

#names = ['cos(2-Met phi)', 'sin(2-Met phi)', 'cos(2-Met psi)', 'sin(2-Met psi)', 'cos(3-Val phi)', 'sin(3-Val phi)', 'cos(3-Val psi)', 'sin(3-Val psi)',
#         'cos(4-Thr phi)', 'sin(4-Thr phi)', 'cos(4-Thr psi)', 'sin(4-Thr psi)', 'cos(5-Glu phi)', 'sin(5-Glu phi)', 'cos(5-Glu psi)', 'sin(5-Glu psi)',
#         'cos(6-Ser phi)', 'sin(6-Ser phi)', 'cos(6-Ser psi)', 'sin(6-Ser psi)', 'cos(7-Ser phi)', 'sin(7-Ser phi)', 'cos(7-Ser psi)', 'sin(7-Ser psi)',
#         'cos(8-Asp phi)', 'sin(8-Asp phi)', 'cos(8-Asp psi)', 'sin(8-Asp psi)', 'cos(9-Tyr phi)', 'sin(9-Tyr phi)', 'cos(9-Tyr psi)', 'sin(9-Tyr psi)', 
#         'cos(10-Ser phi)', 'sin(10-Ser phi)', 'cos(10-Ser psi)', 'sin(10-Ser psi)', 'cos(11-Ser phi)', 'sin(11-Ser phi)', 'cos(11-Ser psi)', 'sin(11-Ser psi)']

#names = ['', '', '', '', '', '', '', '',
#         '', '', '', '', 'cos(5-Gln phi)', 'sin(5-Gln phi)', 'cos(5-Gln psi)', 'sin(5-Gln psi)',
#         '', '', '', '', '', '', '', '',
#         'cos(8-Asn phi)', 'sin(8-Asn phi)', 'cos(8-Asn psi)', 'sin(8-Asn psi)', '', '', '', '', 
#         '', '', '', '', '', '', '', '']

#names = ['', '', '', '', '', '', '', '',
#         '', '', '', '', '', '', '', '',
#         '', '', '', '', '', '', '', '',
#         '', '', '', '', 'cos(9-Tyr phi)', 'sin(9-Tyr phi)', 'cos(9-Tyr psi)', 'sin(9-Tyr psi)', 
#         '', '', '', '', '', '', '', '']



names = ['2-Met', '3-Val', '4-Thr', '5-Glu', '6-Ser', '7-Ser', '8-Asp', '9-Tyr', '10-Ser', '11-Ser']

#names =['cos(2-Ala phi)', 'sin(2-Ala phi)', 'cos(2-Ala psi)', 'sin(2-Ala psi)']
#names =['2-Ala phi', '2-Ala psi']
#names = ['cos(2-Ala phi)', 'sin(2-Ala phi)', 'cos(2-Ala psi)', 'sin(2-Ala psi)','cos(3-Ala phi)', 'sin(3-Ala phi)', 'cos(3-Ala psi)', 'sin(3-Ala psi)']
#names = ['2-Ala phi', '2-Ala psi', '3-Ala phi', '3-Ala psi']


comps = np.transpose(pca.components_[0:2, :])
for i in range(comps.shape[0]):
    plt.arrow(0.5, 0, comps[i, 0]*5, comps[i, 1]*5, color = 'r', alpha = 0.5)
#    plt.text(comps[i,0]*5, comps[i,1]*5, names[i], color = 'k', ha = 'center', va = 'center')

plt.savefig('2D_PCA_Biplot_mut_nolabels' + name + '.png')

