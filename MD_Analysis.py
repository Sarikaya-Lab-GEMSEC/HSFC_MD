# =============================================================================
# Created on Thu Jul 11 10:56:42 2019
# Last Edited on Thu Oct 10 12:45 2019
# 
# @author: Pedro Fischer Marques
# GEMSEC Labs - Computational Group
# =============================================================================

# =============================================================================
# IMPORTS
# =============================================================================

#General Libraries
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import multiprocessing.dummy as mp

#PDB Parsing Libraries
#import prody as prd
from Bio import PDB
import scipy.constants as const

#Dimensional Reduction Libraries
from sklearn.manifold import MDS #Multi Dimensional Scaling
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.manifold import SpectralEmbedding

#PCA Libraries
import imageio
from sklearn import preprocessing
from scipy.stats import gaussian_kde
from mpl_toolkits.mplot3d import Axes3D
Axes3D #To silence notification, above import necessary

#GMM Libraries
import itertools
import matplotlib as mpl
from scipy import linalg
from sklearn import mixture

#Ramachandran Libraries
import scipy.stats as stat

#Transformations Libraries
from scipy import signal

# =============================================================================
# CLASSES
# =============================================================================

class Multi_PDB():
    def __init__(self, file_list):
        self.files = file_list
        
    #Create a list of all backbone angles across all conditions
    #Parameters:
    #   -files: list of str; a list of all pdbs with angles to be combined
    #Returns:
    #   -combined_angles: pandas DataFrame; a dataframe containing all phi/psi angles
    #                     for each condition, separated by rows of np.nan
    def get_combo_angles(self):
        #Import necessary packages
        from MD_Analysis import Angle_Calc
        
        with mp.Pool() as pool:
            backbone_list = pool.map(lambda x: Angle_Calc.get_backbone(self.files[x]),
                                     range(len(self.files)))
            angle_list = pool.map(lambda x: Angle_Calc.get_angles(x), backbone_list)
            combined_angles = pd.concat(angle_list)
        return combined_angles

class Angle_Calc():
    def __init__(self, in_file):
        self.file = in_file
        self.angles = pd.DataFrame()
        self.chi_angles = pd.DataFrame()
    
    #Get a backbone from a pdb
    #Parameters:
    #   -file: str; name of the pdb to parse a backbone from
    #Returns:
    #   -backbone: Bio.PDB structure object; the backbone structure given by the pdb
    def __get_backbone(self):
        #Parse the pdb file for its structure and then backbone
        structure = prd.parsePDB(self.file)
        back_only = prd.writePDB(self.file[:-4] + "_backbone.pdb", structure.select('name N CA C'))
        
        #Parse through the backbone
        parser = PDB.PDBParser()
        backbone = parser.get_structure(os.path.basename(self.file)[:-4], back_only)
    
        return backbone
    
    #Get phi/psi angles from a backbone pdb
    #Parameters:
    #   -pdb: str; the pdb file to get angles from
    #Returns:
    #   -angles_by_frame: pandas DataFrame; contains all phi/psi angles (columns) 
    #                     for each frame (rows) from the simulation in the pdb file
    def get_phi_psi(self):
        #Get phi/psi angles from biopython
        backbone = self.__get_backbone()
        model_list = PDB.Selection.unfold_entities(backbone, 'M')
        with mp.Pool() as pool:
            chain_list = pool.map(lambda x: x['A'], model_list)
            poly_list = pool.map(lambda x: PDB.Polypeptide.Polypeptide(x), chain_list)
            angle_list = pool.map(lambda x: x.get_phi_psi_list(), poly_list)
            rowstuff = pool.map(lambda x: np.reshape(x,[1,len(x)*2])[0][2:-2] * (180/np.pi), angle_list)
            rowlist = list(rowstuff)
        
        #Generate a dataframe and store angles
        clmns = []
        end_marks = []
        for i in range(10):
            clmns.append('phi' f'{i+1}')
            clmns.append('psi' f'{i+1}')
            end_marks.append(np.nan)
            end_marks.append(np.nan)
    
        angles_by_frame = pd.DataFrame(columns = np.linspace(1,20,num = 20))
        angles_by_frame = pd.DataFrame(rowlist,index=np.linspace(1,len(rowlist),num=len(rowlist)),columns=clmns)
        end_marks = pd.DataFrame(end_marks, index = clmns)
        angles_by_frame = angles_by_frame.append(end_marks.T)

        self.angles = angles_by_frame
        return angles_by_frame
    
    #Define all possible chi angles for each residue
    #Returns:
    #   -chi_atoms: dict; a dictionary of all possible chi angle combinations
    #               for each amino acid
    #Credit:
        # Copyright (c) 2014 Lenna X. Peterson, all rights reserved
        # lenna@purdue.edu
    def __gen_chi_list(self):
        chi_atoms = dict(
            chi1=dict(
                ARG=['N', 'CA', 'CB', 'CG'],
                ASN=['N', 'CA', 'CB', 'CG'],
                ASP=['N', 'CA', 'CB', 'CG'],
                CYS=['N', 'CA', 'CB', 'SG'],
                GLN=['N', 'CA', 'CB', 'CG'],
                GLU=['N', 'CA', 'CB', 'CG'],
                HIS=['N', 'CA', 'CB', 'CG'],
                ILE=['N', 'CA', 'CB', 'CG1'],
                LEU=['N', 'CA', 'CB', 'CG'],
                LYS=['N', 'CA', 'CB', 'CG'],
                MET=['N', 'CA', 'CB', 'CG'],
                PHE=['N', 'CA', 'CB', 'CG'],
                PRO=['N', 'CA', 'CB', 'CG'],
                SER=['N', 'CA', 'CB', 'OG'],
                THR=['N', 'CA', 'CB', 'OG1'],
                TRP=['N', 'CA', 'CB', 'CG'],
                TYR=['N', 'CA', 'CB', 'CG'],
                VAL=['N', 'CA', 'CB', 'CG1'],
            ),
            chi2=dict(
                ARG=['CA', 'CB', 'CG', 'CD'],
                ASN=['CA', 'CB', 'CG', 'OD1'],
                ASP=['CA', 'CB', 'CG', 'OD1'],
                GLN=['CA', 'CB', 'CG', 'CD'],
                GLU=['CA', 'CB', 'CG', 'CD'],
                HIS=['CA', 'CB', 'CG', 'ND1'],
                ILE=['CA', 'CB', 'CG1', 'CD1'],
                LEU=['CA', 'CB', 'CG', 'CD1'],
                LYS=['CA', 'CB', 'CG', 'CD'],
                MET=['CA', 'CB', 'CG', 'SD'],
                PHE=['CA', 'CB', 'CG', 'CD1'],
                PRO=['CA', 'CB', 'CG', 'CD'],
                TRP=['CA', 'CB', 'CG', 'CD1'],
                TYR=['CA', 'CB', 'CG', 'CD1'],
            ),
            chi3=dict(
                ARG=['CB', 'CG', 'CD', 'NE'],
                GLN=['CB', 'CG', 'CD', 'OE1'],
                GLU=['CB', 'CG', 'CD', 'OE1'],
                LYS=['CB', 'CG', 'CD', 'CE'],
                MET=['CB', 'CG', 'SD', 'CE'],
            ),
            chi4=dict(
                ARG=['CG', 'CD', 'NE', 'CZ'],
                LYS=['CG', 'CD', 'CE', 'NZ'],
            ),
            chi5=dict(
                ARG=['CD', 'NE', 'CZ', 'NH1'],
            ),
        )
        
        return chi_atoms
    
    #Calculate the chi angle between a set of four atoms from a residue
    #Parameters:
    #   -residue: Bio.PDB residue object; the residue to calculate chi angles for
    #   -group: str; the chi angle to be calculated (i.e. chi1, chi2, etc.)
    #Returns:
    #   -float; the value of the chi angle in degrees (between -180 and 180)
    def __calc_chi(self, residue, group):
        #Define all possible chi angles for each residue
        chi_atoms = self.__gen_chi_list()

        #Convert Needed data to appropriate formats
        res_atoms = PDB.Selection.unfold_entities(residue, 'A')
        res_name = residue.get_resname()
        atom_names = []
        for atom in res_atoms:
            atom_names.append(atom.get_name())
        
        #Gather all four atoms to calculate the angle for
        atom1 = res_atoms[atom_names.index(chi_atoms[group][res_name][0])].get_vector()
        atom2 = res_atoms[atom_names.index(chi_atoms[group][res_name][1])].get_vector()
        atom3 = res_atoms[atom_names.index(chi_atoms[group][res_name][2])].get_vector()
        atom4 = res_atoms[atom_names.index(chi_atoms[group][res_name][3])].get_vector()
        #Return the dihedral angle between the atoms
        return PDB.calc_dihedral(atom1, atom2, atom3, atom4)*(180/const.pi)

    #Get chi angles for each residue from a pdb
    #Parameters:
    #   -file: str; name of the pdb to gather chi angles from
    #Returns:
    #   -chi_df: pandas DataFrame; a dataframe of all chi angles for each residue
    #            at each time frame during the simulation
    def get_chi(self):
        #Define all possible chi angles for each residue
        chi_atoms = self.__gen_chi_list()
            
        #Import the pdb structure file
        parser = PDB.PDBParser()
        pep = parser.get_structure(self.file[:-4], self.file)
        
        #Get a list of each residue
        model_list = PDB.Selection.unfold_entities(pep, 'M')
        with mp.Pool() as pool:
            res_list = pool.map(lambda x: PDB.Selection.unfold_entities(x, 'R'), model_list)
        
        chi_dict = {"Residue": [], "Chi 1": [], "Chi 2": [], "Chi 3": [],
                    "Chi 4": [], "Chi 5": []}
        
        #Break down the list into individual residues
        for frame in res_list:
            for res in frame:
                res_name = res.get_resname()
                chi_dict["Residue"].append(res_name + ' - Frame ' + f'{res_list.index(frame) + 1}')
                
                if res_name in chi_atoms["chi5"]:
                    chi_dict["Chi 1"].append(self.__calc_chi(res, "chi1"))
                    chi_dict["Chi 2"].append(self.__calc_chi(res, "chi2"))
                    chi_dict["Chi 3"].append(self.__calc_chi(res, "chi3"))
                    chi_dict["Chi 4"].append(self.__calc_chi(res, "chi4"))
                    chi_dict["Chi 5"].append(self.__calc_chi(res, "chi5"))
                elif res_name in chi_atoms["chi4"]:
                    chi_dict["Chi 1"].append(self.__calc_chi(res, "chi1"))
                    chi_dict["Chi 2"].append(self.__calc_chi(res, "chi2"))
                    chi_dict["Chi 3"].append(self.__calc_chi(res, "chi3"))
                    chi_dict["Chi 4"].append(self.__calc_chi(res, "chi4"))
                    chi_dict["Chi 5"].append(0.0)
                elif res_name in chi_atoms["chi3"]:
                    chi_dict["Chi 1"].append(self.__calc_chi(res, "chi1"))
                    chi_dict["Chi 2"].append(self.__calc_chi(res, "chi2"))
                    chi_dict["Chi 3"].append(self.__calc_chi(res, "chi3"))
                    chi_dict["Chi 4"].append(0.0)
                    chi_dict["Chi 5"].append(0.0)
                elif res_name in chi_atoms["chi2"]:
                    chi_dict["Chi 1"].append(self.__calc_chi(res, "chi1"))
                    chi_dict["Chi 2"].append(self.__calc_chi(res, "chi2"))
                    chi_dict["Chi 3"].append(0.0)
                    chi_dict["Chi 4"].append(0.0)
                    chi_dict["Chi 5"].append(0.0)
                elif res_name in chi_atoms["chi1"]:
                    chi_dict["Chi 1"].append(self.__calc_chi(res, "chi1"))
                    chi_dict["Chi 2"].append(0.0)
                    chi_dict["Chi 3"].append(0.0)
                    chi_dict["Chi 4"].append(0.0)
                    chi_dict["Chi 5"].append(0.0)
                else:
                    chi_dict["Chi 1"].append(0.0)
                    chi_dict["Chi 2"].append(0.0)
                    chi_dict["Chi 3"].append(0.0)
                    chi_dict["Chi 4"].append(0.0)
                    chi_dict["Chi 5"].append(0.0)
        
        chi_df = pd.DataFrame.from_dict(chi_dict)
        self.chi_angles = chi_df
        return chi_df
    
    #Calculate the sin and cos values of each angle in a given dataframe
    #Parameters:
    #   -angle_df: pandas DataFrame; a dataframe of angles for each time frame
    #Returns:
    #   -sc_df: pandas DataFrame; a dataframe of the sin/cos values for each residue
    #           at each time frame
    def get_sin_cos(self, angle_df):
        #Create the new dataframe
        sc_df = pd.DataFrame()
        
        #Apply sin and cos transformations to each column in angle_df
        for col in angle_df.columns:
            if col == 'Residue':
                sc_df.insert(0, 'Residue', angle_df['Residue'])
            else:
                sc_df['Sin - ' + col] = angle_df[col].map(lambda x: np.sin(x))
                sc_df['Cos - ' + col] = angle_df[col].map(lambda x: np.cos(x))
                
        return sc_df
                
#Includes all currently used methods of dimensionality reduction
class Dim_Reduction():
    def __init__(self, in_file, angle_df, save_dir):
        self.file = in_file
        self.fname = os.path.basename(in_file)[:-4]
        self.angles = angle_df
        self.scaled = preprocessing.scale(angle_df.dropna(axis = 0))
        self.sdir =  save_dir

    #Function to complete Principal Component Analysis on a given dataset
    #Parameters:
    #   -angles: pandas DataFrame; dataframe of all angles to be graphed with PCA
    #   -file: str; the selected pdb file
    #   -dir_name: str; the directory to save the graph into
    #   -graph_type: str; "2d" or "3d", defines the type of graph to produce
    #   -biplot: bool; defines whether to create a biplot or not
    #Returns:
    #   -pca_df: pandas DataFrame; dataframe of all PCs and their values for each
    #            time frame
    #   -pca: PCA object of sklearn; 
    def pca(self, graph_type, biplot = True):      
        #Get all the PCA components from the data
        pca = PCA()
        pca.fit(self.scaled)
        pca_data = pca.transform(self.scaled)
        per_var = np.round(pca.explained_variance_ratio_ * 100, decimals = 1)
        labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]
        
        #Create a Scree Plot of the components
        plt.close()
        plt.bar(x = range(1, len(per_var) + 1), height = per_var, tick_label = labels)
        plt.ylabel('Percentage of Explained Variance')
        plt.xlabel('Principal Component')
        plt.title('Scree Plot')
        plt.savefig(self.sdir + 'Scree Plot - ' + self.fname + '.png')
        plt.show()
        plt.close()
        
        pca_df = pd.DataFrame(pca_data, columns = labels)
    
        #Generate the PCA Graph based on PC1 and PC2
        if graph_type == '2d':
            plt.scatter(pca_df.PC1, pca_df.PC2, s = 0.05)
            plt.title('Torsion Angle PCA Graph')
            plt.xlabel('PC1 - {0}%'.format(per_var[0]))
            plt.ylabel('PC2 - {0}%'.format(per_var[1]))
            if biplot:
                comps = np.transpose(pca.components_[0:2, :])
                for i in range(comps.shape[0]):
                    plt.arrow(0, 0, comps[i, 0], comps[i, 1], color = 'r', alpha = 0.5)
                plt.savefig(self.sdir + '2D PCA Biplot - ' + self.fname + '.png')
            else:
                plt.savefig(self.sdir + '2D PCA - ' + self.fname + '.png')
            plt.close()
            
        #Generate the PCA Graph Based on PC1, PC2, and PC3
        elif graph_type == '3d':
            ax = plt.axes(projection = '3d')
            ax.scatter3D(pca_df.PC1, pca_df.PC2, pca_df.PC3, s = 0.01,
                         depthshade = True)
            ax.set_xlabel('PC1 - {0}%'.format(per_var[0]))
            ax.set_ylabel('PC2 - {0}%'.format(per_var[1]))
            ax.set_zlabel('PC3 - {0}%'.format(per_var[2]))
            if biplot:
                comps = np.transpose(pca.components_[0:3, :])
                for i in range(comps.shape[0]):
                    ax.plot([0, comps[i, 0]], [0, comps[i, 1]], [0, comps[i, 2]], color = 'r', alpha = 0.5)
                    plt.savefig(self.sdir + '3D PCA Biplot - ' + self.fname + '.png', pad_inches = 0)
            else:
                plt.savefig(self.sdir + '3D PCA - ' + self.fname + '.png', pad_inches = 0)
            plt.close()
        
        else:
            raise Exception('Graph Type must be either "2d" or "3d".')
        
        return pca_df, pca
    
    def ICA(self, components):
        ICA_data = FastICA(n_components = components).fit_transform(self.scaled)
        
        plt.scatter(ICA_data[:, 0], ICA_data[:, 1], s = 0.05)
        plt.title("Phi-Psi Independent Component Analysis")
        plt.savefig(self.sdir + "Independent Component Analysis - " + self.fname + ".png")
        return ICA_data
    
    def Multi_Dim_Scaling(self, components):
        mds_data = MDS(n_components = components).fit_transform(self.scaled)
        
        plt.scatter(mds_data[:, 0], mds_data[:, 1], s = 0.05)
        plt.title("Phi-Psi Multi Dimensional Scaling")
        plt.savefig(self.sdir + "Multi Dimensional Scaling - " + self.fname + ".png")
        return mds_data
    
    def Spectral_Embed(self, components):
        se_data = SpectralEmbedding(n_components = components).fit_transform(self.scaled)
        
        plt.scatter(se_data[:, 0], se_data[:, 1], s = 0.05)
        plt.title("Phi-Psi Spectral Embedding")
        plt.savefig(self.sdir + "Spectral Embedding - " + self.fname + ".png")
        return se_data
    
    def t_SNE(self, components):
        tsne_data = TSNE(n_components = components).fit_transform(self.scaled)
        
        plt.scatter(tsne_data[:, 0], tsne_data[:, 1], s = 0.05)
        plt.title("Phi-Psi t-Stochastic Neighbor Embedding")
        plt.savefig(self.sdir + "t-Stochastic Neighbor Embedding - " + self.fname + ".png")
        return tsne_data
    
    def IsoMap(self, components):
        IM_data = Isomap(n_components = components).fit_transform(self.scaled)
        
        plt.scatter(IM_data[:, 0], IM_data[:, 1], s = 0.05)
        plt.title("Phi-Psi Isomap")
        plt.savefig(self.sdir + "Isomap - " + self.fname + ".png")
        return IM_data
        
class PCA_Components():
    def __init__(self, pca, save_dir):
        self.pca = pca
        self.pca_df = pca[0]
        self.sdir = save_dir
    
    def __name_append(s, t, p):
        name = ''
        if s != '':
            name += ' - ' + s
            if t != '':
                name += '_' + t
            if p != '':
                name += '_' + p
        elif t != '':
            name += ' - ' + t
            if p != '':
                name += '_' + p
        elif p != '':
            name += ' - ' + p
        
        return name
    
    #Function to gather loading scores after PCA is completed
    #Parameters:
    #   -pca: sklearn PCA object; 
    #   -PC: str; 
    #   -n: int
    #   -bottom: bool; 
    #Returns:
    #   -top_LS: pandas DataFrame; 
    #   -bot_LS: pandas DataFrame; 
    def load_score(self, PC, n = 3, bottom = False):        
        #Gather and return loading scores for all PCs
        #Optional: provide "n" for how many top/bottom scores to display
        if PC.lower() == "all":
            with mp.Pool() as pool:
                #Collect all scores
                all_scores = pool.map(lambda x: pd.Series(self.pca.components_[x]),
                                 range(len(self.pca.components_)))
                #Sort the scores in descending order
                all_sorted_scores = pool.map(lambda x: x.abs().sort_values(ascending = False), all_scores)
                #Gather the top "n" components and their scores
                all_top_n = pool.map(lambda x: x[0:n].index.values, all_sorted_scores)
                all_top_n_scores = pool.map(lambda x: x[0:n].values, all_sorted_scores)
                top_LS = pd.DataFrame.from_dict({"PC": all_top_n, "Score": all_top_n_scores})
                #Gather the bottom "n" components and their scores
                if bottom:
                    all_bot_n = pool.map(lambda x: x[-n:].index.values, all_sorted_scores)
                    all_bot_n_scores = pool.map(lambda x: x[-n:].values, all_sorted_scores)
                    bot_LS = pd.DataFrame.from_dict({"PC": all_bot_n, "Score": all_bot_n_scores})
                    return top_LS, bot_LS
            return top_LS
        
        #Gather and return loading scores for a given PC
        else:
            PC = int(PC)
            loading_scores = pd.Series(self.pca.components_[PC])
            # sort loading scores by magnitude
            sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)
            # get names
            top_n = sorted_loading_scores[0:n].index.values
            top_n_scores = sorted_loading_scores[0:n].values
            top_LS = pd.DataFrame.from_dict({"PC": top_n, "Score": top_n_scores})
            if bottom:
                bot_n = sorted_loading_scores[-n:].index.values
                bot_n_scores = sorted_loading_scores[-n:].values
                bot_LS = pd.DataFrame.from_dict({"PC": bot_n, "Score": bot_n_scores})
                return top_LS, bot_LS
            return top_LS

    #FUNCTION DESCRIPTION
    #Parameters:
    #   -pca_df: pandas DataFrame;
    #   -wd: str;
    #   -seq: str;
    #   -temp: str;
    #   -pH: str;
    def PC_den_plt(self, seq='', temp='', pH=''):
        file = 'PCA Density Plot' + self.__name_append(seq, temp, pH)
    
        pc1 = self.pca_df.PC1
        pc2 = self.pca_df.PC2
    
        #For a single PC plot
        f = plt.figure()
        sns.kdeplot(pc1, pc2, shade = True, shade_lowest = False, n_levels = 75,
                    cmap = "terrain_r", cbar = True)
        
        #For Multiple PC plots
        #    f, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2, 2)
        #    
        #    sns.kdeplot(pc1, pc2, shade = True, n_levels = 75, cmap = "Purples", ax = ax1)
        #    sns.kdeplot(pc1, pc2, shade = True, n_levels = 75, cmap = "Blues", ax = ax2)
        #    sns.kdeplot(pc1, pc2, shade = True, n_levels = 75, cmap = "GnBu", ax = ax3)
        #    sns.kdeplot(pc1, pc2, shade = True, n_levels = 75, cmap = "Oranges", ax = ax4)

        f.savefig(self.sdir + file + '.png', pad_inches = 0)
        plt.close()
        
    #FUNCTION DESCRIPTION    
    #Parameters:
    #   -pca: list of pandas DataFrame [0] and sklearn pca object [1]; 
    #   -wd: str;
    #   -seq: str;
    #   -temp: str;
    #   -pH: str;
    def PC_prob_map(self, seq='', temp='', pH=''):
        file = 'PCA Probability Map' + self.__name_append(seq, temp, pH)
        
        per_var = np.round(self.pca[1].explained_variance_ratio_ * 100, decimals = 1)

        xmin = self.pca_df.PC1.min()
        xmax = self.pca_df.PC1.max()
        ymin = self.pca_df.PC2.min()
        ymax = self.pca_df.PC2.max()
        X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([X.ravel(), Y.ravel()])
        values = np.vstack([self.pca_df.PC1, self.pca_df.PC2])
        kernel = gaussian_kde(values)
        Z = np.reshape(kernel(positions).T, X.shape)
        Z1 = Z*((xmax-xmin)/100)*((ymax-ymin)/100)
        
        ax = sns.heatmap(Z1, cmap="BuPu")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        plt.title('Title')
        plt.xlabel('PC1 - {0}%'.format(per_var[0]))
        plt.ylabel('PC2 - {0}%'.format(per_var[1]))
        plt.savefig(self.sdir + file + '.png', pad_inches = 0)
        plt.close()
    
    #FUNCTION DESCRIPTION
    #Parameters:
    #   -pca: list of pandas DataFrame [0] and sklearn pca object [1]; 
    #   -wd: str;
    #   -seq: str;
    #   -temp: str;
    #   -pH: str;
    def gen_2d_PCA_gif(self, seq='', temp='', pH=''):        
        #Appropriately name the file
        file = 'Evolution of Conformation in PC Space' + self.__name_append(seq, temp, pH)
        
        #Break down PCA input data to relevant pieces
        per_var = np.round(self.pca[1].explained_variance_ratio_ * 100, decimals = 1)

        #Select time frames to highlight in the gif
        frame_list = np.linspace(0, len(self.pca_df), num = len(self.pca_df)/10)[:-1]
        with mp.Pool() as pool:    
            frames = pool.map(lambda x: int(x), frame_list)
        names = []
        #Generate all (progressive) scatter plot images
        for frame in frames:
            plt.scatter(self.pca_df.PC1[:frame], self.pca_df.PC2[:frame], c = 'b', s = 0.01)
            #Integrate a plt_by_sim functionality here to clarify sudden jumps
            plt.scatter(self.pca_df.PC1[frame], self.pca_df.PC2[frame], c = 'r')
            plt.xlim(-5, 8)
            plt.ylim(-5, 8)
            plt.xlabel('PC1 - {0}%'.format(per_var[0]))
            plt.ylabel('PC2 - {0}%'.format(per_var[1]))
            plt.title(file + ' - Frame ' + f'{frame}')
            plt.savefig(self.sdir + file + ' - Frame ' + f'{frame}' + '.png', pad_inches = 0)
            plt.close()
            names.append(file + ' - Frame ' + f'{frame}' + '.png')
    
        #Concatenate the images into a gif
        with imageio.get_writer(self.sdir + file + '.gif', mode = 'I',
                                duration = 0.005) as writer:
            for filename in names:
                image = imageio.imread(self.sdir + filename)
                writer.append_data(image)
    
        #Delete all individual images
        for frame in frames:
            os.remove(self.sdir + file + ' - Frame ' + f'{frame}' + '.png')

class GMM_Clustering():
    def __init__(self, in_file, pca_df, save_dir):
        self.file = in_file
        self.pca_df = pca_df
        self.sdir =  save_dir

    #Plot all points in a color clustered PC graph
    #Parameters:
    #   -
    #Returns:
    #   -
    def __plot_clusters(self, X, Y_, means, covariances, index, title):
        #Colors of clusters
        color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange', 'crimson', 'g', 'darkviolet',
                              'darkgoldenrod', 'teal', 'purple', 'burlywood'])

        #NEED COMMENTS
        splot = plt.subplot(1, 1, 1 + index)
        for i, (mean, covar, color) in enumerate(zip(
                means, covariances, color_iter)):
            v, w = linalg.eigh(covar)
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            u = w[0] / linalg.norm(w[0])
            # as the DP will not use every component it has access to
            # unless it needs it, we shouldn't plot the redundant
            # components.
            if not np.any(Y_ == i):
                continue
            plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], s=.1, color=color)

            # Plot an ellipse to show the Gaussian component
            angle = np.arctan(u[1] / u[0])
            angle = 180. * angle / np.pi  # convert to degrees
            ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
            ell.set_clip_box(splot.bbox)
            ell.set_alpha(0.5)
            splot.add_artist(ell)
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        plt.xticks(())
        plt.yticks(())
        plt.title(title)
        plt.savefig(self.sdir + title + '.png')
    
    #Cluster PC points based on Gaussian Mixture Similarities
    #Parameters:
    #   -
    #Returns:
    #   -
    def cluster_PCs(self, n, dim):
        name = os.path.basename(self.file)[:-4]
        
        #Use Gaussian Mixture Clustering to cluster PC data
        gmm = mixture.GaussianMixture(n_components = n).fit(self.pca_df.iloc[:,0:dim].values)
        #Plot the Clusters
        self.__plot_clusters(self.pca_df.iloc[:,0:dim].values, gmm.predict(self.pca_df.iloc[:,0:dim].values), 
                     gmm.means_, gmm.covariances_, 0, 'Gaussian Mixture - ' + name, self.sdir)
        
        #Collect a dataframe of the cluster each point is placed in
        predictions = gmm.predict(self.pca_df.iloc[:,0:dim].values)
        
        #Generate labels for a bar plot
        cats = []
        for i in range(n):
            cats.append(f'{i}')
        
        #Count how many points are in each cluster
        counts = {}
        for i in range(len(predictions)):
            key = f'{predictions[i]}'
            if key in counts:
                counts[key] += 1
            else:
                counts[key] = 1
    
        #Plot a cluster density bar chart
        plt.close()
        plt.bar(cats, height = counts.values())
        plt.title('Cluster Distribution - ' + name)
        plt.xlabel('Clusters')
        plt.ylabel('Number of Frames') 
        plt.savefig(self.sdir + 'Cluster Density - ' + name + '.png')
    
        return predictions, gmm.means_, gmm.covariances_
    
    #Calculate the proportions of each cluster made up by each simulation
    #Parameters:
    #   -
    #Returns:
    #   -
    def clust_prop(nc, pred, sims):
        #
        n_sims = len(sims)
        sim_size = len(pred)/n_sims
        cp_d = {}
        cp_d['Simulation'] = sims
        for cluster in range(nc):
            frames = []
            for frame in range(len(pred)):
                if pred[frame] == cluster:
                    frames.append(frame)
        
            min_f = 0
            max_f = sim_size
            sim_frames = {}
            for i in range(n_sims):
                sim_frames[i] = []
                for f in frames:
                    if f>= min_f and f < max_f:
                        sim_frames[i].append(f)
                    min_f += sim_size
                    max_f += sim_size
        
            cp_d[cluster] = []
            for i in range(n_sims):
                cp_d[cluster].append(len(sim_frames[i])/len(frames))
    
        cp_df = pd.DataFrame.from_dict(cp_d, orient = 'index')
        return cp_df
    
class GMM_Transitions():
    #FUNCTION DESCRIPTION
    #Parameters:
    #   -
    #Returns:
    #   -
    def count_trans(pred):
        trans_count = 0
    
        for i in range(len(pred[:])):
            if i == len(pred[:]) - 1:
                continue
            elif pred[i+1] != pred[i]:
                trans_count += 1
        return trans_count
    
    #FUNCTION DESCRIPTION
    #Parameters:
    #   -
    #Returns:
    #   -
    def transition_frequency(total, unique):
        tf = []
        for key in list(unique.keys()):
            tf.append(unique[key]/total)
        return tf

    #FUNCTION DESCRIPTION
    #Parameters:
    #   -
    #Returns:
    #   -
    def plot_tf(trans, tf, wd, name):
        plt.close()
        plt.bar(x = list(trans.keys()), height = tf)
        plt.xticks(rotation = 75)
        plt.title('Transition Frequencies Between Clusters')
        plt.xlabel('Transition')
        plt.ylabel('Frequency')
        plt.savefig(wd + 'Transition Frequencies Between Clusters - ' + name + '.png')
        plt.close()
    
class Ramachandran():
    def __init__(self, save_dir):
        self.sdir = save_dir
    
    #Create a Ramachandran Plot for one set of phi/psi angles
    #Parameters:
    #   -
    #Returns:
    #   -
    def plt_one(self, angles, res, name):
        #Load Angles
        phis = angles['phi' + res].dropna(axis = 0)
        psis = angles['psi' + res].dropna(axis = 0)
    
        #Calculate point density
        phipsi = np.vstack([phis, psis])
        density = stat.gaussian_kde(phipsi)(phipsi)
    
        #Create Scatter Plot
        fig, ax = plt.subplots()
        ax.scatter(phis, psis, c = density, s = 100, edgecolor = '')
    
        #REPLACE WITH SNS.KDE_PLOT INSTEAD
        
        #Create Title, labels, and axes
        plt.title('Residue ' + res + ' Ramachandran')
        ax.set_xlabel("phi")
        ax.set_ylabel("psi")
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        ax.axhline(0, color = 'black')
        ax.axvline(0, color = 'black')
        
        #Save the figure and close it
        plt.savefig(self.sdir + name + ' - Residue ' + res + ' Ramachandran.png')
        plt.close()
        
    #Generates plots for all residues of the chain
    #Parameters:
    #   -
    #Returns:
    #   -
    def plt_all(self, angles, file):    
        #Gather all amino acid residues to plot
        res_list = np.linspace(1, len(angles.columns)//2, num = len(angles.columns)//2)
        with mp.Pool() as pool:
            res_list = pool.map(lambda x: int(x), res_list)
            res_list = pool.map(lambda x: str(x), res_list)
        
        #Plot each residue individually
        for res in res_list:
            self.plt_one(angles, res, file)
    
    #Generates plots for each cluster in PC space
    #Parameters:
    #   -
    #Returns:
    #   -
    def plt_clust(self, gmm, angles):
        #Collects the number of clusters
        nc = gmm[1]
        #Drops all EoS lines
        clst_angles = angles.dropna(axis = 0)
        #Appends a column with predictions to each frame
        clst_angles['Clusters'] = gmm[0][0]
        #Gather the frames for each cluster, then plot all for each grouping
        for i in range(nc):
            cluster_phi_psi = clst_angles[clst_angles['Cluster'] == i]
            self.plt_all(cluster_phi_psi, 'Cluster ' + f'{i}')
            
class Hilbert():
    def __init__(self, data_2d):
        self.data = data_2d
        
    ## ROT rotates and flips a quadrant appropriately.
    #  Parameters:
    #    Input, integer N, the length of a side of the square.  
    #    N must be a power of 2.
    #    Input/output, integer X, Y, the coordinates of a point.
    #    Input, integer RX, RY, ???
    def __rot(self, n, x, y, rx, ry):
        if (ry == 0):
            #Reflect.
            if (rx == 1):
                x = n - 1 - x
                y = n - 1 - y
            #Flip.
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
    def __xy2d(self, x, y):
        
        m = 10    # index of hilbert curve
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
            xcopy, ycopy = self.__rot(s, xcopy, ycopy, rx, ry)
            s = ( s // 2 )
        return d

    #FUNCTION DESCRIPTION
    #Parameters:
    #   -
    #Returns:
    #   -
    def hilb_curve(self):
        #Drop all EoS rows
        data_2d = self.data.dropna(axis = 0)
        
        #Transform and round data to integer values into pixel space
        #   - Normalize data to be between 0 and 1
        #   - multiplying by 1023 because we are using order 10 (0-1023 is 1024)
        xmin = min(data_2d.min())
        xmax = max(data_2d.max())
        transformed_data = data_2d.apply(lambda x : (x-xmin)*1023/(xmax-xmin))
        transformed_data = transformed_data.apply(lambda x: np.int64(x))

        #Combine pairs of values into one column
        combined_data = pd.DataFrame(index = transformed_data.index)
        label_count = 1
        for i in range(len(transformed_data.columns)):
            if 'Sin' in list(transformed_data.columns)[i]:
                if label_count % 2 != 0:
                    combined_data['phi'+str(label_count//2 + 1)]=transformed_data.iloc[:,i:i+2].values.tolist()
                else:
                    combined_data['psi'+str(label_count//2)]=transformed_data.iloc[:,i:i+2].values.tolist()
                label_count += 1
            elif 'phi' in transformed_data.columns[i] and 'Cos' not in transformed_data.columns[i]:
                combined_data['AA'+str(i//2 + 1)]=transformed_data.iloc[:,i:i+2].values.tolist()
            else:
                continue
        
        #Convert 2d into 1d
        hilbert_data = np.zeros((len(combined_data), len(combined_data.columns)))
        for i in range(len(combined_data)):
            for j in range(len(combined_data.columns)):
                hilbert_data[i, j] = self.__xy2d(combined_data.iloc[i,j][0],combined_data.iloc[i,j][1])
 
        #Add index and column titles to hilbert data
        hilbert_data=pd.DataFrame(hilbert_data, columns = combined_data.columns)
        
        return hilbert_data

class RMSD():
    def __init__(self, angle_df):
        self.angles = angle_df.dropna(axis = 0)
        
    #
    #Parameters:
    #   -angles: pandas DataFrame; contains dihedral angles for all frames of the peptide
    #   -row: 
    #Returns:
    #   -rmsd_vals:
    def __rmsd_calc(angles, row):
        rmsd_vals = []
        remaining = angles.index[row:]
        for r in remaining:
            rmsd_vals.append(np.sqrt(((angles.iloc[row,:] - angles.iloc[r,:])**2).mean()))
        
        return rmsd_vals
    
    #Calculates the RMSD of dihedral angles for a given peptide dataset
    #Parameters:
    #   -angles: pandas DataFrame; contains dihedral angles for all frames of the peptide
    #   -frames: list of int; frames to be compared for rmsd. Defaults to all frames.
    #Returns:
    #   -rmsd_df: pandas DataFrame; contains rmsd between the frame sets
    def get_dihedral_rmsd(self, frames = [], seq = '', pH = '', temp = ''):        
        if not frames:
            #DO RMSD COMPARISON ON Dihedral Angles FOR ALL FRAMES
            rmsd_df = pd.DataFrame(index = self.angles.index, columns = self.angles.index)
#            np.fill_diagonal(rmsd_df.values, 0)
            rmsd_df.apply(lambda x: self.__rmsd_calc(self.angles, x.index), axis = 1,
                          result_type = 'broadcast')
# =============================================================================
#             for i in range(len(self.angles)):
#                 for j in range(len(self.angles)):
#                     rmsd_df.iloc[i, j] = np.sqrt(((self.angles.iloc[i,:] - self.angles.iloc[j,:]) ** 2).mean())
# =============================================================================
        else:
            #DO RMSD COMPARISON ON Dihedral Angles FOR PASSED FRAMES
            rmsd_df = pd.DataFrame(index = frames, columns = frames)
            for i in frames:
                for j in frames:
                    rmsd_df.iloc[i, j] = np.sqrt(((self.angles.iloc[i,:] - self.angles.iloc[j,:]) ** 2).mean())
        
        #Plot the RMSD Heatmap
        hmap = sns.heatmap(rmsd_df, cmap = "Blues")
        hmap.set_title("Dihedral RMSD Heatmap - ")
        hmap.set_xlabel("")
        hmap.set_ylabel("")
        plt.savefig()
        plt.close()
        
        return rmsd_df
    
    #Calculates the RMSD of Alpha Carbons for a given peptide dataset
    #Parameters:
    #   -angles: pandas DataFrame; contains dihedral angles for all frames of the peptide
    #   -frames: list of int; frames to be compared for rmsd. Defaults to all frames.
    #Returns:
    #   -rmsd_df: pandas DataFrame; contains rmsd between the frame sets
    def get_CA_rmsd(self, frames = [], seq = '', pH = '', temp = ''):
        if not frames:
            rmsd_df = pd.DataFrame(index = self.angles.index, columns = self.angles.index)
            #DO RMSD COMPARISON ON C(alpha) FOR ALL FRAMES
            
        else:
            rmsd_df = pd.DataFrame(index = frames, columns = frames)
            #DO RMSD COMPARISON ON C(alpha) FOR PASSED FRAMES
            
        
        #Plot the RMSD Heatmap
        sns.heatmap(rmsd_df)
        return rmsd_df
    
class Transformations():
    def __init(self, angle_data):
        self.data = angle_data
        
    def hilb_transform_row(self, imag = False, real = False): #Only ONE (real or imag) can be True
        hil=[]
        
        if real and imag:
            raise Exception("Please select only EITHER real or imaginary values to display.")
        elif imag:
            for i in range(len(self.data)):
                hil.append(np.imag(signal.hilbert(self.data.iloc[i,:])))
        elif real:
            for i in range(len(self.data)):
                hil.append(np.real(signal.hilbert(self.data.iloc[i,:])))
        else:
            for i in range(len(self.data)):
                hil.append(np.abs(signal.hilbert(self.data.iloc[i,:])))
        
        hil=pd.DataFrame(hil,index=self.data.index,columns=self.data.columns)

        hilcols= []
        for c in self.data.columns:
            hilcols.append(c+'_imag')
        hil.columns = hilcols
        hil = pd.concat([self.data,hil],axis=1)

        return hil
    
    def hilb_transform_col(self, imag = False, real = False):
        hil=[]
        
        if real and imag:
            raise Exception("Please select only EITHER real or imaginary values to display.")
        elif imag:
            for col in self.data.columns:
                hil[col] = np.imag(signal.hilbert(self.data[col].values))
        elif real:
            for col in self.data.columns:
                hil[col] = np.real(signal.hilbert(self.data[col].values))
        else:
            for col in self.data.columns:
                hil[col] = np.abs(signal.hilbert(self.data[col].values))
        
        hil=pd.DataFrame(hil,index=self.data.index,columns=self.data.columns)

        hilcols= []
        for c in self.data.columns:
            hilcols.append(c+'_imag')
        hil.columns = hilcols
        hil = pd.concat([self.data,hil],axis=1)

        return hil

    def hilb_transform(self, imag = False, real = False):
        if real and imag:
            raise Exception("Please select only EITHER real or imaginary values to display.")
        elif imag:
            htrow = self.hilb_transform_row(imag = True)
            htcol = self.hilb_transform_col(imag = True)
        elif real:
            htrow = self.hilb_transform_row(real = True)
            htcol = self.hilb_transform_col(real = True)
        else:
            htrow = self.hilb_transform_row()
            htcol = self.hilb_transform_col()
        
        hil=pd.concat([htrow, htcol], axis = 1)       
        
        return hil
    
    def four_transform(self):
        ft_df = signal.fft(self.data)
        return ft_df
    
    def wavelet_transform(self):
#        signal.cwt()
        return