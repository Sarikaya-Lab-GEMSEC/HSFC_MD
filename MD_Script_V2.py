# =============================================================================
# Created on Thu Sep 26 15:34:52 2019
# Last Edited on Thu Oct 10 12:45 2019
# 
# @author: Pedro Fischer Marques
# GEMSEC Labs - Computational Group
# =============================================================================
import easygui as eg
import time
from MD_Analysis import *

pdb = eg.fileopenbox(msg = "Select the PDB file to be analyzed")
save_dir = eg.diropenbox(msg = "Select the save directory") + "\\"

AC = Angle_Calc(pdb)
AC.get_phi_psi()
sc_df = AC.get_sin_cos(AC.angles)
hil_df = Hilbert(sc_df).hilb_curve()
hil2_df = Hilbert(hil_df).hilb_curve()

DR_sc = Dim_Reduction(pdb, sc_df, save_dir)
t1 = time.time()
DR_sc.pca("2d")
t2 = time.time()
DR_sc.ICA(2)
t3 = time.time()
#DR_sc.Multi_Dim_Scaling(2)
DR_sc.IsoMap(2)
t4 = time.time()
DR_sc.Spectral_Embed(2)
t5 = time.time()
DR_sc.t_SNE(2)
t6 = time.time()

sc_pca_t = t2 - t1
sc_ICA_t = t3 - t2
sc_IM_t = t4 - t3
sc_SE_t = t5 - t4
sc_tSNE_t = t6 - t5

DR_hil_1 = Dim_Reduction(pdb, hil_df, save_dir)
t1 = time.time()
DR_hil_1.pca("2d")
t2 = time.time()
DR_hil_1.ICA(2)
t3 = time.time()
#DR_hil_1.Multi_Dim_Scaling(2)
DR_hil_1.IsoMap(2)
t4 = time.time()
DR_hil_1.Spectral_Embed(2)
t5 = time.time()
DR_hil_1.t_SNE(2)
t6 = time.time()

hil1_pca_t = t2 - t1
hil1_ICA_t = t3 - t2
hil1_IM_t = t4 - t3
hil1_SE_t = t5 - t4
hil1_tSNE_t = t6 - t5

DR_hil_2 = Dim_Reduction(pdb, hil2_df, save_dir)
t1 = time.time()
DR_hil_2.pca("2d")
t2 = time.time()
DR_hil_2.ICA(2)
t3 = time.time()
#DR_hil_2.Multi_Dim_Scaling(2)
DR_hil_2.IsoMap(2)
t4 = time.time()
DR_hil_2.Spectral_Embed(2)
t5 = time.time()
DR_hil_2.t_SNE(2)
t6 = time.time()

hil2_pca_t = t2 - t1
hil2_ICA_t = t3 - t2
hil2_IM_t = t4 - t3
hil2_SE_t = t5 - t4
hil2_tSNE_t = t6 - t5