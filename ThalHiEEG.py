import numpy as np
from numpy import average, std
import pickle
from numpy.random import random, randint, normal, shuffle,uniform
import scipy
from scipy.stats import ttest_ind
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram,linkage
from scipy.stats.mstats import zscore
import seaborn as sns
import fnmatch
import os  # handy system and path functions
import sys  # to get file system encoding
import csv
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
import pandas as pd  
import matplotlib
print('Python version ' + sys.version)
print('Pandas version ' + pd.__version__)
print('Matplotlib version ' + matplotlib.__version__)
import mne
#import FOOOF
from mne.time_frequency import tfr_morlet
plt.ion() #turning interactive plotter off
#print(matplotlib.is_interactive())
#matplotlib.use('Agg')


def save_object(obj, filename):
	''' Simple function to write out objects into a pickle file
	usage: save_object(obj, filename)
	'''
	with open(filename, 'wb') as output:
		pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
	#M = pickle.load(open(f, "rb"))	


def read_object(filename):
	''' short hand for reading object because I can never remember pickle syntax'''
	o = pickle.load(open(filename, "rb"))	
	return o



# where data are
ROOT='/data/backed_up/shared/ThalHi_data/eeg_preproc/'
print(os.listdir(ROOT))

#where data should go
OUT='/home/kahwang/bsh/ThalHi_data/TFR/'

# Compiling epochs
all_subs_cue={}
all_subs_probe={}
#all_subs_resp={}
for sub in os.listdir(ROOT):
    if sub not in ['73','200','201','103','96', '137', 'pilot', 'Em']:# and (sub=='Em' or sub=='112' or sub=='82'):#or sub=='80'): # sub 73 is the file with the 1 missing event, 103 and  96 are noisy subs
        this_sub_path=ROOT+sub
        
        all_subs_cue[sub]=mne.read_epochs(this_sub_path+'/cue-epo.fif')

        all_subs_probe[sub] = {}
        for condition in ['IDS', 'EDS', 'stay']:	
        	all_subs_probe[sub][condition] =  mne.read_epochs(this_sub_path+'/probe_'+condition+'_events-epo.fif')

	# When comibing cue and probe, we will probably have to rely on info in the "metadata" field, and match block/trial numbers to deal with the issue that there appears to be mismatch in #num of events in the epochs
	# look for: probe_EDS_events-epo.fif probe_IDS_events-epo.fif probe_stay_events-epo.fif
	# response lock is in "response events-epo.fif", not separately by IDS/EDS/stay
	# each subject has around 415 epochs (some rejected?)


### run trial by trial TFR, without any baseline, then save.
freqs=np.arange(.5,38.,2.)
n_cycles = freqs / 2.


for sub in all_subs_cue.keys():
	#tfi = tfr_morlet(all_subs_cue[sub], freqs=freqs, average=False,n_cycles=n_cycles, use_fft=True, return_itc=False, decim=1, n_jobs=12)
	#save_object(tfi, OUT+sub+'_cueTFR')  

	for condition in ['IDS', 'EDS', 'stay']:	
		tfi = tfr_morlet(all_subs_probe[sub][condition], freqs=freqs, average=False,n_cycles=n_cycles, use_fft=True, return_itc=False, decim=1, n_jobs=12)
		save_object(tfi, OUT+sub+'_' + condition + '_probeTFR') 

	# after saving to EpochTFR format, the triggers can be found in "trig_id", and can be selected by tfi['EDS_trig']
	# can then find meta data (performance) by doing tfi['IDS_trig'].metadata


### Contrast power between conditions for cue
cue_ave_TFR = {}
for sub in all_subs_cue.keys():
	cue_ave_TFR[sub] = {}
	tfi = read_object(OUT+sub+'_cueTFR')  

	for condition in ['EDS_trig', 'IDS_trig', 'Stay_trig']:
		cue_ave_TFR[sub][condition] =  tfi[condition][tfi[condition].metadata['trial_Corr']==1].average()  # not enough error trials to compare corr vs. error, so only pull #.apply_baseline(mode='logratio',baseline=[-0.8, -0.3])
		#cue_ave_TFR[sub][condition].data = cue_ave_TFR[sub][condition].data*10 #convert to db

# substract conditions within each subject
cue_ave_TFR_EDS_v_IDS = {}
cue_ave_TFR_IDS_v_Stay = {}
for sub in all_subs_cue.keys():
	cue_ave_TFR_EDS_v_IDS[sub] = cue_ave_TFR[sub]['EDS_trig'] - cue_ave_TFR[sub]['IDS_trig'] 
	cue_ave_TFR_IDS_v_Stay[sub] = cue_ave_TFR[sub]['IDS_trig'] - cue_ave_TFR[sub]['Stay_trig'] 

#grand average across subjects
group_ave_cue_TFR_IDS_v_Stay = mne.grand_average(list(cue_ave_TFR_IDS_v_Stay.values()))
group_ave_cue_TFR_EDS_v_IDS = mne.grand_average(list(cue_ave_TFR_EDS_v_IDS.values()))
group_ave_cue_TFR_EDS_v_IDS.plot_topo()
group_ave_cue_TFR_IDS_v_Stay.plot_topo()



### Contrast power bewteen conditions for probe (more like cue + probe)
probe_ave_TFR = {}
for sub in all_subs_cue.keys():

	probe_ave_TFR[sub] = {}
	
	for condition in ['IDS', 'EDS', 'stay']:
		tfi = read_object(OUT+sub+'_' + condition + '_probeTFR') 
		probe_ave_TFR[sub][condition] = tfi[tfi.metadata['trial_Corr']==1].average()


# substract conditions within each subject
probe_ave_TFR_EDS_v_IDS = {}
probe_ave_TFR_IDS_v_Stay = {}
for sub in all_subs_probe.keys():
	probe_ave_TFR_EDS_v_IDS[sub] = probe_ave_TFR[sub]['EDS'] - probe_ave_TFR[sub]['IDS'] 
	probe_ave_TFR_IDS_v_Stay[sub] = probe_ave_TFR[sub]['IDS'] - probe_ave_TFR[sub]['Stay'] 


group_ave_probe_TFR_IDS_v_Stay = mne.grand_average(list(probe_ave_TFR_IDS_v_Stay.values()))
group_ave_probe_TFR_EDS_v_IDS = mne.grand_average(list(probe_ave_TFR_EDS_v_IDS.values()))
group_ave_probe_TFR_EDS_v_IDS.plot_topo()
group_ave_probe_TFR_IDS_v_Stay.plot_topo()







    
