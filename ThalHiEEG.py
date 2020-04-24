import numpy as np
from numpy import average, std
import pickle
from numpy.random import random, randint, normal, shuffle,uniform
import scipy
from scipy import sparse
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
import mne
#import FOOOF
from mne.time_frequency import tfr_morlet
plt.ion() #turning interactive plotter off
#print(matplotlib.is_interactive())
#matplotlib.use('Agg')

# where data are
ROOT='/data/backed_up/shared/ThalHi_data/eeg_preproc/'
print(os.listdir(ROOT))

#where data should go
OUT='/home/kahwang/bsh/ThalHi_data/TFR/'

# Compiling epochs
all_subs_cue = {}
all_subs_probe = {}
all_subs_ITI = {}
#all_subs_resp={}

included_subjects = ['128', '112', '108', '110', '120', '98', '86', '82', '115', '94', '76', '91', '80', '95', '121', '114', '125', '70',
'107', '111', '88', '113', '131', '130', '135', '140', '167', '145', '146', '138', '147', '176', '122', '118', '103', '142']


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


def mirror_evoke(ep):
	
	e = ep.copy()
	nd = np.concatenate((np.flip(e._data[:,:,e.time_as_index(0)[0]:e.time_as_index(1.5)[0]], axis=2), e._data, np.flip(e._data[:,:,e.time_as_index(e.tmax-1.5)[0]:e.time_as_index(e.tmax)[0]],axis=2)),axis=2)
	tnmin = e.tmin - 1.5
	tnmax = e.tmax + 1.5 
	e._set_times(np.arange(tnmin,tnmax+e.times[2]-e.times[1],e.times[2]-e.times[1]))
	e._data = nd

	return e

def mirror_iti(ep):
	
	e = ep.copy()
	nd = np.concatenate((np.flip(e._data[:,:,e.time_as_index(0)[0]:e.time_as_index(3)[0]], axis=2), e._data, np.flip(e._data[:,:,e.time_as_index(e.tmax-3)[0]:e.time_as_index(e.tmax)[0]],axis=2)),axis=2)
	tnmin = e.tmin - 3
	tnmax = e.tmax + 3 
	e._set_times(np.arange(tnmin,tnmax+e.times[2]-e.times[1],e.times[2]-e.times[1]))
	e._data = nd

	return e


def load_epochs(included_subjects):

	for sub in included_subjects:
	    if sub not in ['73','96','137', '143', '200', '201', 'pilot', 'Em', 'ITI_epochs']:# and (sub=='Em' or sub=='112' or sub=='82'):#or sub=='80'): # sub 73 is the file with the 1 missing event, 103 and  96 are noisy subs. 200 and 201 are patients.
	        this_sub_path=ROOT+sub
	        all_subs_cue[sub]=mne.read_epochs(this_sub_path+'/cue-epo.fif')

	        iti_path = ROOT + 'ITI_epochs/' + sub
	        all_subs_ITI[sub]=mne.read_epochs(iti_path+'/ITI-epo.fif')

	        all_subs_probe[sub] = {}
	        for condition in ['IDS', 'EDS', 'stay']:	
	        	all_subs_probe[sub][condition] =  mne.read_epochs(this_sub_path+'/probe_'+condition+'_events-epo.fif')

	# When comibing cue and probe, we will probably have to rely on info in the "metadata" field, and match block/trial numbers to deal with the issue that there appears to be mismatch in #num of events in the epochs
	# look for: probe_EDS_events-epo.fif probe_IDS_events-epo.fif probe_stay_events-epo.fif
	# response lock is in "response events-epo.fif", not separately by IDS/EDS/stay
	# each subject has around 415 epochs (some rejected?)

	return all_subs_cue, all_subs_ITI, all_subs_probe


########################################################################
### run trial by trial TFR, without any baseline, then save.
########################################################################

def run_TFR():
	#freqs=np.arange(1,40.,1.)
	#n_cycles = 6 #freqs / 2.

	freqs = np.logspace(*np.log10([1, 40]), num=30)
	n_cycles = 6#freqs / 2.

	for sub in all_subs_cue.keys():
		
		#tfr on cue
		tfi = tfr_morlet(mirror_evoke(mirror_evoke(mirror_evoke(all_subs_cue[sub].crop(tmin=0, tmax=1.5)))), freqs=freqs, average=False,n_cycles=n_cycles, use_fft=True, return_itc=False, decim=1, n_jobs=24)
		# double mirror, then crop
		tfi = tfi.crop(tmin = 0, tmax = all_subs_cue[sub].tmax)
		save_object(tfi, OUT+sub+'_cueTFR')  

		#tfr on iti
		tfi = tfr_morlet(mirror_iti(mirror_iti(all_subs_ITI[sub])), freqs=freqs, average=False,n_cycles=n_cycles, use_fft=True, return_itc=False, decim=1, n_jobs=24)
		# double mirror, then crop
		tfi = tfi.crop(tmin = all_subs_ITI[sub].tmin, tmax = all_subs_ITI[sub].tmax)
		save_object(tfi, OUT+sub+'_itiTFR')  

		#tfr on probe
		for condition in ['IDS', 'EDS', 'stay']:	
			tfi = tfr_morlet(mirror_evoke(mirror_evoke(mirror_evoke(all_subs_probe[sub][condition].crop(tmin=-.7, tmax=3)))), freqs=freqs, average=False,n_cycles=n_cycles, use_fft=True, return_itc=False, decim=1, n_jobs=24)
			tfi = tfi.crop(tmin = -.7, tmax = all_subs_probe[sub][condition].tmax)
			save_object(tfi, OUT+sub+'_' + condition + '_probeTFR') 

	#after saving to EpochTFR format, the triggers can be found in "trig_id", and can be selected by tfi['EDS_trig']
	#can then find meta data (performance) by doing tfi['IDS_trig'].metadata



########################################################################
### Load TFR without normalization for fooof
########################################################################
def run_spect_foof():
	import fooof

	cue_ave_TFR = {}
	for sub in all_subs_cue.keys():
		cue_ave_TFR[sub] = {}
		try:
			tfi = read_object(OUT+sub+'_cueTFR')  

			for condition in ['EDS_trig', 'IDS_trig', 'Stay_trig']:
				cue_ave_TFR[sub][condition] =  tfi[condition][tfi[condition].metadata['trial_Corr']==1].average()
		except:
			continue		


	# substract conditions within each subject
	cue_ave_TFR_EDS = {}
	cue_ave_TFR_IDS = {}
	cue_ave_TFR_Stay = {}

	for sub in cue_ave_TFR.keys():
		cue_ave_TFR_EDS[sub] = cue_ave_TFR[sub]['EDS_trig']
		cue_ave_TFR_IDS[sub] = cue_ave_TFR[sub]['IDS_trig']
		cue_ave_TFR_Stay[sub] = cue_ave_TFR[sub]['Stay_trig']


	#grand average across subjects
	group_ave_cue_TFR_EDS = mne.grand_average(list(cue_ave_TFR_EDS.values()))
	group_ave_cue_TFR_IDS = mne.grand_average(list(cue_ave_TFR_IDS.values()))
	group_ave_cue_TFR_Stay = mne.grand_average(list(cue_ave_TFR_Stay.values()))	

	return group_ave_cue_TFR_EDS, group_ave_cue_TFR_IDS, group_ave_cue_TFR_Stay



########################################################################
### Extract behavioral data from metadata field
########################################################################

def run_behav(all_subs_cue):
	all_subs_cue, _, _ = load_epochs(included_subjects)

	Behav = {}
	Behav['Acc'] = {'EDS': np.zeros(34), 'IDS': np.zeros(34), 'Stay': np.zeros(34)} 
	Behav['RT'] = {'EDS': np.zeros(34), 'IDS': np.zeros(34), 'Stay': np.zeros(34)} 

	for i,sub in enumerate(all_subs_cue.keys()):
		Behav['Acc']['EDS'][i] = all_subs_cue[sub]['EDS_trig'].metadata['trial_Corr'].mean()
		Behav['Acc']['IDS'][i] = all_subs_cue[sub]['IDS_trig'].metadata['trial_Corr'].mean()
		Behav['Acc']['Stay'][i] = all_subs_cue[sub]['Stay_trig'].metadata['trial_Corr'].mean()

		#all_subs_cue[sub]['EDS_trig'].metadata['rt'] = all_subs_cue[sub]['EDS_trig'].metadata['rt'].convert_objects(convert_numeric=True)  
		#all_subs_cue[sub]['IDS_trig'].metadata['rt'] = all_subs_cue[sub]['IDS_trig'].metadata['rt'].convert_objects(convert_numeric=True) 
		#all_subs_cue[sub]['Stay_trig'].metadata['rt'] = all_subs_cue[sub]['Stay_trig'].metadata['rt'].convert_objects(convert_numeric=True) 

		Behav['RT']['EDS'][i] = all_subs_cue[sub]['EDS_trig'].metadata['rt'].convert_objects(convert_numeric=True).mean()
		Behav['RT']['IDS'][i] = all_subs_cue[sub]['IDS_trig'].metadata['rt'].convert_objects(convert_numeric=True).mean()
		Behav['RT']['Stay'][i] = all_subs_cue[sub]['Stay_trig'].metadata['rt'].convert_objects(convert_numeric=True).mean()

	return Behav	

# import scipy 
# scipy.stats.ttest_rel(Behav['RT']['EDS'], Behav['RT']['IDS'])  #***
# scipy.stats.ttest_rel(Behav['RT']['IDS'], Behav['RT']['Stay']) #****

# scipy.stats.ttest_rel(Behav['Acc']['EDS'], Behav['Acc']['IDS']) 
# scipy.stats.ttest_rel(Behav['Acc']['IDS'], Behav['Acc']['Stay']) #***
# scipy.stats.ttest_rel(Behav['Acc']['EDS'], Behav['Acc']['Stay']) #****

# RT
# EDS .89 .18
# IDS .84 .18
# Stay.79 .16

# Acc
# EDS .89 .09
# IDS .89 .08
# Stay.93 .07


	########################################################################
	### Contrast power bewteen conditions for probe
	########################################################################
def run_baselinecorr(db_baseline):	
	'''run and save baseline correction. Default to db'''

	probe_ave_TFR = {}
	db_baseline = True

	for sub in included_subjects:

		probe_ave_TFR[sub] = {}
		
		# append ITI data as baseline in tfi object
		if db_baseline:
			btfi = read_object(OUT+sub+'_itiTFR')
			btfi = np.mean(np.mean(btfi.data, axis=0), axis=2)
		
		for condition in ['IDS', 'EDS', 'stay']:
			tfi = read_object(OUT+sub+'_' + condition + '_probeTFR')
			
			if db_baseline:
				bp = np.repeat(np.repeat(btfi[np.newaxis, :,:], tfi.data.shape[0], axis=0)[:,:,:,np.newaxis], 100, axis=3)
				tfi.data = np.concatenate((bp, tfi.data), axis=3)
				tfi.times = np.arange(tfi.times[0]-(100*(tfi.times[-1]-tfi.times[-2])), tfi.times[-1]+tfi.times[-1]-tfi.times[-2], tfi.times[-1]-tfi.times[-2])
				probe_ave_TFR[sub][condition] = tfi[tfi.metadata['trial_Corr']==1].average().apply_baseline(mode='logratio',baseline=[-0.89, -0.7])
				probe_ave_TFR[sub][condition].data = probe_ave_TFR[sub][condition].data * 10 #convert to db
			else:
				probe_ave_TFR[sub][condition] = tfi[tfi.metadata['trial_Corr']==1].average()


	# substract conditions within each subject
	probe_ave_TFR_EDS = {}
	probe_ave_TFR_IDS = {}
	probe_ave_TFR_Stay = {}

	for sub in included_subjects:
		probe_ave_TFR_EDS[sub] = probe_ave_TFR[sub]['EDS'].copy()
		probe_ave_TFR_IDS[sub] = probe_ave_TFR[sub]['IDS'].copy()
		probe_ave_TFR_Stay[sub] = probe_ave_TFR[sub]['stay'].copy()

	### Because baseline corr takes so long, save to HD.
	if db_baseline:
		for sub in included_subjects:
			fn = '/home/kahwang/bsh/ThalHi_data/TFR/' + sub + '_EDS_probeTFRDBbc'
			probe_ave_TFR_EDS[sub].save(fn)
			fn = '/home/kahwang/bsh/ThalHi_data/TFR/' + sub + '_IDS_probeTFRDBbc'
			probe_ave_TFR_IDS[sub].save(fn)
			fn = '/home/kahwang/bsh/ThalHi_data/TFR/' + sub + '_Stay_probeTFRDBbc'
			probe_ave_TFR_Stay[sub].save(fn)	
	else:
		for sub in included_subjects:
			fn = '/home/kahwang/bsh/ThalHi_data/TFR/' + sub + '_EDS_probeTFRNobc'
			probe_ave_TFR_EDS[sub].save(fn)
			fn = '/home/kahwang/bsh/ThalHi_data/TFR/' + sub + '_IDS_probeTFRNobc'
			probe_ave_TFR_IDS[sub].save(fn)
			fn = '/home/kahwang/bsh/ThalHi_data/TFR/' + sub + '_Stay_probeTFRNobc'
			probe_ave_TFR_Stay[sub].save(fn)	


if __name__ == "__main__":


	########################################################################
	### Contrast power bewteen conditions for probe
	########################################################################

	probe_ave_TFR_EDS = {}
	probe_ave_TFR_IDS = {}
	probe_ave_TFR_Stay = {}
	probe_ave_TFR_EDS_v_IDS = {}
	probe_ave_TFR_IDS_v_Stay = {}

	for sub in included_subjects:
		
		fn = '/home/kahwang/bsh/ThalHi_data/TFR/' + sub + '_EDS_probeTFRNobc'
		probe_ave_TFR_EDS[sub] = mne.time_frequency.read_tfrs(fn)[0].crop(tmin = -0.65, tmax = 1.5)
		fn = '/home/kahwang/bsh/ThalHi_data/TFR/' + sub + '_IDS_probeTFRNobc'
		probe_ave_TFR_IDS[sub] = mne.time_frequency.read_tfrs(fn)[0].crop(tmin = -0.65, tmax = 1.5)
		fn = '/home/kahwang/bsh/ThalHi_data/TFR/' + sub + '_Stay_probeTFRNobc'
		probe_ave_TFR_Stay[sub] = mne.time_frequency.read_tfrs(fn)[0].crop(tmin = -0.65, tmax = 1.5)

		probe_ave_TFR_EDS_v_IDS[sub] = probe_ave_TFR_EDS[sub] - probe_ave_TFR_IDS[sub] 
		probe_ave_TFR_IDS_v_Stay[sub] = probe_ave_TFR_IDS[sub]  - probe_ave_TFR_Stay[sub]  
	
	group_ave_probe_TFR_IDS_v_Stay = mne.grand_average(list(probe_ave_TFR_IDS_v_Stay.values()))
	group_ave_probe_TFR_EDS_v_IDS = mne.grand_average(list(probe_ave_TFR_EDS_v_IDS.values()))
	group_ave_probe_TFR_EDS = mne.grand_average(list(probe_ave_TFR_EDS.values()))
	group_ave_probe_TFR_IDS = mne.grand_average(list(probe_ave_TFR_IDS.values()))
	group_ave_probe_TFR_Stay = mne.grand_average(list(probe_ave_TFR_Stay.values()))

				
	##### Code to try out spatiotemproal clustering 	
	# neighb file came from fieldtrip: https://github.com/fieldtrip/fieldtrip/blob/master/template/neighbours/biosemi64_neighb.mat
	ch_con, ch_names = mne.channels.read_ch_connectivity('biosemi64_neighb.mat') 
	ch_con = ch_con.toarray() #ch*ch connectivity matrix
	n_ch = ch_con.shape[0]
	n_freq = probe_ave_TFR_EDS[sub].data.shape[1]
	freq_con = np.eye(n_freq) + np.eye(n_freq, k=1) + np.eye(n_freq, k=-1)  #freq*freq connectivity matrix

	## attemp to create connectivity matrix that combines ch and freq
	# input into clustering needs to be in sub by time by freq*channel
	# original data shape for each subj is ch by freq by time
	# need to reshape into subj by time by 'freq*ch
	#now attempt to create the ((n_freq * n_ch) by (n_freq * n_ch)) con matrix

	con = np.zeros((n_freq * n_ch, n_freq * n_ch))
	# now fill this matrix based on ch_con and freq_con. 
	for row in np.arange(con.shape[0]):  
		for col in np.arange(con.shape[1]):  
			freq_ri, ch_ri = divmod(row, n_ch) #row is freq, and col is ch accoridng to the reshape behavior
			freq_ci, ch_ci = divmod(col, n_ch) #row is freq, and col is ch accoridng to the reshape behavior

			if freq_con[freq_ri, freq_ci] ==1 and ch_con[ch_ri, ch_ci] ==1: #A ch*freq pair can only be connected if both ch*ch and freq*freq con matrices are connected 
				con[row, col] = 1
	#sns.heatmap(con)  			
	con = sparse.csr_matrix(con)
	
	## Construct intput into permutation, annoyingly mne expect the data shape to be different from its own TFR object. UGH.
	# original data shape for each subj is ch by freq by time
	DEDS = np.zeros((len(probe_ave_TFR_IDS.keys()),group_ave_probe_TFR_EDS.data.shape[2], group_ave_probe_TFR_EDS.data.shape[1], group_ave_probe_TFR_EDS.data.shape[0])) 
	DIDS = np.zeros((len(probe_ave_TFR_EDS.keys()),group_ave_probe_TFR_EDS.data.shape[2], group_ave_probe_TFR_EDS.data.shape[1], group_ave_probe_TFR_EDS.data.shape[0]))
	DStay = np.zeros((len(probe_ave_TFR_EDS.keys()),group_ave_probe_TFR_EDS.data.shape[2], group_ave_probe_TFR_EDS.data.shape[1], group_ave_probe_TFR_EDS.data.shape[0]))
	
	# now need to be in subj by time by freq by ch
	for i, sub in enumerate(probe_ave_TFR_EDS.keys()):
		DEDS[i,:,:,:] = probe_ave_TFR_EDS[sub].data[:,:,:].transpose(2,1,0)
		DIDS[i,:,:,:] = probe_ave_TFR_IDS[sub].data[:,:,:].transpose(2,1,0)   
		DStay[i,:,:,:] = probe_ave_TFR_Stay[sub].data[:,:,:].transpose(2,1,0)
	# now then reshape into subj by time by freq*ch for the permutation
	DEDS = DEDS.reshape((DEDS.shape[0],DEDS.shape[1], DEDS.shape[2]*DEDS.shape[3]))
	DIDS = DIDS.reshape((DIDS.shape[0],DIDS.shape[1], DIDS.shape[2]*DIDS.shape[3]))
	DStay = DStay.reshape((DStay.shape[0],DStay.shape[1], DStay.shape[2]*DStay.shape[3]))
	
	# create contrast
	DEDSvIDS = DEDS - DIDS #EDS - IDS
	DIDSvStay = DIDS - DStay
	DEDSvStay = DEDS - DStay

	#output should go back to original data dimension, which is chn by freq by time
	#This is to try the tfce
	threshold_tfce = dict(start=0, step=0.2)
	
	def do_permutation(input, pthreshold, conmat, n_freq, n_ch):
		t_obs, clusters, cluster_pv, _ = mne.stats.spatio_temporal_cluster_1samp_test(input, threshold = pthreshold, step_down_p =.05, n_permutations=1024, n_jobs = 24, out_type='mask', connectivity = conmat, t_power = 1) 
		t_obs = t_obs.reshape((t_obs.shape[0], n_freq, n_ch)).T
		cl = np.where(cluster_pv < .05)[0]
		mask = np.sum(np.array(clusters)[cl], axis=0)
		mask = mask.reshape((mask.shape[0],n_freq, n_ch)).T

		return t_obs, mask	
	
	# permute EDS v IDS
	t_obs, mask = do_permutation(DEDSvIDS, threshold_tfce, con, n_freq, n_ch)	
	EDSvIDSplot = group_ave_probe_TFR_EDS_v_IDS.copy()
	EDSvIDSplot.data = t_obs *mask	
	EDSvIDStplot = group_ave_probe_TFR_EDS_v_IDS.copy()
	EDSvIDStplot.data = t_obs
	
	#permute IDS v stay
	t_obs, mask = do_permutation(DIDSvStay, threshold_tfce, con, n_freq, n_ch)	
	IDSvStayplot = group_ave_probe_TFR_EDS_v_IDS.copy()
	IDSvStayplot.data = t_obs *mask	
	IDSvStaytplot = group_ave_probe_TFR_EDS_v_IDS.copy()
	IDSvStaytplot.data = t_obs


	#permute EDS v stay
	t_obs, mask = do_permutation(DEDSvStay, threshold_tfce, con, n_freq, n_ch)
	EDSvStayplot = group_ave_probe_TFR_EDS_v_IDS.copy()
	EDSvStayplot.data = t_obs *mask	
	EDSvStaytplot = group_ave_probe_TFR_EDS_v_IDS.copy()
	EDSvStaytplot.data = t_obs
	

	#plotting
	# EDSvIDSplot.plot_topo()
	# EDSvIDStplot.plot_topo()
	# IDSvStayplot.plot_topo()
	# IDSvStaytplot.plot_topo()
	# EDSvStayplot.plot_topo()
	# EDSvStaytplot.plot_topo()	
	#group_ave_probe_TFR_EDS_v_IDS.plot_topo(tmin=-0.65, tmax=1.5)
	#group_ave_probe_TFR_IDS_v_Stay.plot_topo(tmin=-0.65, tmax=1.5)
	#group_ave_probe_TFR_EDS.plot_topo(tmin=-0.65, tmax=1.5)
	#group_ave_probe_TFR_IDS.plot_topo(tmin=-0.65, tmax=1.5)
	#group_ave_probe_TFR_Stay.plot_topo(tmin=-0.65, tmax=1.5)

	







	################################################################################
	################################################################################
	################################################################################
	#################### LEFT OVER
	################################################################################
	################################################################################
	################################################################################

	########################################################################
	### Contrast power between conditions for cue
	########################################################################

	# cue_ave_TFR = {}
	# for sub in all_subs_cue.keys():
	# 	cue_ave_TFR[sub] = {}
	# 	tfi = read_object(OUT+sub+'_cueTFR')  

	# 	# append ITI data as baseline in tfi object
	# 	btfi = read_object(OUT+sub+'_itiTFR')
	# 	btfi = np.mean(np.mean(btfi.data, axis=0), axis=2)
	# 	bp = np.repeat(np.repeat(btfi[np.newaxis, :,:], tfi.data.shape[0], axis=0)[:,:,:,np.newaxis], 100, axis=3)
	# 	tfi.data = np.concatenate((bp, tfi.data), axis=3)
	# 	tfi.times = np.arange(tfi.times[1]*(-100), tfi.times[-1] + tfi.times[1], tfi.times[1])

	# 	for condition in ['EDS_trig', 'IDS_trig', 'Stay_trig']:
	# 		cue_ave_TFR[sub][condition] =  tfi[condition][tfi[condition].metadata['trial_Corr']==1].average().apply_baseline(mode='logratio',baseline=[-0.1, 0])  # not enough error trials to compare corr vs. error, so only pull #.apply_baseline(mode='logratio',baseline=[-0.8, -0.3])
	# 		cue_ave_TFR[sub][condition].data = cue_ave_TFR[sub][condition].data*10 #convert to db

	# # substract conditions within each subject
	# cue_ave_TFR_EDS_v_IDS = {}
	# cue_ave_TFR_IDS_v_Stay = {}
	# cue_ave_TFR_EDS = {}
	# cue_ave_TFR_IDS = {}
	# cue_ave_TFR_Stay = {}

	# for sub in all_subs_cue.keys():
	# 	cue_ave_TFR_EDS_v_IDS[sub] = cue_ave_TFR[sub]['EDS_trig'] - cue_ave_TFR[sub]['IDS_trig'] 
	# 	cue_ave_TFR_IDS_v_Stay[sub] = cue_ave_TFR[sub]['IDS_trig'] - cue_ave_TFR[sub]['Stay_trig'] 
	# 	cue_ave_TFR_EDS[sub] = cue_ave_TFR[sub]['EDS_trig']
	# 	cue_ave_TFR_IDS[sub] = cue_ave_TFR[sub]['IDS_trig']
	# 	cue_ave_TFR_Stay[sub] = cue_ave_TFR[sub]['Stay_trig']


	# #grand average across subjects
	# group_ave_cue_TFR_IDS_v_Stay = mne.grand_average(list(cue_ave_TFR_IDS_v_Stay.values()))
	# group_ave_cue_TFR_EDS_v_IDS = mne.grand_average(list(cue_ave_TFR_EDS_v_IDS.values()))
	# group_ave_cue_TFR_EDS = mne.grand_average(list(cue_ave_TFR_EDS.values()))
	# group_ave_cue_TFR_IDS = mne.grand_average(list(cue_ave_TFR_IDS.values()))
	# group_ave_cue_TFR_Stay = mne.grand_average(list(cue_ave_TFR_Stay.values()))


	# group_ave_cue_TFR_EDS.plot_topo(title='EDS') #, vmin=-1, vmax=1)
	# group_ave_cue_TFR_IDS.plot_topo(title='IDS')#, vmin=-1, vmax=1)
	# group_ave_cue_TFR_Stay.plot_topo(title='Stay')#, vmin=-1, vmax=1)
	# group_ave_cue_TFR_EDS_v_IDS.plot_topo()
	# group_ave_cue_TFR_IDS_v_Stay.plot_topo()



	# #plot individual subject 
	# for sub in all_subs_cue.keys():
	# 	cue_ave_TFR_EDS_v_IDS[sub].plot_topo(title=sub, zlim=)



	# ### get delta and theta

	# fc = ['AF3', 'AFz', 'AF4', 'F3', 'F1', 'Fz', 'F2', 'F4', 'FC4', 'FC2', 'FCz', 'FC1', 'FC6', 'C2']
	# pc = ['P4', 'PO4', 'P6', 'PO8']

	# #chi = mne.pick_channels(ch_names=group_ave_probe_TFR_EDS.ch_names, include=fc, ordered=True)
	# chi = mne.pick_channels(ch_names=group_ave_probe_TFR_EDS.ch_names, include=fc, ordered=True)


	# power_df = pd.DataFrame()

	# for i, sub in enumerate(all_subs_probe.keys()):
	# 	tdf = pd.DataFrame()

	# 	for i, condition in enumerate(['E', 'I', 'S']):

	# 		if condition == 'E':
	# 			tdf.loc[i,'theta'] = np.mean(probe_ave_TFR_EDS[sub].data[chi, 1:5, 150:400])
	# 			tdf.loc[i,'delta'] = np.mean(probe_ave_TFR_EDS[sub].data[chi, 5:10, 150:400])
	# 			tdf.loc[i,'condition'] = 'EDF'
	# 		if condition == 'I':
	# 			tdf.loc[i,'theta'] = np.mean(probe_ave_TFR_IDS[sub].data[chi, 1:5, 275:400])
	# 			tdf.loc[i,'delta'] = np.mean(probe_ave_TFR_IDS[sub].data[chi, 5:10, 275:400])
	# 			tdf.loc[i,'condition'] = 'IDF'
	# 		if condition == 'S':
	# 			tdf.loc[i,'theta'] = np.mean(probe_ave_TFR_Stay[sub].data[chi, 1:5, 275:400])
	# 			tdf.loc[i,'delta'] = np.mean(probe_ave_TFR_Stay[sub].data[chi, 5:10, 275:400])
	# 			tdf.loc[i,'condition'] = 'Stay'
	# 	tdf['subject'] = sub

	# 	power_df = pd.concat([power_df, tdf])

	# 	# EDS_delta[i] = np.mean(probe_ave_TFR_EDS[sub].data[chi, 1:5, 275:400])
	# 	# EDS_theta[i] = np.mean(probe_ave_TFR_EDS[sub].data[chi, 5:10, 275:400])
	# 	# IDS_delta[i] = np.mean(probe_ave_TFR_IDS[sub].data[chi, 1:5, 275:400])
	# 	# IDS_theta[i] = np.mean(probe_ave_TFR_IDS[sub].data[chi, 5:10, 275:400])
	# 	# Stay_delta[i] = np.mean(probe_ave_TFR_Stay[sub].data[chi, 1:5, 275:400])
	# 	# Stay_theta[i] = np.mean(probe_ave_TFR_Stay[sub].data[chi, 5:10, 275:400])





	# ##### Check indiv subject plot
	# # for sub in all_subs_probe.keys():
	# # 	probe_ave_TFR[sub]['EDS'].plot_topo(title=sub, tmin=-0.67, tmax=1.5)





	# ########################################################################
	# ### Extract Timecourse from Sensors, into df, and do permutation
	# ########################################################################
	# fc = ['AF3', 'AFz', 'AF4', 'F3', 'F1', 'Fz', 'F2', 'F4', 'FC4', 'FC2', 'FCz', 'FC1', 'FC6', 'C2']
	# pc = ['P4', 'PO4', 'P6', 'PO8']

	# #chi = mne.pick_channels(ch_names=group_ave_probe_TFR_EDS.ch_names, include=fc, ordered=True)
	# chi = mne.pick_channels(ch_names=group_ave_probe_TFR_EDS.ch_names, include=fc, ordered=True)
	# freqs=np.arange(1,40.,1.)


	# D1 = np.zeros((len(probe_ave_TFR_IDS.keys()),probe_ave_TFR_IDS['128'].data.shape[2], probe_ave_TFR_IDS['128'].data.shape[1]))
	# D2 = np.zeros((len(probe_ave_TFR_EDS.keys()),probe_ave_TFR_EDS['128'].data.shape[2], probe_ave_TFR_EDS['128'].data.shape[1]))

	# for i, sub in enumerate(probe_ave_TFR_EDS.keys()):
	# 	D1[i,:,:] = np.mean(probe_ave_TFR_EDS[sub].data[chi,:,:], axis=0).transpose(1,0)
	# 	D2[i,:,:] = np.mean(probe_ave_TFR_IDS[sub].data[chi,:,:], axis=0).transpose(1,0)   
	# 	# this will be in sub by time by freq

	# #crop
	# D1 = D1[:,125:1000,:]
	# D2 = D2[:,125:1000,:]
	# D = D1-D2 # EDS - IDS
	# timevec = probe_ave_TFR_EDS['128'].times[125:1000]
	# t_obs, clusters, cluster_pv, h0 = mne.stats.permutation_cluster_1samp_test(D, out_type = 'mask', n_permutations=1000, n_jobs = 16, t_power = 1) 
	# print(cluster_pv)


	# plt.figure()

	# # Create new stats image with only significant clusters
	# T_obs_plot = np.nan * np.ones_like(t_obs)
	# for c, p_val in zip(clusters, cluster_pv):
	#     if p_val <= 0.01:
	#         T_obs_plot[c] = t_obs[c]

	# #vmax = 10#np.max(np.abs(t_obs))
	# #vmin = -10#-vmax
	# plt.imshow(t_obs.T, cmap=plt.cm.gray, extent=[timevec[0], timevec[-1], freqs[0], freqs[-1]],
	#            aspect='auto', vmin=-10, vmax=10, origin='lower')
	# plt.imshow(T_obs_plot.T, cmap=plt.cm.RdBu_r, extent=[timevec[0], timevec[-1], freqs[0], freqs[-1]],
	#            aspect='auto', vmin=-20, vmax=20, origin='lower')
	# #
	# # plt.imshow(t_obs.T, cmap=plt.cm.RdBu_r, extent=[timevec[0], timevec[-1], freqs[0], freqs[-1]],
	# #            aspect='auto', vmin=-10, vmax=10, origin='lower')
	# #plt.colorbar()
	# plt.xlabel('Time (ms)')
	# plt.ylabel('Frequency (Hz)')



	# group_ave_probe_TFR_EDS.plot(vmin=-5, vmax=5, tmin=-0.65, tmax=1.5, picks = fc)
	# group_ave_probe_TFR_IDS.plot(vmin=-5, vmax=5, tmin=-0.65, tmax=1.5, picks = fc)
	# group_ave_probe_TFR_Stay.plot(vmin=-5, vmax=5, tmin=-0.65, tmax=1.5, picks = fc)

	# group_ave_probe_TFR_EDS.plot(vmin=-5, vmax=5, tmin=-0.65, tmax=1.5, picks = pc)
	# group_ave_probe_TFR_IDS.plot(vmin=-5, vmax=5, tmin=-0.65, tmax=1.5, picks = pc)
	# group_ave_probe_TFR_Stay.plot(vmin=-5, vmax=5, tmin=-0.65, tmax=1.5, picks = pc)


	# # vector of theta subjects
	# theta = np.mean(np.mean(D1[:,180:440,:],axis=1)[:,3:5], axis=1)
	# alpha = np.mean(np.mean(D1[:,460:700,:],axis=1)[:,9:15], axis=1)

	# theta_diff = np.mean(np.mean(D1[:,180:440,:],axis=1)[:,3:5], axis=1) - np.mean(np.mean(D2[:,180:440,:],axis=1)[:,3:5], axis=1)
	# ########################################################################
	# ### run trial by trial ITC, without any baseline, then save.
	# ########################################################################
	# cue_ave_ITC = {}

	# for sub in all_subs_cue.keys():
	# 	cue_ave_ITC[sub] = {}
	# 	for condition in ['EDS_trig', 'IDS_trig', 'Stay_trig']:
	# 		_, cue_ave_ITC[sub][condition] = tfr_morlet(all_subs_cue[sub][condition][all_subs_cue[sub][condition].metadata['trial_Corr']==1], freqs=freqs, average=True,n_cycles=n_cycles, use_fft=True, return_itc=True, decim=1, n_jobs=6)

	# 	#save_object(tfi, OUT+sub+'_cueITC')  
	# cue_ave_ITC_EDS = {}
	# cue_ave_ITC_IDS = {}
	# cue_ave_ITC_Stay = {}
	# cue_ave_ITC_EDS_v_IDS = {}
	# cue_ave_ITC_IDS_v_Stay = {}
	# for sub in all_subs_cue.keys():
	# 	cue_ave_ITC_EDS_v_IDS[sub] = cue_ave_ITC[sub]['EDS_trig'] - cue_ave_ITC[sub]['IDS_trig'] 
	# 	cue_ave_ITC_IDS_v_Stay[sub] = cue_ave_ITC[sub]['IDS_trig'] - cue_ave_ITC[sub]['Stay_trig'] 
	# 	cue_ave_ITC_EDS[sub] = cue_ave_ITC[sub]['EDS_trig']
	# 	cue_ave_ITC_IDS[sub] = cue_ave_ITC[sub]['IDS_trig']
	# 	cue_ave_ITC_Stay[sub] = cue_ave_ITC[sub]['Stay_trig']

	# group_ave_cue_ITC_IDS_v_Stay = mne.grand_average(list(cue_ave_ITC_IDS_v_Stay.values()))
	# group_ave_cue_ITC_EDS_v_IDS = mne.grand_average(list(cue_ave_ITC_EDS_v_IDS.values()))
	# group_ave_cue_ITC_EDS_v_IDS.plot_topo()
	# group_ave_cue_ITC_IDS_v_Stay.plot_topo()

	# group_ave_cue_ITC_EDS = mne.grand_average(list(cue_ave_ITC_EDS.values()))

	# group_ave_cue_ITC_IDS = mne.grand_average(list(cue_ave_ITC_IDS.values()))


	# #for probe
	# probe_ave_ITC = {}

	# for sub in all_subs_probe.keys():
	# 	probe_ave_ITC[sub] = {}
	# 	for condition in ['EDS', 'IDS', 'stay']:
	# 		_, probe_ave_ITC[sub][condition] = tfr_morlet(all_subs_probe[sub][condition][all_subs_probe[sub][condition].metadata['trial_Corr']==1], freqs=freqs, average=True,n_cycles=n_cycles, use_fft=True, return_itc=True, decim=1, n_jobs=6)

	# 	#save_object(tfi, OUT+sub+'_cueITC')  
	# probe_ave_ITC_EDS = {}
	# probe_ave_ITC_IDS = {}
	# probe_ave_ITC_Stay = {}
	# probe_ave_ITC_EDS_v_IDS = {}
	# probe_ave_ITC_IDS_v_Stay = {}
	# for sub in all_subs_probe.keys():
	# 	probe_ave_ITC_EDS_v_IDS[sub] = probe_ave_ITC[sub]['EDS'] - probe_ave_ITC[sub]['IDS'] 
	# 	probe_ave_ITC_IDS_v_Stay[sub] = probe_ave_ITC[sub]['IDS'] - probe_ave_ITC[sub]['stay'] 
	# 	probe_ave_ITC_EDS[sub] = probe_ave_ITC[sub]['EDS']
	# 	probe_ave_ITC_IDS[sub] = probe_ave_ITC[sub]['IDS']
	# 	probe_ave_ITC_Stay[sub] = probe_ave_ITC[sub]['stay']

	# group_ave_probe_ITC_IDS_v_Stay = mne.grand_average(list(probe_ave_ITC_IDS_v_Stay.values()))
	# group_ave_probe_ITC_EDS_v_IDS = mne.grand_average(list(probe_ave_ITC_EDS_v_IDS.values()))
	# group_ave_probe_ITC_EDS_v_IDS.plot_topo()
	# group_ave_probe_ITC_IDS_v_Stay.plot_topo()

	# group_ave_probe_ITC_EDS = mne.grand_average(list(probe_ave_ITC_EDS.values()))
	# group_ave_probe_ITC_IDS = mne.grand_average(list(probe_ave_ITC_IDS.values()))



	# ########################################################################
	# ### Sensor level evoke response for cue
	# ########################################################################
	    
	# cue_ave_Evoke = {}

	# for sub in all_subs_cue.keys():
	# 	cue_ave_Evoke[sub] = {}
	# 	for condition in ['EDS_trig', 'IDS_trig', 'Stay_trig']:
	# 		cue_ave_Evoke[sub][condition] = all_subs_cue[sub][condition].average()


	# cue_ave_Evoke_EDS = {}
	# cue_ave_Evoke_IDS = {}
	# cue_ave_Evoke_Stay = {}
	# #cue_ave_Evoke_EDS_v_IDS = {}
	# #cue_ave_Evoke_IDS_v_Stay = {}
	# for sub in all_subs_cue.keys():
	# 	#cue_ave_Evoke_EDS_v_IDS[sub] = cue_ave_Evoke[sub]['EDS_trig'] - cue_ave_Evoke[sub]['IDS_trig'] 
	# 	#cue_ave_Evoke_IDS_v_Stay[sub] = cue_ave_Evoke[sub]['IDS_trig'] - cue_ave_Evoke[sub]['Stay_trig'] 
	# 	cue_ave_Evoke_EDS[sub] = cue_ave_Evoke[sub]['EDS_trig']
	# 	cue_ave_Evoke_IDS[sub] = cue_ave_Evoke[sub]['IDS_trig']
	# 	cue_ave_Evoke_Stay[sub] = cue_ave_Evoke[sub]['Stay_trig']

	# group_ave_cue_Evoke_EDS = mne.grand_average(list(cue_ave_Evoke_EDS.values()))
	# group_ave_cue_Evoke_IDS = mne.grand_average(list(cue_ave_Evoke_IDS.values()))
	# group_ave_cue_Evoke_Stay = mne.grand_average(list(cue_ave_Evoke_Stay.values()))

	# # Code to try out spatiotemproal clustering on evoke data
	# con = mne.channels.find_ch_connectivity(group_ave_cue_Evoke_EDS.info, "eeg")

	# #input into clustering needs to be in sub x time by channel.

	# D1 = np.zeros((len(cue_ave_Evoke_EDS.keys()),cue_ave_Evoke_EDS['128'].data.shape[1], cue_ave_Evoke_EDS['128'].data.shape[0]))
	# D2 = np.zeros((len(cue_ave_Evoke_EDS.keys()),cue_ave_Evoke_EDS['128'].data.shape[1], cue_ave_Evoke_EDS['128'].data.shape[0]))

	# for i, sub in enumerate(cue_ave_Evoke_EDS.keys()):
	# 	D1[i,:,:] = cue_ave_Evoke_EDS[sub].data.transpose(1,0)
	# 	D2[i,:,:] = cue_ave_Evoke_IDS[sub].data.transpose(1,0)   

	# #tfce = dict(start=.2, step=.2) #don't know what this is
	# t_obs, clusters, cluster_pv, h0 = mne.stats.spatio_temporal_cluster_test([D1,D2], .05, n_permutations=1000, n_jobs = 8) 

	# evoked = mne.combine_evoked([group_ave_cue_Evoke_EDS, - group_ave_cue_Evoke_IDS], weights='equal')
	# time_unit = dict(time_unit="s")
	# evoked.plot_joint(title="", ts_args=time_unit, topomap_args=time_unit)



	# ########################################################################
	# ### Sensor level evoke response for probe
	# ########################################################################
	    
	# probe_ave_Evoke = {}

	# for sub in all_subs_probe.keys():
	# 	probe_ave_Evoke[sub] = {}
	# 	for condition in ['EDS', 'IDS', 'stay']:
	# 		probe_ave_Evoke[sub][condition] = all_subs_probe[sub][condition].average()


	# probe_ave_Evoke_EDS = {}
	# probe_ave_Evoke_IDS = {}
	# probe_ave_Evoke_Stay = {}
	# #probe_ave_Evoke_EDS_v_IDS = {}
	# #probe_ave_Evoke_IDS_v_Stay = {}
	# for sub in all_subs_probe.keys():
	# 	#probe_ave_Evoke_EDS_v_IDS[sub] = probe_ave_Evoke[sub]['EDS_trig'] - probe_ave_Evoke[sub]['IDS_trig'] 
	# 	#probe_ave_Evoke_IDS_v_Stay[sub] = probe_ave_Evoke[sub]['IDS_trig'] - probe_ave_Evoke[sub]['Stay_trig'] 
	# 	probe_ave_Evoke_EDS[sub] = probe_ave_Evoke[sub]['EDS']
	# 	probe_ave_Evoke_IDS[sub] = probe_ave_Evoke[sub]['IDS']
	# 	probe_ave_Evoke_Stay[sub] = probe_ave_Evoke[sub]['stay']

	# group_ave_probe_Evoke_EDS = mne.grand_average(list(probe_ave_Evoke_EDS.values()))
	# group_ave_probe_Evoke_IDS = mne.grand_average(list(probe_ave_Evoke_IDS.values()))
	# group_ave_probe_Evoke_Stay = mne.grand_average(list(probe_ave_Evoke_Stay.values()))

	# # Code to try out spatiotemproal clustering on evoke data
	# con = mne.channels.find_ch_connectivity(probe_ave_Evoke[sub][condition].info, "eeg")

	# #input into clustering needs to be in sub x time by channel.

	# D1 = np.zeros((len(probe_ave_Evoke_EDS.keys()),probe_ave_Evoke_EDS['128'].data.shape[1], probe_ave_Evoke_EDS['128'].data.shape[0]))
	# D2 = np.zeros((len(probe_ave_Evoke_EDS.keys()),probe_ave_Evoke_EDS['128'].data.shape[1], probe_ave_Evoke_EDS['128'].data.shape[0]))
	# D3 = np.zeros((len(probe_ave_Evoke_EDS.keys()),probe_ave_Evoke_EDS['128'].data.shape[1], probe_ave_Evoke_EDS['128'].data.shape[0]))

	# for i, sub in enumerate(probe_ave_Evoke_EDS.keys()):
	# 	D1[i,:,:] = probe_ave_Evoke_Stay[sub].data.transpose(1,0)
	# 	D2[i,:,:] = probe_ave_Evoke_IDS[sub].data.transpose(1,0)   
	# 	D3[i,:,:] = probe_ave_Evoke_EDS[sub].data.transpose(1,0)   

	# D1 = D1[:,25:870,:]
	# D2 = D2[:,25:870,:]
	# D3 = D3[:,25:870,:]
	# time = probe_ave_Evoke_EDS[sub].times[25:870]
	# DD = np.dstack((np.mean(D1,axis=2), np.mean(D2, axis=2), np.mean(D3, axis=2)))

	# D = D1-D2
	# #tfce = dict(start=.2, step=.2) #don't know what this is
	# t_obs, clusters, cluster_pv, h0 = mne.stats.spatio_temporal_cluster_1samp_test(D, n_permutations=1000, n_jobs = 16) 

	# mne.concatenate_epochs(list(probe_ave_Evoke_EDS.values()))


	# # evoked = mne.combine_evoked([group_ave_probe_Evoke_EDS, group_ave_probe_Evoke_IDS], weights='equal')
	# # time_unit = dict(time_unit="s")
	# # evoked.plot_joint(title="", ts_args=time_unit, topomap_args=time_unit)






