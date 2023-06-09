### associated with figure 2

### Load Supercluster Signals ###
cluster_dir = "/oak/stanford/groups/trc/data/Brezovec/2P_Imaging/20221109_cluster_pca/"
file = os.path.join(cluster_dir, "20221202_SC_signals.npy")
supercluster_signals = np.load(file)
supercluster_signals = supercluster_signals.T
supercluster_signals_fly = np.reshape(supercluster_signals,([9,3384,501]))

### for a given supercluster, i need to know the original median z-depth for each fly ###
dataset_path = "/oak/stanford/groups/trc/data/Brezovec/2P_Imaging/20190101_walking_dataset"
fly_names = ['fly_087', 'fly_089', 'fly_094', 'fly_097', 'fly_098', 'fly_099', 'fly_100', 'fly_101', 'fly_105']

z_corrections = []
for fly in tqdm.tqdm(fly_names):
    to_load = os.path.join(dataset_path, fly, 'warp', '20201220_warped_z_depth.nii')
    z_corrections.append(np.array(nib.load(to_load).get_data(), copy=True))
z_corrections = np.asarray(z_corrections)

superclusters_3d = np.load(os.path.join(cluster_dir, "20221130_pca_clsuters_in_luke_OG.npy"))
superclusters_3d = superclusters_3d[...,::-1] ### FLIP Z !!!!!!!!!!!!
superclusters_3d.shape

original_z_depth = []
for fly in tqdm.tqdm(range(9)):
    for cluster in range(501):
        ind = np.where(superclusters_3d==cluster)
        original_z_depth.append(np.median(z_corrections[fly,ind[0],ind[1],ind[2]]))
original_z_depth = np.asarray(original_z_depth)
original_z_depth = np.reshape(original_z_depth,(9,501))
original_z_depth = original_z_depth.astype('int')

### load behavior ###
fly_names = ['fly_087', 'fly_089', 'fly_094', 'fly_097', 'fly_098', 'fly_099', 'fly_100', 'fly_101', 'fly_105']
dataset_path = "/oak/stanford/groups/trc/data/Brezovec/2P_Imaging/20190101_walking_dataset"

fictrac = []
for fly in fly_names:
    fictrac_raw = brainsss.load_fictrac(os.path.join(dataset_path, fly, 'func_0', 'fictrac'))
    # Smooth raw fictrac data
    behavior = 'dRotLabZ'
    fictrac_smoothed = scipy.signal.savgol_filter(np.asarray(fictrac_raw[behavior]),25,3)
    fps=50
    fictrac_smoothed = fictrac_smoothed * 180 / np.pi * fps # now in deg/sec
    fictrac.append(fictrac_smoothed)
fictrac_timestamps = np.arange(0,30*60*1000,20)
file = os.path.join(dataset_path, 'fly_087', 'func_0', 'imaging')
neural_timestamps = brainsss.load_timestamps(file)


### Identify turn bouts ###
turn_thresh = 200
peaks_all_fly = []
peak_heights_all_fly = []
for fly in range(9):
    peaks = {'L':[],'R':[]}
    heights = {'L':[],'R':[]}
    for turn,scalar in zip(['L', 'R'],[1,-1]):
        
        found_peaks = scipy.signal.find_peaks(fictrac[fly]*scalar, height=turn_thresh)
        pks = found_peaks[0]
        pk_height = found_peaks[1]['peak_heights']
        
        ### remove peaks that are too close to beginning or end of recording
        # will do 20sec window
        # here 20sec is 1000 tps
        ind = np.where(pks>88000)[0]
        pks = np.delete(pks,ind)
        pk_height = np.delete(pk_height,ind)
        
        ind = np.where(pks<2000)[0]
        pks = np.delete(pks,ind)
        pk_height = np.delete(pk_height,ind)
        
        peaks[turn] = pks
        heights[turn] = pk_height
    peaks_all_fly.append(peaks)
    peak_heights_all_fly.append(heights)

### get neural traces for each bout ###
neural_traces_L = {}
neural_traces_R = {}
neural_traces_L_std = {}
neural_traces_R_std = {}

for cluster in tqdm.tqdm(range(500)):
    
    neural_traces_L[cluster] = []
    neural_traces_R[cluster] = []
    neural_traces_L_std[cluster] = []
    neural_traces_R_std[cluster] = []

    for fly in range(len(peaks_all_fly)):
    
        z = original_z_depth[fly,cluster]
        neural_timestamps_cluster = neural_timestamps[:,z]
        
        for beh in ['L', 'R']:
            
            peaks_in_ms = fictrac_timestamps[peaks_all_fly[fly][beh]]
            
            for peak in peaks_in_ms:
                
                #this will give the index of the first neural data after the peak
                # 40 index will be the bin from 0 to 532ms after the peak
                middle = np.searchsorted(neural_timestamps_cluster,peak)
                if beh == 'L':
                    neural_traces_L[cluster].append(supercluster_signals_fly[fly,middle-60:middle+60,cluster])
                    neural_traces_L_std[cluster].append(supercluster_signals_fly_std[fly,middle-60:middle+60,cluster])
                elif beh == 'R':
                    neural_traces_R[cluster].append(supercluster_signals_fly[fly,middle-60:middle+60,cluster])
                    neural_traces_R_std[cluster].append(supercluster_signals_fly_std[fly,middle-60:middle+60,cluster])

    neural_traces_L[cluster] = np.asarray(neural_traces_L[cluster])
    neural_traces_R[cluster] = np.asarray(neural_traces_R[cluster])
    neural_traces_L_std[cluster] = np.asarray(neural_traces_L_std[cluster])
    neural_traces_R_std[cluster] = np.asarray(neural_traces_R_std[cluster])

### Get signifance for each time window
windows = [(0,31),(33,44),(44,59),(59,61)]
# these correspond to -30 to -15 seconds, -15 to -8, -8 to -0.4, -0.4 to 0.4

all_Ps = {}
all_means = {}
for k,window in enumerate(windows):
    start = window[0]
    stop = window[1]
    Ps = []
    Ts = []
    means = []
    for cluster in range(250):

    	# the +250 gets the matching cluster in the other hemisphere
        L = neural_traces_L[cluster][:,start:stop] - neural_traces_L[cluster+250][:,start:stop]
        R = neural_traces_R[cluster][:,start:stop] - neural_traces_R[cluster+250][:,start:stop]
        L = np.mean(L,axis=1)
        R = np.mean(R,axis=1)
        means.append(np.abs(np.mean(L)-np.mean(R)))

        ### calculate p value!
        t,p = scipy.stats.ttest_ind(L, R)
        
    all_Ps[k] = Ps
    all_means[k] = means