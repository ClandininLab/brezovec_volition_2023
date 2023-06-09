#################################
### Load Supercluster Signals ###
#################################
cluster_dir = "/oak/stanford/groups/trc/data/Brezovec/2P_Imaging/20221109_cluster_pca/"
file = os.path.join(cluster_dir, "20221202_SC_signals.npy")
supercluster_signals = np.load(file)
supercluster_signals = supercluster_signals.T
supercluster_signals_fly = np.reshape(supercluster_signals,([9,3384,501]))

#########################################################################################
### for a given supercluster, i need to know the original median z-depth for each fly ###
#########################################################################################
dataset_path = "/oak/stanford/groups/trc/data/Brezovec/2P_Imaging/20190101_walking_dataset"
fly_names = ['fly_087', 'fly_089', 'fly_094', 'fly_097', 'fly_098', 'fly_099', 'fly_100', 'fly_101', 'fly_105']

z_corrections = []
for fly in tqdm.tqdm(fly_names):
    to_load = os.path.join(dataset_path, fly, 'warp', '20201220_warped_z_depth.nii')
    z_corrections.append(np.array(nib.load(to_load).get_data(), copy=True))
z_corrections = np.asarray(z_corrections)

superclusters_3d = np.load(os.path.join(cluster_dir, "20221130_pca_clsuters_in_luke_OG.npy"))
superclusters_3d = superclusters_3d[...,::-1] ### flip z
superclusters_3d.shape

original_z_depth = []
for fly in tqdm.tqdm(range(9)):
    for cluster in range(501):
        ind = np.where(superclusters_3d==cluster)
        original_z_depth.append(np.median(z_corrections[fly,ind[0],ind[1],ind[2]]))
original_z_depth = np.asarray(original_z_depth)
original_z_depth = np.reshape(original_z_depth,(9,501))
original_z_depth = original_z_depth.astype('int')

#####################
### load behavior ###
#####################
fly_names = ['fly_087', 'fly_089', 'fly_094', 'fly_097', 'fly_098', 'fly_099', 'fly_100', 'fly_101', 'fly_105']
dataset_path = "/oak/stanford/groups/trc/data/Brezovec/2P_Imaging/20190101_walking_dataset"

fictrac = []
fictrac_fwd = []
fictrac_any = []
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

###########################
### Identify turn bouts ###
###########################
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

#####################################
### get neural trace of each bout ###
#####################################
neural_traces_L = {}
neural_traces_R = {}

for cluster in tqdm.tqdm(range(500)):
    
    neural_traces_L[cluster] = []
    neural_traces_R[cluster] = []
    

    for fly in range(9):
    
        z = original_z_depth[fly,cluster]
        neural_timestamps_cluster = neural_timestamps[:,z]
        
        for beh in ['L', 'R']:
            
            peaks_in_ms = fictrac_timestamps[peaks_all_fly[fly][beh]]
            
            for peak in peaks_in_ms:
                
                #this will give the index of the first neural data after the peak
                # 40 index will be the bin from 0 to 532ms after the peak
                middle = np.searchsorted(neural_timestamps_cluster,peak)
                if beh == 'L':
                    neural_traces_L[cluster].append(supercluster_signals_fly[fly,middle-60:middle+60,cluster]) #60
                elif beh == 'R':
                    neural_traces_R[cluster].append(supercluster_signals_fly[fly,middle-60:middle+60,cluster])
                    
    neural_traces_L[cluster] = np.asarray(neural_traces_L[cluster])
    neural_traces_R[cluster] = np.asarray(neural_traces_R[cluster])


### define the subset of superclusters to average over
clusters = [57,77,185] ### these are the supercluster that comprise the pre-motor center

# notation here is "left turn left brain, etc."
lt_lb = np.zeros(neural_traces_L[0].shape)
for cluster in clusters:
    lt_lb+=neural_traces_L[cluster]
lt_lb /= len(clusters)

rt_lb = np.zeros(neural_traces_R[0].shape)
for cluster in clusters:
    rt_lb+=neural_traces_R[cluster]
rt_lb /= len(clusters)

### these are the matching superclusters in the other hemisphere
clusters = [i+250 for i in clusters]

lt_rb = np.zeros(neural_traces_L[0].shape)
for cluster in clusters:
    lt_rb+=neural_traces_L[cluster]
lt_rb /= len(clusters)

rt_rb = np.zeros(neural_traces_R[0].shape)
for cluster in clusters:
    rt_rb+=neural_traces_R[cluster]
rt_rb /= len(clusters)

neural_L = lt_lb-lt_rb
neural_R = rt_lb-rt_rb
neural_low_res = {'L': neural_L, 'R': neural_R}
neural_low_res['L'].shape, neural_low_res['R'].shape

######################################
### reorganize the behavior traces ###
######################################
window = 1500 # fictrac is in units of 50ms, so this window is 30sec
beh_traces_L = []
beh_traces_R = []

for fly in final_flies:
    
    beh_traces_L_fly = []
    beh_traces_R_fly = []
    
    for beh in ['L', 'R']:
        peaks_in_ms = peaks_all_fly[fly][beh]
        for peak in peaks_in_ms:
            bout = fictrac[fly][peak-window:peak+window]
            if beh == 'L':
                beh_traces_L_fly.append(bout)
            elif beh == 'R':
                beh_traces_R_fly.append(bout)
                
    beh_traces_L.append(np.asarray(beh_traces_L_fly))
    beh_traces_R.append(np.asarray(beh_traces_R_fly))
    
beh_traces_L_stacked = np.empty((0,window*2))
beh_traces_R_stacked = np.empty((0,window*2))
for fly in range(len(final_flies)):
    beh_traces_L_stacked = np.concatenate((beh_traces_L_stacked, beh_traces_L[fly]))
    beh_traces_R_stacked = np.concatenate((beh_traces_R_stacked, beh_traces_R[fly]))

# high res behavior
beh_high_res = {'L': beh_traces_L_stacked,
                'R': beh_traces_R_stacked}

# sort bouts based on behavior amount
start = 800 #this is -15 sec
stop = 1450 #this is -1 sec
bouts_filtered = {'L': [], 'R': []}
for beh in ['L', 'R']:
    pre_beh_amount = np.sum(np.abs(beh_high_res[beh])[:,start:stop],axis=1)
    ind = np.argsort(pre_beh_amount)
    bouts_filtered[beh] = ind

############################
### Bootstrap Prediction ###
############################

thresh = 0
accuracy = {}
for num_bout in [27, 'all']: # test for bouts with least activity
    for t in ['0', 'early']: # test for early neural activity and activity at time of turn

    	# define appropriate time windows
        if t == '0':
            start = 59
            stop = 61
        elif t == 'early':
            start = 32
            stop = 57

        acc = []
        for iteration in range(10000): #bootstrap interations

        	# get bootstramp sample
            if num_bout == 'all':
                n_bout = 1000
                bt_L = choices(range(1362), k=n_bout)
                bt_R = choices(range(2424), k=n_bout)
            
            if num_bout == 27:
                n_bout = num_bout
                bt_L = choices(range(n), k=n_bout)
                bt_R = choices(range(n), k=n_bout)

            # get each neural trace
            r_bouts = neural_low_res['R'][bouts_filtered['R'][bt_R]]
            l_bouts = neural_low_res['L'][bouts_filtered['L'][bt_L]]
            
            # get single neural value averaged across the time window
            r = r_bouts[:,start:stop]
            l = l_bouts[:,start:stop]
            R = np.mean(r,axis=1)
            L = np.mean(l,axis=1)

            ### whether a correct prediction is above or below zero varies depending on the region and the time window
            ### direction correct for IPS
#             if t == '0':
#                 acc.append(np.mean([np.sum(L>thresh)/n_bout, np.sum(R<thresh)/n_bout]))
#             elif t == 'early':
#                 acc.append(np.mean([np.sum(L<thresh)/n_bout, np.sum(R>thresh)/n_bout]))
            
            ### direction correct for LH
            if t == '0':
                acc.append(np.mean([np.sum(L>thresh)/n_bout, np.sum(R<thresh)/n_bout]))
            elif t == 'early':
                acc.append(np.mean([np.sum(L>thresh)/n_bout, np.sum(R<thresh)/n_bout]))
        accuracy[F'{num_bout}{t}'] = acc
