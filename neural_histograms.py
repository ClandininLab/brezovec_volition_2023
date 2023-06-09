### associated with figure 1

### Load Supercluster Signals ###
cluster_dir = "/oak/stanford/groups/trc/data/Brezovec/2P_Imaging/20221109_cluster_pca/"
file = os.path.join(cluster_dir, "20221202_SC_signals.npy")
supercluster_signals = np.load(file)
supercluster_signals.shape

### Load behavior
fly_names = ['fly_087', 'fly_089', 'fly_094', 'fly_097', 'fly_098', 'fly_099', 'fly_100', 'fly_101', 'fly_105']
dataset_path = "/oak/stanford/groups/trc/data/Brezovec/2P_Imaging/20190101_walking_dataset"
behaviors = ['dRotLabY', 'dRotLabZ', 'dRotLabX']; shorts = ['Y', 'Z', 'X']
fictrac = {'Y': [], 'Z': [], 'X': []}
for fly in fly_names:
    fly_dir = os.path.join(dataset_path, fly)
    fictrac_raw = bbb.load_fictrac(os.path.join(fly_dir, 'func_0', 'fictrac'))
    for behavior, short in zip(behaviors, shorts):
        fictrac[short].append(fictrac_raw[behavior])
for short in shorts:    
    fictrac[short] = np.asarray(fictrac[short])

### Match behavior and neural times
# Create camera timepoints
fps=50
camera_rate = 1/fps * 1000 # camera frame rate in ms
expt_len = 1000*30*60
x_original = np.arange(0,expt_len,camera_rate)

# new timepoints
width = 30000 
step = 20 #in ms
time_shifts = list(range(-width,width,step))
xs = np.asarray(time_shifts)/1000

# neural timepoints
file = os.path.join(dataset_path, 'fly_087', 'func_0', 'imaging')
neural_timestamps = brainsss.load_timestamps(file)

# pick supercluster for which to calculate neural-behavior histogram
cluster = 77
x_data = supercluster_signals[cluster,:] / np.std(supercluster_signals[cluster,:])
y_data = supercluster_signals[cluster+250,:] / np.std(supercluster_signals[cluster+250,:])

# convert fictrac to real-world units
#behavior = fictrac['Z']* 180 / np.pi * fps ### for rotational velocity
sphere_radius = 4.5e-3 # in m
behavior = fictrac['Y'] * sphere_radius * fps * 1000 ###for forward velocity

########################
### Create histogram ###
########################

# Define bins
start=-4
stop=4
num_bins=30
min_num_samples=10
bins = np.linspace(start,stop,num_bins)

# Assign neural values to bin numbers
idx_x, idx_y = np.digitize(x_data,bins), np.digitize(y_data,bins)

binned_signal = []
sample_count = []
for i in range(num_bins):
    mask_x = (idx_x == i)
    for j in range(num_bins):
        mask_y = (idx_y == j)
        mask = mask_x & mask_y
        
        ### for turns
#       thresh = np.percentile(np.abs(behavior[mask]),90)
#       indicies = np.abs(behavior[mask])>thresh
#       binned_signal.append(np.mean(behavior[mask][indicies]))
     
        ### for fwd
        binned_signal.append(np.percentile(behavior[mask],90))

        sample_count.append(np.count_nonzero(~np.isnan(behavior[mask])))

binned_signal = np.reshape(binned_signal,(num_bins, num_bins))
sample_count = np.reshape(sample_count,(num_bins, num_bins))