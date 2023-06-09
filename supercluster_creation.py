##################################################################################
### transform PCA into meanbrain space so we can symmetrize across the midline ###
##################################################################################

### load eigenvectors ###
main_dir = "/oak/stanford/groups/trc/data/Brezovec/2P_Imaging/20210130_superv_depth_correction/"
file = os.path.join(main_dir,'20210214_eigen_vectors_ztrim.npy')
vectors = np.load(file).real
print(f'vectors are {vectors.shape} voxel by PC')

### load PCA labels ###
labels_file = '/oak/stanford/groups/trc/data/Brezovec/2P_Imaging/20210130_superv_depth_correction/labels.pickle'
with open(labels_file, 'rb') as handle:
    cluster_model_labels = pickle.load(handle)

### convert to image ###
running_sum = 0
new = []
for z in range(9,49-9):
    num_clusters = len(np.unique(cluster_model_labels[z]))
    new.append(vectors[running_sum:num_clusters+running_sum,:])
    running_sum += num_clusters

maps = []
for pc in range(100):
    all_ = []
    for z in range(9,49-9):
        colored_by_betas = np.zeros((256*128))
        for cluster_num in range(len(np.unique(cluster_model_labels[z]))):
            cluster_indicies = np.where(cluster_model_labels[z][:]==cluster_num)[0]
            colored_by_betas[cluster_indicies] = new[z-9][cluster_num,pc]
        colored_by_betas = colored_by_betas.reshape(256,128)
        all_.append(colored_by_betas)
    all_ = np.asarray(all_)
    maps.append(all_)
maps = np.asarray(maps)

maps = np.moveaxis(maps,1,-1)
maps = np.moveaxis(maps,0,-1)

pad = np.zeros((256,128,9,100))
out = np.concatenate((pad,maps,pad),axis=2)

### Load Luke Mean ###
luke_path = "/oak/stanford/groups/trc/data/Brezovec/2P_Imaging/anat_templates/20210310_luke_exp_thresh.nii"
res_luke_mean = (0.65,0.65,1)
luke_mean = np.asarray(nib.load(luke_path).get_data().squeeze(), dtype='float32')
luke_mean = luke_mean[:,:,::-1] #flipz
luke_mean = ants.from_numpy(luke_mean)
luke_mean.set_spacing(res_luke_mean)
luke_mean_lowres =  ants.resample_image(luke_mean,(256,128,49),use_voxels=True)

### Load JFRC2018 ###
fixed_path = "/oak/stanford/groups/trc/data/Brezovec/2P_Imaging/anat_templates/JRC2018_FEMALE_38um_iso_16bit.nii"
res_JRC2018 = (0.38, 0.38, 0.38)
fixed = np.asarray(nib.load(fixed_path).get_data().squeeze(), dtype='float32')
fixed = ants.from_numpy(fixed)
fixed.set_spacing(res_JRC2018)
fixed_lowres = ants.resample_image(fixed,(2,2,2),use_voxels=False)

### Load Atlas ###
atlas_path = "/oak/stanford/groups/trc/data/Brezovec/2P_Imaging/anat_templates/jfrc_2018_rois_improve_reorient_transformed.nii"
atlas = np.asarray(nib.load(atlas_path).get_data().squeeze(), dtype='float32')
atlas = ants.from_numpy(atlas)
atlas.set_spacing((.76,.76,.76))
atlas = ants.resample_image(atlas,(2,2,2),use_voxels=False)

moving = ants.from_numpy(out[:,:,::-1,:])
moving.set_spacing((2.6076, 2.6154, 5.3125,1))

### warp! ###
out = ants.registration(fixed_lowres, luke_mean_lowres, type_of_transform='Affine')
maps_voxel_res = ants.apply_transforms(fixed_lowres,
                                       moving,
                                       out['fwdtransforms'][0],
                                       interpolator='nearestNeighbor',
                                       imagetype=3)
pca_in_FDA = maps_voxel_res.numpy()

### symmetrize
pca_in_FDA_sym = (np.abs(pca_in_FDA) + np.abs(pca_in_FDA[::-1,...])) / 2
pca_in_FDA_hemi = pca_in_FDA_sym[:157,...]

#################################################
### now warp the supervoxel labels themselves ###
#################################################
all_ = []
running_sum = 0
for z in range(9,49-9):
    colored_by_betas = np.zeros((256*128))
    running_sum_temp = 0
    for cluster_num in np.unique(cluster_model_labels[z]):
        cluster_indicies = np.where(cluster_model_labels[z][:]==cluster_num)[0]
        colored_by_betas[cluster_indicies] = cluster_num+1 # need to not have any 0 clusters because using 0 for padding
        running_sum_temp += 1
    colored_by_betas = colored_by_betas.reshape(256,128)
    all_.append(colored_by_betas+running_sum)
    running_sum += running_sum_temp
all_ = np.asarray(all_)
pad = np.zeros((9,256,128))
supervoxels = np.concatenate((pad,all_,pad),axis=0)
supervoxels = np.moveaxis(supervoxels,0,2)

supervoxels = ants.from_numpy(supervoxels[:,:,::-1])
supervoxels.set_spacing((2.6076, 2.6154, 5.3125)) ### matching this to the slightly off luke mean

supervoxels_in_FDA = ants.apply_transforms(fixed_lowres,
                                       supervoxels,
                                       out['fwdtransforms'][0],
                                       interpolator='nearestNeighbor')
supervoxels_in_FDA_hemi = supervoxels_in_FDA[:157,...]

####################################################
### convert the warped PCA back into supervoxels ###
####################################################
pca_in_FDA_hemi_supervoxels = []
for super_id in tqdm.tqdm(np.unique(supervoxels_in_FDA_hemi)):
    ind = np.where(supervoxels_in_FDA_hemi==super_id)
    pca_in_FDA_hemi_supervoxels.append(np.mean(pca_in_FDA_hemi[ind[0],ind[1],ind[2],:],axis=0))
pca_in_FDA_hemi_supervoxels = np.asarray(pca_in_FDA_hemi_supervoxels)

###########################
### FINALLY can cluster ###
###########################
t0 = time.time()
print('clustering.........')
all_labels = []
for n_clusters in range(1,500): # trying anywhere between 1 and 500 clusters
    model = AgglomerativeClustering(distance_threshold=None, #first run with =0
                                    n_clusters=n_clusters, #and with n_clusters =None
                                    memory=main_dir,
                                    linkage='ward')
    model = model.fit(pca_in_FDA_hemi_supervoxels)
    all_labels.append(model.labels_)
all_labels = np.asarray(all_labels)

super_clusters = np.zeros((157, 146, 91, 499))
for i,super_id in enumerate(tqdm.tqdm(np.unique(supervoxels_in_FDA_hemi))):
    ind = np.where(supervoxels_in_FDA_hemi==super_id)
    super_clusters[ind[0],ind[1],ind[2],:] = all_labels[:,int(i)] + 1
#convert back from hemi
super_clusters_full = np.concatenate((super_clusters,super_clusters[::-1,...]),axis=0)

#save
save_file = "/oak/stanford/groups/trc/data/Brezovec/2P_Imaging/20221109_cluster_pca/superclusters_more"
np.save(save_file, super_clusters_full)