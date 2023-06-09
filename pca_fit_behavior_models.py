############################
### define key variables ###
############################

fly_names = ['fly_087', 'fly_089', 'fly_094', 'fly_097', 'fly_098', 'fly_099', 'fly_100', 'fly_101', 'fly_105']
alphas = [1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7] # for regularization
num_pcss = [1,10,30,100,250,500,1000,2000] # how many pcs to use as input
behaviors = ['Y_pos', 'Z_pos', 'Z_neg'] #forward, left, and right velocities
use_super_pcs = True # these pcs are fit on the superfly
folds = 5

fly_scores = {}
### loop over flies
for k,fly in enumerate(fly_names):
    fly_scores[fly] = {}
    print(fly)
    t0=time.time()
    ### loop over behaviors
    for behavior in behaviors:
        all_all_scores = []
        for num_pcs in num_pcss:
            Y = flies[fly].fictrac.fictrac[behavior][z]
            
            if use_super_pcs:
                start = k*3384
                end = (k+1)*3384
                X = temporal_super[start:end,:num_pcs]
            else:
                X = temporal[fly][:,:num_pcs] #time by pc

            all_scores = []
            ### loop over regularization alphas
            for alpha in alphas:
                scores = []
                for fold in range(folds):
                    model = Ridge(alpha=alpha).fit(X[train_indicies[fold],:],Y[train_indicies[fold]]) #Ridge is from sklearn
                    scores.append(model.score(X[test_indicies[fold],:],Y[test_indicies[fold]]))
                all_scores.append(np.mean(scores))
            all_all_scores.append(all_scores)
        print(time.time()-t0)

        max_scores_per_pcs = []
        max_scores_per_pcs_alpha = []
        for i,num_pcs in enumerate(num_pcss):
            max_scores_per_pcs.append(np.max(all_all_scores[i]))
            max_scores_per_pcs_alpha.append(np.argmax(all_all_scores[i]))

        fly_scores[fly][behavior] = max_scores_per_pcs
        fly_scores[fly][behavior + '_alpha'] = max_scores_per_pcs_alpha