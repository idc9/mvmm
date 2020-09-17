import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


from mvmm.single_view.gaussian_mixture import GaussianMixture
from mvmm.multi_view.BlockDiagMVMM import BlockDiagMVMM
from mvmm.multi_view.MVMM import MVMM
from mvmm.multi_view.TwoStage import TwoStage

from mvmm.multi_view.block_diag.toy_data import get_01_block_diag
from mvmm.multi_view.toy_data import sample_gmm, setup_grid_mean_view_params
from mvmm.multi_view.block_diag.graph.bipt_community import community_summary

# block diagonal matrix with 5 2x2 blocks
Pi_true = get_01_block_diag(block_shapes=[(2, 2)] * 5)
Pi_true /= Pi_true.sum()
n_view_components = Pi_true.shape

# set up cluster parameters
clust_param_config = {'n_features': [1, 1],
                      'cluster_std': .25,  # how noisey each cluster is
                      'random_state': 44}
view_params = setup_grid_mean_view_params(n_view_components,
                                          **clust_param_config)

# sample data
view_data, Y_true = sample_gmm(view_params, Pi_true, n_samples=500,
                               random_state=233)


# spcify view spcific models
base_view_models = [GaussianMixture(n_components=n_view_components[v])
                    for v in range(2)]

# initialize with a few EM of the basic MVMM
base_start = MVMM(base_view_models=base_view_models, max_n_steps=10)

# setup block diagonal model
base_final = BlockDiagMVMM(n_blocks=5, base_view_models=base_view_models)

mvmm = TwoStage(base_start=base_start, base_final=base_final,
                n_init=1, random_state=883)

mvmm.fit(view_data)


##################
# Visualizations #
##################

# Observed data and True Pi matrix
plt.figure(figsize=(17, 8))

plt.subplot(1, 2, 1)
plt.scatter(view_data[0], view_data[1], color='black', s=10)
plt.xlabel('First view')
plt.ylabel('Second view')
plt.title('Observed data')

plt.subplot(1, 2, 2)
sns.heatmap(Pi_true.T,  # transpose so the first view is on the rows
            annot=True, cmap='Blues', vmin=0, linewidths=.2, cbar=False,
            mask=Pi_true.T == 0)
plt.xlabel("First view clusters")
plt.ylabel("Second view clusters")
plt.title("True Pi matrix")
plt.xticks(np.arange(n_view_components[0]) + .5,
           np.arange(1, n_view_components[0] + 1))
plt.yticks(np.arange(n_view_components[1]) + .5,
           np.arange(1, n_view_components[1] + 1))
plt.savefig('obs_data_and_true_pi.png', dpi=200, bbox_inches='tight')


D_est = mvmm.final_.bd_weights_
bd_summary, D_est_bd_perm = community_summary(D_est,
                                              zero_thresh=mvmm.final_.zero_thresh)
# Estimated D matrix
plt.figure(figsize=(8, 8))
sns.heatmap(D_est_bd_perm.T,  # transpose so the first view is on the rows
            annot=True, cmap='Blues', vmin=0, linewidths=.2, cbar=False,
            mask=D_est_bd_perm.T == 0)
plt.xlabel("First view clusters")
plt.ylabel("Second view clusters")
plt.title("Estimated D matrix (permuted to reveal block diagonal structure)")
plt.xticks(np.arange(n_view_components[0]) + .5,
           np.arange(1, n_view_components[0] + 1))
plt.yticks(np.arange(n_view_components[1]) + .5,
           np.arange(1, n_view_components[1] + 1))
plt.savefig('D_est.png', dpi=200, bbox_inches='tight')
