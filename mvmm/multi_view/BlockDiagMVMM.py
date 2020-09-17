import numpy as np
from warnings import warn, simplefilter, catch_warnings
from copy import deepcopy
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.fixes import logsumexp
from cvxpy import SolverError
from datetime import datetime
from textwrap import dedent

from mvmm.base import _em_docs
from mvmm.multi_view.MVMM import MVMM
from mvmm.utils import get_seeds
from mvmm.multi_view.block_diag.utils import asc_sort
from mvmm.linalg_utils import eigh_wrapper
from mvmm.opt_utils import solve_problem_cp_backups, solve_problem_cp
from mvmm.multi_view.block_diag.graph.linalg import geigh_sym_laplacian_bp,\
    get_unnorm_laplacian_bp
from mvmm.multi_view.block_diag.graph.bipt_community import community_summary,\
    get_nonzero_block_mask

from mvmm.multi_view.block_diag.sub_prob_fixed_zeros import \
    get_cp_prob_fixed_zeros
from mvmm.multi_view.block_diag.sub_prob_cp_sym_lap import \
    get_cp_problem_sym_lap
from mvmm.multi_view.block_diag.sub_prob_cp_un_lap import \
    get_cp_problem_un_lap
from mvmm.multi_view.block_diag.utils import get_lin_coef
from mvmm.multi_view.block_diag.graph.bipt_spect_partitioning import \
    run_bipt_spect_partitioning

# TODO-FEAT: n_blocks = 1


class BlockDiagMVMM(MVMM):

    def __init__(self,
                 base_view_models=None,
                 n_blocks=None,
                 max_n_steps=200,
                 abs_tol=1e-9,
                 rel_tol=None,
                 n_init=1,
                 init_params_method='init',
                 init_params_value=None,
                 init_weights_method='uniform',
                 init_weights_value=None,
                 random_state=None,
                 verbosity=0,
                 history_tracking=0,
                 eval_weights=None,
                 rel_eta=None,
                 eval_pen_base='guess_from_init',
                 eval_pen_incr=2,
                 eval_pen_decr=0.75,
                 n_pen_tries=30,
                 init_pen_c=1e-2,
                 init_pen_use_bipt_sp=False,
                 init_pen_K='default',
                 lap='sym',
                 rel_epsilon=1e-2,
                 exclude_vdv_constr=False,
                 cp_kws={},  # {'abstol': 1e-3, 'reltol': 1e-3, 'feastol': 1e-9}
                 zero_thresh=1e-6,
                 fine_tune_n_steps=200):

        super().__init__(base_view_models=base_view_models,
                         max_n_steps=max_n_steps,
                         abs_tol=abs_tol,
                         rel_tol=rel_tol,
                         n_init=n_init,
                         init_params_method=init_params_method,
                         init_params_value=init_params_value,
                         init_weights_method=init_weights_method,
                         init_weights_value=init_weights_value,
                         random_state=random_state,
                         verbosity=verbosity,
                         history_tracking=history_tracking)

        self.n_blocks = n_blocks
        self.eval_weights = eval_weights
        self.eval_pen_base = eval_pen_base
        self.eval_pen_incr = eval_pen_incr
        self.eval_pen_decr = eval_pen_decr

        self.init_pen_c = init_pen_c
        self.init_pen_use_bipt_sp = init_pen_use_bipt_sp
        self.init_pen_K = init_pen_K

        self.lap = lap
        self.rel_eta = rel_eta
        self.rel_epsilon = rel_epsilon
        self.exclude_vdv_constr = exclude_vdv_constr
        self.cp_kws = cp_kws
        self.zero_thresh = zero_thresh
        self.n_pen_tries = n_pen_tries

        self.fine_tune_n_steps = fine_tune_n_steps
        self.__mode = 'lap_pen'

    def _check_fitting_parameters(self, X):

        if self.abs_tol is not None and self.abs_tol < 0.:
            raise ValueError("Invalid value for 'abs_tol': %.5f "
                             "Tolerance must be non-negative"
                             % self.abs_tol)

        if self.n_init < 1:
            raise ValueError("Invalid value for 'n_init': %d "
                             "Estimation requires at least one run"
                             % self.n_init)

        if self.max_n_steps < 0:
            raise ValueError("Invalid value for 'max_n_steps': %d "
                             ", must be positive."
                             % self.max_n_steps)

        if self.eval_weights is not None:
            # self.eval_weights = np.sort(self.eval_weights)

            if sum(self.eval_weights < 0) >= 1:
                raise ValueError('eval_weights must be positive.')

            if len(self.eval_weights) > min(self.n_view_components):
                raise ValueError('len(eval_weights) must not be larger than '
                                 'the smallest number of view components.')

            if self.n_blocks is not None and \
                    len(self.eval_weights) != self.n_blocks:
                raise ValueError("Invalid value for eval_weights: {}."
                                 "Must have length == n_blocks"
                                 .format(self.eval_weights))

        if self.fine_tune_n_steps is not None and self.fine_tune_n_steps <= 0:
            raise ValueError("Invalid value for fine_tune_n_steps: {}."
                             "Must be either None or positive"
                             .format(self.fine_tune_n_steps))

        if self.n_blocks is not None and self.n_blocks <= 0:
            raise ValueError("Invalid value for n_blocks: {}."
                             "Must be either None or a positive integer"
                             .format(self.n_blocks))

    def _get_parameters(self):
        view_params = []
        for v in range(self.n_views):
            view_params.append(self.view_models_[v]._get_parameters())

        return {'views': view_params,
                'bd_weights': self.bd_weights_}

    def _set_parameters(self, params):

        if 'views' in params.keys():
            for v in range(self.n_views):
                self.view_models_[v]._set_parameters(params['views'][v])

        # weights parameters
        if 'bd_weights' in params.keys():
            self.bd_weights_ = np.array(params['bd_weights']).\
                reshape(self.n_view_components)

            self.weights_ = (self.bd_weights_ + self.epsilon).reshape(-1)

        # make sure each view's weights_ is the marginal of the weights
        if self.weights_mat_ is not None:
            for v in range(self.n_views):
                ax_to_sum = tuple([a for a in range(self.n_views) if a != v])
                view_weights = np.sum(self.weights_mat_, axis=ax_to_sum)
                self.view_models_[v].weights_ = view_weights

        if 'eval_pen' in params.keys():
            self.eval_pen_ = params['eval_pen']

    def _post_initialization(self, init_params, X, random_state):

        if 'weights' in init_params.keys():
            w = deepcopy(init_params['weights'])

            # maybe properly normalized w
            if not np.allclose(w.sum(), self.epsilon_tilde):
                w *= self.epsilon_tilde / w.sum()

            init_params['bd_weights'] = w

            del init_params['weights']

        return init_params

    def get_eval_pen_guess(self, X, c=1.0, use_bipt_sp=True, K='default'):
        e_out = self._e_step(X)

        # log coefficient
        log_resp = e_out['log_resp']
        n_samples = log_resp.shape[0]
        resp = np.exp(log_resp)
        nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
        a = nk / n_samples  # normalize so sum(a) == 1
        log_coef = a.reshape(*self.n_view_components)

        # linear coefficient
        eig_var = e_out['eig_var']
        lin_coef = get_lin_coef(V=eig_var, shape=self.n_view_components,
                                weights=self.eval_weights)

        epsilon = self.epsilon

        entry_guess = log_coef / (epsilon * lin_coef)

        if use_bipt_sp:

            if K == 'default':
                if self.n_blocks is not None:
                    K = self.n_blocks
                else:
                    K = len(self.eval_weights)

            blocks_idx_mat = run_bipt_spect_partitioning(X=self.bd_weights_,
                                                         n_blocks=K)

            between_block_values = entry_guess[np.isnan(blocks_idx_mat)]
            # default = np.median(between_block_values)
            default = np.max(between_block_values)
        else:
            default = np.median(entry_guess.reshape(-1))

        return c * default

    def _initialize_eval_pen(self, X):
        if self.eval_pen_base == 'guess_from_init':

            default = \
                self.get_eval_pen_guess(X,
                                        c=self.init_pen_c,
                                        use_bipt_sp=self.init_pen_use_bipt_sp,
                                        K=self.init_pen_K)

            # if self.verbosity >= 1:
            print('default_lap_pen_base', default)
            self.eval_pen_ = default

        else:
            self.eval_pen_ = self.eval_pen_base

    @property
    def epsilon(self):
        assert 0 < self.rel_epsilon and self.rel_epsilon < 1
        return self.rel_epsilon / np.product(self.n_view_components)

    @property
    def epsilon_tilde(self):
        return 1 - np.product(self.n_view_components) * self.epsilon

    @property
    def eta(self):
        if self.rel_eta is not None:
            assert 0 < self.rel_eta and self.rel_eta < 1
            eta = self.rel_eta / sum(self.n_view_components)
            return self.epsilon_tilde * eta

        else:
            return None

    def update_eval_pen(self, increase=True):
        if increase:
            new_eval_pen = self.eval_pen_ * self.eval_pen_incr
        else:
            new_eval_pen = self.eval_pen_ * self.eval_pen_decr

        self._set_parameters({'eval_pen': new_eval_pen})

    def _best_em_loop(self, X):
        """
        Run the algorithm for multiple initalizations and pick the best solution.
        """

        init_seeds = get_seeds(n_seeds=self.n_init,
                               random_state=self.random_state)

        # lower bounds for each initialization
        init_loss_vals = []
        init_success = []

        for i in range(self.n_init):
            if self.verbosity >= 1:
                time = datetime.now().strftime("%H:%M:%S")
                print('Beginning initialization {} at {}'.format(i + 1, time))

            # initialize parameters if not warm starting
            self.initialize_parameters(X, random_state=init_seeds[i])

            # EM loop with spectral penalty
            with catch_warnings():
                simplefilter('ignore', ConvergenceWarning)
                params, opt_data = \
                    self._em_adaptive_pen(X=X, random_state=init_seeds[i])

            loss_val = opt_data['loss_val']
            success = opt_data['success']

            # update parameters if this initialization is better
            if i == 0:
                new_best = True
            else:
                if loss_val < min(init_loss_vals) and success:
                    new_best = True
                else:
                    new_best = False

            if new_best:
                best_params = params
                best_opt_data = opt_data
                best_opt_data['init'] = i
                best_opt_data['random_state'] = init_seeds[i]

            init_loss_vals.append(loss_val)
            init_success.append(success)

        best_opt_data['init_loss_vals'] = init_loss_vals
        best_opt_data['init_success'] = init_success

        # warn about wrong number of blocks
        if self.n_blocks is not None and \
                (not best_opt_data['n_blocks_est'] == self.n_blocks):

            warn('Eigenvalue penalty did not successfuly enforce'
                 'block diagonal constraint; we got {} blocks,'
                 'but asked for {} blocks'.
                 format(best_opt_data['n_blocks_est'],
                        self.n_blocks),
                 ConvergenceWarning)

        return best_params, best_opt_data

    def _em_adaptive_pen(self, X, random_state=None):
        """
        Run the EM algorithm to convergence. Increase spectral penalty
        value until we reach the requested number of blocks.
        """

        adapt_pen_history = {'n_blocks_est': [],
                             'opt_data': []}
        # TODO-warning: if random_state is None then we won't initialize
        # parameters to the same place when alpha is decreased

        self.__mode == 'lap_pen'

        # set lap pen to base
        self._initialize_eval_pen(X=X)
        eval_pen_init = deepcopy(self.eval_pen_)

        for t in range(self.n_pen_tries):

            if self.verbosity >= 1:
                time = datetime.now().strftime("%H:%M:%S")
                print('Trying eigenvalue penalty ({}/{}) at {}'.
                      format(t + 1, self.n_pen_tries, time))

            # initial_params = deepcopy(self._get_parameters())

            # run EM loop
            adpt_params, adpt_opt_data = self._em_loop(X=X)

            # check if we have  found enough blocks
            comm_summary = community_summary(adpt_params['bd_weights'],
                                             zero_thresh=self.zero_thresh)[0]

            n_blocks_est = comm_summary['n_communities']

            adpt_opt_data['n_blocks_est'] = n_blocks_est
            adpt_opt_data['eval_pen'] = deepcopy(self.eval_pen_)
            adapt_pen_history['n_blocks_est'].append(n_blocks_est)

            if self.history_tracking >= 1:
                adapt_pen_history['opt_data'].append(deepcopy(adpt_opt_data))

            # TODO: what to do about initialization
            if self.n_blocks is not None and n_blocks_est < self.n_blocks:
                # too few blocks, increase eval penalty
                self.update_eval_pen(increase=True)

            elif self.n_blocks is not None and n_blocks_est > self.n_blocks:
                # too many blocks, decrease eval penalty
                self.update_eval_pen(increase=False)

            else:
                break

        adpt_opt_data['adapt_pen_history'] = adapt_pen_history

        # check if we successed in getting the number of requested blocks
        if self.n_blocks is not None:
            success = n_blocks_est == self.n_blocks
        else:
            success = True

        adpt_opt_data['eval_pen_init'] = eval_pen_init
        adpt_opt_data['success'] = success
        adpt_opt_data['n_blocks_est'] = n_blocks_est

        ######################################
        # fine tune block diagonal structure #
        ######################################

        # opt history
        opt_data = {'adpt_opt_data': adpt_opt_data,
                    'success': success,
                    'n_blocks_est': n_blocks_est}

        # fine tune with fixed block diagonal structure
        if self.fine_tune_n_steps is not None and success:

            # re-set parameters
            max_n_steps = deepcopy(self.max_n_steps)
            self.max_n_steps = deepcopy(self.fine_tune_n_steps)
            self.__mode = 'fine_tune_bd'

            # True/False array of zero elements
            self.zero_mask_bd_ = \
                ~get_nonzero_block_mask(self.bd_weights_,
                                        tol=self.zero_thresh)[0]

            params, ft_opt_data = self._em_loop(X=X)

            # save data
            opt_data['adpt_params'] = adpt_params
            opt_data['ft_opt_data'] = ft_opt_data
            opt_data['loss_val'] = ft_opt_data['loss_val']

            # put back original parameters
            self.__mode = 'lap_pen'
            self.max_n_steps = max_n_steps

        else:
            # loss value should be negative log lik
            opt_data['loss_val'] = adpt_opt_data['history']['obs_nll'][-1]
            opt_data['adpt_params'] = None  # no need to save these again
            opt_data['ft_opt_data'] = None

            params = adpt_params

        return params, opt_data

    def compute_tracking_data(self, X, E_out=None):

        out = {}

        if E_out is None:
            E_out = self._e_step(X)

        # maybe track model history
        if self.history_tracking >= 2:
            out['model'] = deepcopy(self._get_parameters())

        if 'obs_nll' in E_out.keys():
            out['obs_nll'] = E_out['obs_nll']
        else:
            out['obs_nll'] = - self.score(X)

        # if we are fine tuning with a fixed zero mask the loss function
        # is just the observed negative log likelihood
        if self.__mode == 'fine_tune_bd':
            out['loss_val'] = out['obs_nll']
            return out

        # obs_nll = - self.score(X)
        # log_probs = self.log_probs(X)
        # obs_nll = - logsumexp(log_probs, axis=1).mean()

        if self.n_blocks is not None:
            B = self.n_blocks
        else:
            B = len(self.eval_weights)

        # evals of current step
        if 'evals' in E_out.keys():
            evals = E_out['evals']
        else:

            if self.lap == 'sym':

                evals, _ = geigh_sym_laplacian_bp(X=self.bd_weights_,
                                                  rank=B,
                                                  end='smallest',
                                                  method='tsym')

            elif self.lap == 'un':
                Lun = get_unnorm_laplacian_bp(self.bd_weights_)
                all_evals, all_evecs = eigh_wrapper(Lun)
                evals = all_evals[-B:]

        out['raw_eval_sum'] = sum(evals)

        if self.eval_weights is not None:

            eval_sum = evals.T @ asc_sort(self.eval_weights)

        else:
            # vanilla sum
            assert len(evals) == B
            eval_sum = sum(evals)

        out['eval_sum'] = eval_sum
        out['eval_loss'] = self.eval_pen_ * eval_sum
        out['evan_pen'] = deepcopy(self.eval_pen_)

        # overall loss
        out['loss_val'] = out['obs_nll'] + out['eval_loss']

        return out

    def _e_step(self, X):
        """
        Parameters
        ----------
        X:
            The observed data.


        Output
        ------
        E_out: dict
            E_out['log_resp']: array-like

            E_out['obs_nll']: float

            E_out['evals']: array-like, (n_blocks, )

            E_out['eig_var']: array-like

        """

        # standard E-step
        log_prob = self.log_probs(X)
        log_resp = self.log_resps(log_prob)

        obs_nll = - logsumexp(log_prob, axis=1).mean()

        if self.n_blocks is not None:
            B = self.n_blocks
        else:
            B = len(self.eval_weights)

        assert self.__mode in ['lap_pen', 'fine_tune_bd']
        if self.__mode == 'lap_pen':

            if self.lap == 'sym':

                evals, eig_var = geigh_sym_laplacian_bp(X=self.bd_weights_,
                                                        rank=B,
                                                        end='smallest',
                                                        method='tsym')

            elif self.lap == 'un':
                Lun = get_unnorm_laplacian_bp(self.bd_weights_)
                all_evals, all_evecs = eigh_wrapper(Lun)
                eig_var = all_evecs[:, -B:]
                evals = all_evals[-B:]

        if self.__mode == 'fine_tune_bd':
            evals = None
            eig_var = None

        return {'log_resp': log_resp,
                'obs_nll': obs_nll,
                'evals': evals,
                'eig_var': eig_var}

    def _m_step(self, X, E_out):
        log_resp = E_out['log_resp']

        view_params = self._m_step_clust_params(X=X, log_resp=log_resp)

        if self.__mode == 'lap_pen':
            eig_var = E_out['eig_var']
            weights = self._m_step_weights_lap_pen(X=X, log_resp=log_resp,
                                                   eig_var=eig_var)
        elif self.__mode == 'fine_tune_bd':
            assert hasattr(self, 'zero_mask_bd_')
            weights = \
                self._m_step_weights_fixed_zeros(X=X, log_resp=log_resp,
                                                 zero_mask=self.zero_mask_bd_)

        return {'views': view_params, 'bd_weights': weights}

    def _m_step_weights_lap_pen(self, X, log_resp, eig_var):

        # get log coefficient for bd_weights update
        n_samples = log_resp.shape[0]
        resp = np.exp(log_resp)
        nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
        a = nk / n_samples  # normalize so sum(a) == 1
        Gamma = a.reshape(*self.n_view_components)

        #################
        # Solve problem #
        #################

        # TODO:
        if self.n_blocks is not None:
            B = self.n_blocks
        else:
            B = len(self.eval_weights)

        bd_subprob_kws = {'Gamma': Gamma,
                          'eig_var': eig_var,
                          'epsilon': self.epsilon,
                          'alpha': self.eval_pen_,
                          'B': B,
                          'lap': self.lap,
                          'eta': self.eta,
                          'weights': self.eval_weights}

        init_val = 'guess'

        if self.lap == 'sym':
            bd_subprob_kws.update({'trim_od_constrs': True,
                                   'remove_redundant_contr': True,
                                   'remove_const_cols': True,
                                   'exclude_off_diag': False,
                                   'exclude_vdv_constr': self.exclude_vdv_constr})

        success = False

        cp_kws = {'solver': 'ECOS', 'max_iters': 500}

        # ECOS sometimes fails to converge, but the following hack
        # of multiplying the objective function by a constant
        # often rescues it!
        # TODO-FEAT: implement a better solver!
        mult_vals = [None, 1e-2, 1e2, 1e-4, 1e4, 1e-8, 1e8]
        for mult_val in mult_vals:
            try:
                cp_bd_var, objective, constraints = \
                    get_cp_problem(init_val=init_val,
                                   obj_mult=mult_val,
                                   **bd_subprob_kws)

                bd_weights_new, opt_val, prob = \
                    solve_problem_cp(var=cp_bd_var,
                                     objective=objective,
                                     constraints=constraints,
                                     cp_kws=cp_kws,
                                     # warm_start=(init_val is not None),
                                     verbosity=self.verbosity - 1)

                success = True

            except SolverError as e:
                if self.verbosity >= 1:
                    warn('ECOS failed with mult_val = {}.'
                         ' Error message: {}'.format(mult_val, e))

            if success:
                break

        # If ECOS still is not cooperating we can try removing the
        # VDV = I constraints
        if not success and self.lap == 'sym' and not self.exclude_vdv_constr:
            if self.verbosity >= 1:

                warn('Attempting backup by removing VDV'
                     'constraints')
            try:
                # turn off off-diag constraints
                # bd_subprob_kws['exclude_off_diag'] = True
                original = deepcopy(bd_subprob_kws['exclude_vdv_constr'])
                bd_subprob_kws['exclude_vdv_constr'] = True

                cp_bd_var, objective, constraints = \
                    get_cp_problem(init_val=init_val,
                                   **bd_subprob_kws)

                # turn back on off-diag constraints
                # bd_subprob_kws['exclude_off_diag'] = False
                bd_subprob_kws['exclude_vdv_constr'] = original

                bd_weights_new, opt_val, prob = \
                    solve_problem_cp(var=cp_bd_var,
                                     objective=objective,
                                     constraints=constraints,
                                     cp_kws=cp_kws,
                                     # warm_start=(init_val is not None),
                                     verbosity=self.verbosity - 1)

                success = True

            except SolverError as e:
                if self.verbosity >= 1:
                    warn('ECOS failed again with message: {}'.format(e))

        # if nothing is working then simply return Gamma which would be
        # the M step for the basic MVMM
        if not success:
            if self.verbosity >= 1:
                warn('All CVXPY solvers all failed; resorting to backup.')
            bd_weights_new = Gamma

        bd_weights_new = bd_weights_new.reshape(self.bd_weights_.shape)

        # ensure proper normalization that can be lost for numerical reasons
        # not normalizing causes the solver to throw errors sometimes
        # not that this normalization step can cause the loss function
        # to increase
        bd_weights_new = bd_weights_new * \
            (self.epsilon_tilde / bd_weights_new.sum())

        return bd_weights_new

    def _m_step_weights_fixed_zeros(self, X, log_resp, zero_mask):
        # get log coefficient for bd_weights update
        n_samples = log_resp.shape[0]
        resp = np.exp(log_resp)
        nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
        a = nk / n_samples  # normalize so sum(a) == 1
        Gamma = a.reshape(*self.n_view_components)
        Gamma = Gamma / Gamma.sum()

        # setup CVXPY problem

        cp_bd_var, objective, constraints = \
            get_cp_prob_fixed_zeros(Gamma=Gamma,
                                    zero_mask=zero_mask,
                                    epsilon=self.epsilon)

        # solve cvxpy problem form bd_weights
        cp_kws_backups = [{'solver': 'ECOS', 'max_iters': 200},
                          {'solver': 'SCS'}]

        try:
            bd_weights_new, opt_val, prob = \
                solve_problem_cp_backups(cp_kws_backups=cp_kws_backups,
                                         var=cp_bd_var,
                                         objective=objective,
                                         constraints=constraints,
                                         verbosity=self.verbosity - 1)

            bd_weights_new = bd_weights_new.reshape(self.bd_weights_.shape)

        except SolverError:
            if self.verbosity >= 1:
                warn('CVXPY solvers failed for fixed zeros;'
                     'resorting to backup.')
            bd_weights_new = Gamma

        # ensure proper normalization that can be lost for numerical reasons
        # this seems to occasionally cause issues
        bd_weights_new = bd_weights_new * \
            (self.epsilon_tilde / bd_weights_new.sum())

        return bd_weights_new

    def _n_weight_parameters(self):
        """
        Returns the number of parameters for pi
        """
        return (self.bd_weights_ > self.zero_thresh).sum()


def get_cp_problem(lap='sym', **kwargs):

    if lap == 'sym':
        return get_cp_problem_sym_lap(**kwargs)
    elif lap == 'un':
        return get_cp_problem_un_lap(**kwargs)
    else:
        raise ValueError('Bad argument for lap')


BlockDiagMVMM.__doc__ = dedent("""\
Block diagonally constrained multi-view mixture model.

TODO-DOC

Parameters
----------
base_view_models: list of mixture models
    Mixture models for each view. These should specify the number of view components.

n_blocks: int

eval_weights:

rel_eta: float

eval_pen_base: float

eval_pen_incr: float

eval_pen_decr: float

n_pen_tries: int

init_pen_c: float

init_pen_use_bipt_sp: bool

init_pen_K: str

lap: str

rel_epsilon: float

exclude_vdv_constr: bool

cp_kws: dict

zero_thresh: float

fine_tune_n_steps: int

{em_param_docs}

Attributes
----------
weights_

weights_mat_

bd_weights_

zero_mask_

metadata_

""".format(**_em_docs))
