from packages import *

logger = logging.getLogger('mcmc')
class LinearModel:
    '''
    Simple linear model with a single output (y) given the covariates x_1...x_M of the form:
    y = w_1 * x_1 + ... + w_M * x_M + b
    where M = number of features, w are the weights, and b is the bias.
    '''
    def __init__(self):
        self.w = None
        self.b = None 
    
    def encode(self, theta):
        """Split the parameter vector into w and b and store in the model"""
        self.w = theta[0:-1]
        self.b = theta[-1] 
    
    def predict(self, x_in):
        y_out = x_in.dot(self.w) + self.b 
        return y_out
    
    def evaluate_proposal(self, data, theta):
        '''
        Encode the proposed parameters and then use the model to predict
        '''
        self.encode(theta)
        prediction = self.predict(data)
        return prediction


class MCMC_CHAIN:
    """
    Combined and Improved MCMC class with adaptive hyperparameters, 
    better diagnostics, and enhanced numerical stability
    """
    def __init__(self, n_samples, n_burnin, x_data, y_data, x_test=None, y_test=None, 
                 seed=None, variable_name="Unknown", verbose=False, use_adaptive=True, sampler_type="RWM"):
        self.n_samples = n_samples
        self.n_burnin = n_burnin
        self.x_data = x_data
        self.y_data = y_data
        self.x_test = x_test if x_test is not None else x_data
        self.y_test = y_test if y_test is not None else y_data
        self.variable_name = variable_name
        self.verbose = verbose
        self.use_adaptive = use_adaptive
        self.sampler_type = sampler_type
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)

        # Initialize model
        self.model = LinearModel()
        self.theta_size = x_data.shape[1] + 1  # weights for each feature and bias term
        self.X_design = np.column_stack([self.x_data, np.ones(self.x_data.shape[0])])
        # Set hyperparameters (adaptive or default)
        if use_adaptive:
            self.set_default_hyperparameters()
            self.set_optimal_step_sizes()
            # Increase burn-in for better convergence
            self.n_burnin = max(n_burnin, int(n_samples * 0.2))
        else:
            self.set_default_hyperparameters()
            self.adjust_step_sizes()

        # Store diagnostic information
        self.acceptance_rates = []
        self.step_sizes = {'theta': [], 'eta': []}
        self.parameter_traces = None
        self.log_likelihood_trace = None
        
        # Store output
        self.pos_theta = None
        self.pos_tau = None
        self.pos_eta = None
        self.rmse_data = None

    def set_default_hyperparameters(self):
        """Improved weakly informative prior and stable step size"""
        target_var = np.var(self.y_data)
        
        # Weakly informative but stable prior
        self.sigma_squared = 5
        self.nu_1 = 1 #2.1     # Finite mean and variance
        self.nu_2 = 0.5  #1.0     # Weak but stable scale

         # Adjust step sizes based on sampler type
        if self.sampler_type == "MALA":
            # MALA typically needs smaller step sizes due to gradient guidance
            self.step_theta = 0.01  # Smaller than RWM
            self.step_eta = 0.005   # Smaller than RWM
        else:  # RWM
            self.step_theta = 0.02
            self.step_eta = 0.01

    def set_adaptive_hyperparameters(self):
        """Set hyperparameters based on data characteristics"""
        # Adaptive prior variance based on feature scale
        feature_vars = np.var(self.x_data, axis=0)
        target_var = np.var(self.y_data)
        
        # Set sigma_squared to be more informative but not restrictive
        self.sigma_squared = max(1.0, np.mean(feature_vars) * 10)
        
        # Set inverse gamma parameters for tau^2 based on target variance
        self.nu_1 = 3.0  # Shape parameter (higher = more informative)
        self.nu_2 = 2.0 * target_var  # Scale parameter based on data
        
        if self.verbose:
            print(f"Adaptive hyperparameters: sigma²={self.sigma_squared:.3f}, "
                  f"nu1={self.nu_1:.3f}, nu2={self.nu_2:.3f}")
    
    def set_optimal_step_sizes(self):
        """
        Corrected implementation of optimal step sizes based on Roberts & Rosenthal (2001)
        
        Literature:
        - Roberts & Rosenthal (2001): "Optimal scaling for various Metropolis-Hastings algorithms"
        - Gelman et al. (1996): "Efficient Metropolis jumping rules"
        - Haario et al. (2001): "An adaptive Metropolis algorithm"
        """
        
        d_theta = self.theta_size  # Dimension of theta (including bias)
        
        if self.sampler_type == "MALA":
            # Pure isotropic MALA
            optimal_scale = (1.65 ** 2) / (d_theta ** (1/3))
            self.step_theta = 0.8 * optimal_scale
            self.step_eta = None
           
        else:
        
            # Roberts & Rosenthal (2001) optimal scaling: 2.38²/d
            optimal_scale = (2.38 ** 2) / d_theta
            
            if self.verbose:
                print(f"Using Roberts & Rosenthal (2001) optimal scaling: 2.38²/{d_theta} = {optimal_scale:.4f}")
        
            # Method 1: Use empirical covariance if enough data (Haario et al. 2001)
            if self.x_data.shape[0] > d_theta + 5:  # Need more samples than parameters
                try:
                    # Create design matrix including bias term
                    X_design = np.column_stack([self.x_data, np.ones(self.x_data.shape[0])])
                    
                    # Empirical covariance matrix
                    empirical_cov = np.cov(X_design, rowvar=False)
                    
                    # Add regularization to ensure positive definite (Haario et al. 2001)
                    epsilon = 1e-6
                    self.proposal_cov_theta = optimal_scale * empirical_cov + epsilon * np.eye(d_theta)
                    
                    # For scalar proposals as fallback
                    self.step_theta = np.sqrt(optimal_scale * np.mean(np.diag(empirical_cov)))
                    
                    if self.verbose:
                        print(f"Using empirical covariance matrix proposal (Haario et al. 2001)")
                        print(f"Fallback scalar step_theta: {self.step_theta:.5f}")
                        
                except np.linalg.LinAlgError:
                    # Fallback to identity if covariance issues
                    self.proposal_cov_theta = optimal_scale * np.eye(d_theta)
                    self.step_theta = np.sqrt(optimal_scale)
                    
                    if self.verbose:
                        print(f"Covariance estimation failed, using identity matrix")
                        
            else:
                # Method 2: Identity matrix scaled by optimal factor (too few samples)
                self.proposal_cov_theta = optimal_scale * np.eye(d_theta)
                self.step_theta = np.sqrt(optimal_scale)
                    
                if self.verbose:
                    print(f"Insufficient data for empirical covariance, using scaled identity")
                    print(f"Step_theta: {self.step_theta:.4f}")
                
            # Step size for eta (log variance parameter)
            # Based on target variable scale
            y_var = float(np.var(self.y_data))
            if y_var > 0:
                self.step_eta = (2.38 / np.sqrt(1)) * np.sqrt(np.log1p(max(y_var, 1e-12)))
                if self.verbose:
                    print(f"[RWM] step_eta={self.step_eta:.4f}")
            else:
                self.step_eta = 0.01
                
        if self.verbose:
            print(f"Step_eta: {self.step_eta:.5f}")
            
        # Store initial step sizes for tracking
        self.initial_step_theta = self.step_theta
        self.initial_step_eta = self.step_eta

    def adjust_step_sizes(self):
        """Adjust step sizes based on variable scale (original method)"""
        if hasattr(self, 'x_data') and self.x_data is not None:
            x_std = np.std(self.x_data, axis=0)
            if len(x_std) > 0:
                base_step = max(0.05, np.mean(x_std) * 0.05)
                self.step_theta = base_step * 0.5 if self.sampler_type == "MALA" else base_step
            
            if hasattr(self, 'y_data') and self.y_data is not None:
                y_var = np.var(self.y_data)
                if y_var > 0:
                    base_step = max(0.02, np.log(y_var) * 0.02)
                    self.step_eta = base_step * 0.5 if self.sampler_type == "MALA" else base_step

    def rmse(self, predictions, targets):
        """Calculate Root Mean Square Error"""
        return np.sqrt(((predictions - targets) ** 2).mean())
    
    def log_posterior(self, theta, tausq, test=False):
        """
        Calculate log posterior = log likelihood + log prior
        Used for gradient calculation in MALA
        """
        # Get likelihood components
        likelihood_result = self.likelihood_function(theta, tausq, test=test)
        log_likelihood = likelihood_result[0]
        
        # Get log prior
        log_prior = self.prior(self.sigma_squared, self.nu_1, self.nu_2, theta, tausq)
        
        return log_likelihood + log_prior

    def gradient_log_posterior_theta(self, theta, tausq):
        """
        Calculate gradient of log posterior with respect to theta
        This is needed for MALA proposals
        """
        # Gradient of log likelihood with respect to theta
        tausq_safe = max(tausq, 1e-8)
        model_prediction = self.model.evaluate_proposal(self.x_data, theta)
        residuals = self.y_data - model_prediction  
        # Gradient of log likelihood: (1/tausq) * X^T * (y - X*theta)
        grad_likelihood = (1.0 / tausq_safe) * self.X_design.T @ residuals
        
        # Gradient of log prior: -theta / sigma_squared
        grad_prior = -theta / self.sigma_squared
        
        return grad_likelihood + grad_prior

    def _gibbs_tausq(self, theta):
        """
        Sample tau^2 | theta, y ~ Inv-Gamma( nu1 + n/2,  nu2 + RSS/2 )
        consistent with prior log p(tau^2) ∝ -(nu1+1)log(tau^2) - nu2/tau^2.
        """
        resid = self.y_data - self.model.evaluate_proposal(self.x_data, theta)
        rss = float(np.sum(resid ** 2))
        n = len(self.y_data)

        alpha = self.nu_1 + 0.5 * n            # shape
        beta  = self.nu_2 + 0.5 * rss          # scale

        g = np.random.gamma(shape=alpha, scale=1.0 / beta)  # Gamma(alpha, 1/beta)
        tausq = float(1.0 / g)                                # InvGamma
        tausq = max(tausq, 1e-12)
        eta = float(np.log(tausq))
        return tausq, eta

    def mala_proposal_theta(self, theta_current, tausq_current):
        """
        Generate MALA proposal for theta
        """
        # Calculate gradient at current position
        grad_current = self.gradient_log_posterior_theta(theta_current, tausq_current)
        
        # MALA proposal: current + step_size²/2 * gradient + step_size * noise
        drift = (self.step_theta**2 / 2.0) * grad_current
        noise = self.step_theta * np.random.normal(0, 1, self.theta_size)
        
        theta_proposal = theta_current + drift + noise
        
        return theta_proposal


    def mala_log_alpha_theta(self, theta_current, theta_prop, tausq_current):
        """
        Log MH ratio for θ-only MALA with tau^2 held fixed.
        """
        lp_cur  = self.log_posterior(theta_current, tausq_current)
        lp_prop = self.log_posterior(theta_prop,   tausq_current)

        g_cur  = self.gradient_log_posterior_theta(theta_current, tausq_current)
        g_prop = self.gradient_log_posterior_theta(theta_prop, tausq_current)

        h2 = self.step_theta ** 2
        mean_fwd = theta_current + 0.5 * h2 * g_cur
        mean_bwd = theta_prop   + 0.5 * h2 * g_prop

        inv_var = 1.0 / h2
        log_q_fwd = -0.5 * inv_var * np.sum((theta_prop    - mean_fwd) ** 2)
        log_q_bwd = -0.5 * inv_var * np.sum((theta_current - mean_bwd) ** 2)

        return (lp_prop - lp_cur) + (log_q_bwd - log_q_fwd)

    def likelihood_function(self, theta, tausq, test=False):
        """
        Calculate the likelihood of the data given the parameters
        Enhanced with better numerical stability when use_adaptive=True
        """
        if test:
            x_data = self.x_test
            y_data = self.y_test
        else:
            x_data = self.x_data
            y_data = self.y_data
            
        # Make prediction
        model_prediction = self.model.evaluate_proposal(x_data, theta)
        
        if self.use_adaptive:
            # Improved version with numerical stability
            tausq_safe = max(tausq, 1e-8)
            residuals = y_data - model_prediction
            
            # Log likelihood with better numerical stability
            log_likelihood = (-0.5 * len(y_data) * np.log(2 * np.pi * tausq_safe) - 
                            0.5 * np.sum(residuals**2) / tausq_safe)
            
            # Simulation for imputation
            model_simulation = model_prediction + np.random.normal(0, np.sqrt(tausq_safe), 
                                                                size=model_prediction.shape)
            
            # RMSE
            accuracy = np.sqrt(np.mean(residuals**2))
        else:
            # Original version
            model_simulation = model_prediction + np.random.normal(0, tausq, size=model_prediction.shape) 
            accuracy = self.rmse(model_prediction, y_data)
            
            # Calculate the log likelihood
            log_likelihood = np.sum(-0.5 * np.log(2 * np.pi * tausq) - 
                                  0.5 * np.square(y_data - model_prediction) / tausq)
        
        return [log_likelihood, model_prediction, model_simulation, accuracy]

    def prior(self, sigma_squared, nu_1, nu_2, theta, tausq): 
        """Calculate the prior of the parameters"""
        n_params = self.theta_size
        part1 = -1 * (n_params / 2) * np.log(sigma_squared)
        part2 = 1 / (2 * sigma_squared) * (sum(np.square(theta)))
        log_prior = part1 - part2 - (1 + nu_1) * np.log(tausq) - (nu_2 / tausq)
        return log_prior
    
    def sampler(self):
        """
        Run the MCMC sampler with all improvements integrated
        """
        # Define empty arrays to store the sampled posterior values
        pos_theta = np.ones((self.n_samples, self.theta_size))
        pos_tau = np.ones((self.n_samples, 1))
        pos_eta = np.ones((self.n_samples, 1))

        # Record outputs
        pred_y = np.zeros((self.n_samples, self.x_data.shape[0]))
        sim_y = np.zeros((self.n_samples, self.x_data.shape[0]))
        rmse_data = np.zeros(self.n_samples)
        test_pred_y = np.ones((self.n_samples, self.x_test.shape[0]))
        test_sim_y = np.ones((self.n_samples, self.x_test.shape[0]))
        test_rmse_data = np.zeros(self.n_samples)

        # Store log likelihood for diagnostic plots
        log_likelihood_trace = np.zeros(self.n_samples)

        # Initialization
        theta = np.random.randn(self.theta_size)
        pred_y[0,] = self.model.evaluate_proposal(self.x_data, theta)

        # Initialize eta
        eta = float(np.log(np.var(pred_y[0,] - self.y_data)))
        tausq_current = float(np.exp(eta))

        # Calculate initial prior and likelihood
        prior_val = self.prior(self.sigma_squared, self.nu_1, self.nu_2, theta, tausq_current)
        [likelihood, pred_y[0,], sim_y[0,], rmse_data[0]] = self.likelihood_function(theta, tausq_current)
        [_, test_pred_y[0,], test_sim_y[0,], test_rmse_data[0]] = self.likelihood_function(theta, tausq_current, test=True)
        
        log_likelihood_trace[0] = likelihood

        n_accept = 0
        recent_accepts = 0
        adaptive_window = 100

        # Run MCMC sampling
        for ii in np.arange(1, self.n_samples):

            if self.sampler_type == "MALA":
                # ===== θ-only MALA =====
                tausq_current = np.exp(eta)  # hold τ² fixed when proposing θ

                theta_prop = self.mala_proposal_theta(theta, tausq_current)

                # MALA asymmetric correction
                try:
                    log_alpha = self.mala_log_alpha_theta(theta, theta_prop, tausq_current)
                except Exception as e:
                    if self.verbose:
                        print(f"MALA log-alpha failed: {e}")
                    log_alpha = -np.inf

                if np.log(np.random.rand()) < min(0.0, log_alpha):
                    theta = theta_prop
                    n_accept += 1
                    recent_accepts += 1

                # record θ draw
                pos_theta[ii,] = theta

                # Gibbs update for τ² (and η=log τ²)
                tausq_current, eta = self._gibbs_tausq(theta)
                pos_tau[ii,] = tausq_current
                pos_eta[ii,] = eta

                # update likelihood and predictions at new state
                [ll,  pred_y[ii,],  sim_y[ii,],  rmse_data[ii]] = self.likelihood_function(theta, tausq_current)
                [_,   test_pred_y[ii,], test_sim_y[ii,], test_rmse_data[ii]] = self.likelihood_function(theta, tausq_current, test=True)

                log_likelihood_trace[ii] = ll
                likelihood = ll
                prior_val = self.prior(self.sigma_squared, self.nu_1, self.nu_2, theta, tausq_current)

            else:
               # Sample new values using Gaussian random walk
                theta_proposal = theta + np.random.normal(0, self.step_theta, self.theta_size)
                eta_proposal   = eta   + np.random.normal(0, self.step_eta,   1)
                tausq_proposal = np.exp(eta_proposal)

                # Calculate prior and likelihood for proposal
                prior_proposal = self.prior(self.sigma_squared, self.nu_1, self.nu_2,
                                            theta_proposal, tausq_proposal)
                [likelihood_proposal, pred_y[ii,], sim_y[ii,], rmse_data[ii]] = \
                    self.likelihood_function(theta_proposal, tausq_proposal)
                [_, test_pred_y[ii,], test_sim_y[ii,], test_rmse_data[ii]] = \
                    self.likelihood_function(theta_proposal, tausq_proposal, test=True)

                # Calculate acceptance probability
                diff_likelihood  = likelihood_proposal - likelihood
                diff_prior= prior_proposal      - prior_val
                mh_prob   = min(1.0, np.exp(diff_likelihood + diff_prior))

                # accept/reject (RWM only)
                u = np.random.uniform(0, 1)
                if u < mh_prob:
                    n_accept += 1
                    recent_accepts += 1
                    likelihood = likelihood_proposal
                    log_likelihood_trace[ii] = likelihood_proposal
                    prior_val = prior_proposal
                    theta = theta_proposal
                    eta   = eta_proposal
                    pos_theta[ii,] = theta_proposal
                    pos_tau[ii,]   = tausq_proposal
                    pos_eta[ii,]   = eta_proposal
                else:
                    log_likelihood_trace[ii] = likelihood
                    pos_theta[ii,] = pos_theta[ii-1,]
                    pos_tau[ii,]   = pos_tau[ii-1,]
                    pos_eta[ii,]   = pos_eta[ii-1,]
                    pred_y[ii,]    = pred_y[ii-1,]
                    sim_y[ii,]     = sim_y[ii-1,]
                    rmse_data[ii]  = rmse_data[ii-1]
                    test_pred_y[ii,] = test_pred_y[ii-1,]
                    test_sim_y[ii,]  = test_sim_y[ii-1,]
                    test_rmse_data[ii] = test_rmse_data[ii-1]

            # ===== Adaptation =====
            if ii % adaptive_window == 0 and ii > 0:
                recent_accept_rate = recent_accepts / adaptive_window
                self.acceptance_rates.append(recent_accept_rate)
                self.step_sizes['theta'].append(self.step_theta)
                if self.sampler_type != "MALA":  # only track eta for RWM
                    self.step_sizes['eta'].append(self.step_eta)

                if self.sampler_type == "MALA":
                    # aim ≈ 0.57
                    if recent_accept_rate < 0.50:
                        self.step_theta *= 0.9
                        print(f"{self.sampler_type}: Decreasing step size to theta:{self.step_theta:.3f}")
                    elif recent_accept_rate > 0.65:
                        self.step_theta *= 1.1
                        print(f"{self.sampler_type}: Increasing step size to theta:{self.step_theta:.3f}")
                else:
                    # aim 0.15–0.30
                    if recent_accept_rate < 0.15:
                        self.step_theta *= 0.9
                        self.step_eta   *= 0.9
                        print(f"{self.sampler_type}: Decreasing step size to theta:{self.step_theta:.3f}, eta:{self.step_eta:.3f}")
                    elif recent_accept_rate > 0.30:
                        self.step_theta *= 1.1
                        self.step_eta   *= 1.1
                        print(f"{self.sampler_type}: Increasing step size to theta:{self.step_theta:.3f}, eta:{self.step_eta:.3f}")
                recent_accepts = 0

        # summary
        accept_rate = (n_accept / self.n_samples) * 100
        print(f'{self.sampler_type}: {accept_rate:.3f}% of proposals were accepted')

        self.pos_theta = pos_theta[self.n_burnin:, ]
        self.pos_tau   = pos_tau[self.n_burnin:, ]
        self.pos_eta   = pos_eta[self.n_burnin:, ]
        self.rmse_data = rmse_data[self.n_burnin:]

        self.parameter_traces    = pos_theta
        self.log_likelihood_trace= log_likelihood_trace

        results_dict = {'w{}'.format(_): self.pos_theta[:, _].squeeze() for _ in range(self.theta_size-1)}
        results_dict['b']   = self.pos_theta[:, -1].squeeze()
        results_dict['tau'] = self.pos_tau.squeeze()
        results_dict['rmse']= self.rmse_data.squeeze()

        pred_dict = {
            'train_pred': pred_y[self.n_burnin:,:],
            'train_sim' : sim_y[self.n_burnin:,:],
            'test_pred' : test_pred_y[self.n_burnin:,:],
            'test_sim'  : test_sim_y[self.n_burnin:,:],
        }

        results_df = pd.DataFrame.from_dict(results_dict)

        if self.verbose:
            self.run_diagnostics()

        return results_df, pred_dict

    def run_diagnostics(self):
        """Run comprehensive diagnostic checks on the MCMC results"""
        if self.parameter_traces is None or self.log_likelihood_trace is None:
            print("No diagnostic data available")
            return
            
        print(f"\n=== MCMC Diagnostics for {self.variable_name} ===")
        print(f"Total samples: {self.n_samples}")
        print(f"Burn-in samples: {self.n_burnin}")
        print(f"Effective samples: {self.n_samples - self.n_burnin}")
        
        # Acceptance rate diagnostics
        if len(self.acceptance_rates) > 0:
            final_rate = self.acceptance_rates[-1]
            avg_rate = np.mean(self.acceptance_rates)
            print(f"Final acceptance rate: {final_rate:.3f}")
            print(f"Average acceptance rate: {avg_rate:.3f}")
            
            if self.sampler_type == "MALA":
                target_range = (0.5, 0.65)
                print(f"MALA target range: {target_range[0]}-{target_range[1]}")
            else:
                target_range = (0.2, 0.5)
                print(f"RWM target range: {target_range[0]}-{target_range[1]}")
            
            if target_range[0] <= final_rate <= target_range[1]:
                print(f"✓ Acceptance rate is in good range for {self.sampler_type}")
            else:
                print(f"⚠ Acceptance rate may be suboptimal for {self.sampler_type}")
        # Parameter convergence check (simplified)
        if self.pos_theta is not None:
            # Check if parameters have converged by comparing first and second half
            n_effective = len(self.pos_theta)
            if n_effective > 100:
                first_half = self.pos_theta[:n_effective//2]
                second_half = self.pos_theta[n_effective//2:]
                
                convergence_issues = 0
                for i in range(self.theta_size):
                    mean1 = np.mean(first_half[:, i])
                    mean2 = np.mean(second_half[:, i])
                    std_pooled = np.sqrt((np.var(first_half[:, i]) + np.var(second_half[:, i])) / 2)
                    
                    if std_pooled > 0:
                        diff_standardized = abs(mean1 - mean2) / std_pooled
                        if diff_standardized > 0.1:  # Simple convergence check
                            convergence_issues += 1
                
                if convergence_issues == 0:
                    print("✓ Parameters appear to have converged")
                else:
                    print(f"⚠ {convergence_issues}/{self.theta_size} parameters may not have converged")
        
        # Log-likelihood stability
        if self.log_likelihood_trace is not None:
            post_burnin_ll = self.log_likelihood_trace[self.n_burnin:]
            if len(post_burnin_ll) > 10:
                ll_trend = np.polyfit(range(len(post_burnin_ll)), post_burnin_ll, 1)[0]
                if abs(ll_trend) < 0.001:
                    print("✓ Log-likelihood is stable")
                else:
                    print(f"⚠ Log-likelihood shows trend: {ll_trend:.6f}")
        
        # Step size adaptation summary
        if self.step_sizes['theta'] and self.step_sizes['eta']:
            print(f"Step size adaptation: theta {self.step_sizes['theta'][0]:.5f} → {self.step_sizes['theta'][-1]:.5f}")
            print(f"                     eta {self.step_sizes['eta'][0]:.5f} → {self.step_sizes['eta'][-1]:.5f}")
        
        print("=" * 50)
    
    def get_diagnostics_summary(self):
        """Return diagnostic summary as a dictionary for programmatic use"""
        if self.parameter_traces is None:
            return {}
            
        diagnostics = {
            'variable_name': self.variable_name,
            'n_samples': self.n_samples,
            'n_burnin': self.n_burnin,
            'effective_samples': self.n_samples - self.n_burnin
        }
        
        if len(self.acceptance_rates) > 0:
            diagnostics['final_acceptance_rate'] = self.acceptance_rates[-1]
            diagnostics['avg_acceptance_rate'] = np.mean(self.acceptance_rates)
            diagnostics['acceptance_rate_ok'] = 0.2 <= self.acceptance_rates[-1] <= 0.5
        
        return diagnostics