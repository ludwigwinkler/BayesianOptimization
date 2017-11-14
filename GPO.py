import numpy as np
import sklearn
import sklearn.gaussian_process as gp
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


from scipy.stats import norm
from scipy.optimize import minimize

import matplotlib.pyplot as plt

def f(x):
        """The function to predict."""
        return x * np.sin(x)

def df(x):
        return np.sin(x) + x * np.cos(x)

def EI(_x, _gp, _prev_best_value):

        x_to_predict = _x.reshape(-1, 1)

        mu, sigma = _gp.predict(x_to_predict, return_std=True) #  Building the 'blue tube' with the standard deviations

        loss_optimum = _prev_best_value #  Currently the best value for which we search for a better value by looking for the maximum value in the 95% confidence bound, basically the max value in the 'blue tube'

        with np.errstate(divide='ignore'):
                Z = (mu-loss_optimum)/sigma #  The mean mu at every point minus the currently best value normalized with the points std
                EI = (mu - loss_optimum) * norm.cdf(Z) + sigma * norm.pdf(Z)
                PI = norm.cdf(Z)*sigma
                ludi = mu - loss_optimum + 1.96 * sigma
                # From 'TutBayesOpt_Brochu.pdf' page 13 in the Material Folder

        return EI

def plot_gp(_X, _y, _x_pred, _y_pred, _sigmas, _utility, _show=True):

        fig = plt.figure(figsize=(20,20))
        plt.plot(_x_pred, f(_x_pred), 'r:', label=u'$f(x) = x\,\sin(x)$') #  The true function
        plt.plot(_X, _y, 'r.', markersize=10, label=u'Observations') #  Observations so far
        plt.plot(_x_pred, _utility, 'b:', label=u'utility') #  Utility function: predicted_mean - currently_best_value + 1.96*standard_deviation
        plt.plot([np.argmax(utility)/100], [utility[np.argmax(utility)]], 'r*', markersize=10) #  New predicted max which will be evaluated
        plt.plot(_x_pred, _y_pred, 'b-', label=u'Prediction') #  Predicted mean
        plt.fill(np.concatenate([_x_pred, _x_pred[::-1]]),
                 np.concatenate([_y_pred - 1.9600 * _sigmas, (_y_pred + 1.9600 * _sigmas)[::-1]]),
                 alpha=.5,
                 fc='b',
                 ec='None',
                 label='95% confidence interval')
        plt.xlabel('$x$')
        plt.ylabel('$f(x)$')
        plt.ylim(-10, 20)
        plt.xticks(np.arange(0,10,1))
        plt.yticks(np.arange(-10,20,1))
        plt.grid()
        plt.legend(loc='upper left')
        if _show == True:
                plt.show()

# kernel = kernel = C(1.0, (0.01, 100)) * RBF(10, (0.01, 100))
kernel = RBF(10, (0.01, 100)) #  Defining the kernel
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, alpha=1e-5) #  Initializing the GP regressor
obs = np.array([3.]).reshape((-1,1)) #  Setting a first obs as the start for the search
x = np.atleast_2d(np.linspace(0,10,1000)).T #  All the x's with which we build the tube and among which we look for the maximum

for i in range(20):

        obs_y = f(obs).ravel() # Obtain deterministic value from the function f
        gp.fit(obs, obs_y) #  Fit the GP to the observations we have
        y_pred, sigma = gp.predict(x, return_std=True) #  Build the 'blue tube' with the GP with predicted mean and standard deviation

        utility = EI(x, gp, np.max(obs_y)) # Utility function which subtracts the currently best observed (!) value from the top of the 'blue tube'

        print('Iteration ', i, np.sum(utility))
        print('prev best: y=', np.max(y_pred))
        print('new best: x=', np.argmax(utility)/100, ' y(predicted/actually)=(',utility[np.argmax(utility)]+np.max(y_pred),'/', f(np.argmax(utility)/100), ') utility: ', utility[np.argmax(utility)])
        print()

        plot_gp(obs, obs_y, x, y_pred, sigma, _utility = utility, _show=True)
        obs = np.append(obs, np.array(np.argmax(utility)/100).reshape((1,1)), axis=0) #  Add the largest value above the currently best observed value in the 'blue tube' to the current observations
        obs_y = f(obs).ravel()




