import numpy as np
np.core.arrayprint._line_width = 300
import scipy
import sklearn
import sklearn.metrics.pairwise
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


def f(x):
        return x*np.sin(x)

def df(x):
        return np.sin(x) + x*np.cos(x)

class Kernel:
        '''
        Class to conveniently access kernel specific values
        '''
        def __init__(self):
                self.sigma=1


class GP:
        '''
        X: (samples, features)
        y: (samples,1)
        '''
        def __init__(self, _X=None, _function=None, _dfunction=None):

                self.X = _X #  Observed datapoints (features x samples)
                self.function = _function #  True, deterministic function
                self.dfunction = _dfunction #  Derivative function (e.g. f(x)=x*sin(x) -> df(x) = sin(x) + x*cos(x))
                self.y = self.function(self.X) #  Observed targets (samples x 1)
                self.dy = self.dfunction(self.X)
                self.sigma_kernel = 10
                self.noise = 1e-5

                self.K = self.RBF_Kernel(self.X, self.X.T)
                self.K_inv = self.invert_matrix(self.K, _noise= 1e-10)
                self.K_inv_noise = self.invert_matrix(self.K, _noise= 1e-10)

                self.K_with_derivatives = self.construct_K_with_derivatives()
                self.K_with_derivatives_inv = self.invert_matrix(self.K_with_derivatives, _noise=1e-10)

                self.plotting_resolution = 10000

        def construct_K_with_derivatives(self):

                K = self.RBF_Kernel(self.X, self.X.T)
                K_d = self.RBF_Kernel_onederivative(self.X, self.X.T)
                K_dd = self.RBF_Kernel_twoderivative(self.X, self.X.T)

                K_upperhalf = np.concatenate((K, K_d), axis=1)
                K_lowerhalf = np.concatenate((K_d, K_dd), axis=1)

                K_with_derivatives = np.concatenate((K_upperhalf, K_lowerhalf), axis=0)

                # print('K_with_derivatives')
                # print(K_with_derivatives)
                # print()
                # print('K')
                # print(self.K)
                # print()
                return K_with_derivatives

        def predict(self, _xs):

                k = self.RBF_Kernel(_xs, self.X.T)
                k_starstar = self.RBF_Kernel(_xs, _xs.T)

                mean = k.dot(self.K_inv).dot(self.y)
                sigma = np.diag(k_starstar - k.dot(self.K_inv).dot(k.T)).reshape((-1,1))

                print('predict')
                print(k.shape)

                return mean, sigma

        def predict_with_derivatives(self, _xs):

                k = self.RBF_Kernel(_xs, self.X.T)
                k_derivative = self.RBF_Kernel(_xs, self.X.T)
                k_with_derivatives = np.concatenate((k,k_derivative), axis=1)
                k_starstar = self.RBF_Kernel(_xs, _xs.T)

                y_with_derivatives = np.concatenate((self.y, self.dy),axis=0)
                # print('y')
                # print(self.y)
                # print('dy')
                # print(self.dy)

                mean_with_derivatives = k_with_derivatives.dot(self.K_with_derivatives_inv).dot(y_with_derivatives)
                # print(k_with_derivatives.shape, self.K_with_derivatives_inv.shape, y_with_derivatives.shape, '=', mean_with_derivatives.shape)
                sigma_with_derivatives = np.diag(k_starstar - k_with_derivatives.dot(self.K_with_derivatives_inv).dot(k_with_derivatives.T)).reshape((-1,1))

                return mean_with_derivatives, sigma_with_derivatives

        def plot(self, _xmin = 0, _xmax = 10):
                xs = np.linspace(_xmin, _xmax, self.plotting_resolution).reshape((-1,1))

                mean, sigma = self.predict(xs)
                mean_with_derivatives, sigma_with_derivatives = self.predict_with_derivatives(xs)

                # print('dy')
                # print(self.dy)
                # print('mean_with_derivative')
                # print(mean_with_derivatives[3000,0])

                print(mean[:10,:])
                print(mean_with_derivatives[:10,:])
                print(sigma[:10,:])
                print(sigma_with_derivatives[:10,:])


                fig = plt.figure(figsize=(10,10))
                plt.plot(xs, self.function(xs), 'r:', label=u'$f(x) = x\,\sin(x)$') #  The true function
                plt.plot(self.X, self.y, 'r.', markersize=10)#, label=u'Observations') #  Observations so far
                # plt.plot(_x_pred, _utility, 'b:', label=u'utility') #  Utility function: predicted_mean - currently_best_value + 1.96*standard_deviation
                # # plt.plot([np.argmax(utility)/100], [utility[np.argmax(utility)]], 'r*', markersize=10) #  New predicted max which will be evaluated
                plt.plot(xs, mean, 'b-', label=u'Mean Prediction') #  Predicted mean
                plt.plot(xs, mean_with_derivatives, 'b:', label=u'Mean with Derivatives Prediction') #  Predicted mean
                plt.fill(np.concatenate([xs, xs[::-1]]),
                         np.concatenate([(mean - 1.9600 * sigma), (mean + 1.9600 * sigma)[::-1]]),
                         alpha=.5,
                         fc='b',
                         ec='None',
                         label='95% confidence interval')
                # plt.fill(np.concatenate([xs, xs[::-1]]),
                #          np.concatenate([(mean - 1.9600 * sigma_with_derivatives), (mean + 1.9600 * sigma_with_derivatives)[::-1]]),
                #          alpha=.1,
                #          fc='b',
                #          ec='None',
                #          label='95% confidence interval')
                plt.xlabel('$x$')
                plt.ylabel('$f(x)$')
                plt.ylim(-10, 20)
                plt.xticks(np.arange(0,10,1))
                plt.yticks(np.arange(-5,15,1))
                plt.grid()
                plt.legend(loc='upper left')
                plt.show()

        def RBF_Kernel(self, _x, _y):

                kernel = 2*np.exp(-0.5*np.square(_x-_y))

                return kernel

        def RBF_Kernel_onederivative(self, _x, _y):
                '''
                See 'GPDerivative_Solak.pdf' for information on the derivatives of the RBF-Kernel
                :param _x: 
                :param _y: 
                :return: 
                '''

                derivative_part = -(_x-_y)
                exponential_part = 2*np.exp(-0.5*np.square(_x-_y))

                return np.multiply(derivative_part, exponential_part)

        def RBF_Kernel_twoderivative(self, _x, _y):
                '''
                See 'GPDerivative_Solak.pdf' for information on the derivatives of the RBF-Kernel
                :param _x: 
                :param _y: 
                :return: 
                '''

                derivative_part = -np.square((_x-_y))
                exponential_part = 2*np.exp(-0.5*np.square(_x-_y))

                return np.multiply(derivative_part, exponential_part)

        def invert_matrix(self, _matrix, _noise=1e-5):

                return np.linalg.inv(_matrix + _noise*np.eye(_matrix.shape[0]))



def SKLearn_GP():
        x = np.linspace(0,10,1000).reshape((-1,1))
        kernel = RBF(10, (0.01, 100)) #  Defining the kernel
        sklearn_gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, alpha=1e-10, optimizer=None) #  Initializing the GP regressor
        sklearn_obs = np.array([7., 1.]).reshape((-1,1)) #  Setting a first obs as the start for the search

        sklearn_gp.fit(sklearn_obs, f(sklearn_obs).ravel())
        y_pred, sigma = sklearn_gp.predict(x, return_std=True)

        fig = plt.figure(figsize=(10,10))
        plt.plot(x, f(x), 'r:', label=u'$f(x) = x\,\sin(x)$') #  The true function
        plt.plot(sklearn_obs, f(sklearn_obs).ravel(), 'r.', markersize=10, label=u'Observations') #  Observations so far
        # plt.plot(_x_pred, _utility, 'b:', label=u'utility') #  Utility function: predicted_mean - currently_best_value + 1.96*standard_deviation
        # plt.plot([np.argmax(utility)/100], [utility[np.argmax(utility)]], 'r*', markersize=10) #  New predicted max which will be evaluated
        plt.plot(x, y_pred, 'b-', label=u'Prediction') #  Predicted mean
        plt.fill(np.concatenate([x, x[::-1]]),
                 np.concatenate([y_pred - 1.9600 * sigma, (y_pred + 1.9600 * sigma)[::-1]]),
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
        plt.title('Scikit-Learn')



# x = np.array([[1, 7]]).T
# y = np.array([[10,12,13]]).T
# print(x)
# print(y)
# print('My rbf kernel')
# print(np.exp(-(np.square(x-x.T))))
# print(sklearn.metrics.pairwise.rbf_kernel(x,x))
# print(np.exp(-(np.square(x-y.T))))
# print('sklearn rbf kernel')
# print(sklearn.metrics.pairwise.rbf_kernel(x,y))
# print('difference is')
# print(np.exp(-(np.square(x-y.T))) - sklearn.metrics.pairwise.rbf_kernel(x,y))

# exit()

X = np.array([[ 3., 5.]]).T
dfX = df(X)

gp = GP(X, f, df)
gp.plot()

# SKLearn_GP()
# plt.show()
