import numpy as np
np.core.arrayprint._line_width = 300
import matplotlib.pyplot as plt
from scipy.stats import norm


def f(x):
        return x*np.sin(x)

def df(x):
        return -(np.sin(x) + x*np.cos(x))

class GPO:
        '''
        X: (samples, features)
        y: (samples,1)
        '''
        def __init__(self, _starting_point=None, _function=None, _dfunction=None):

                self.X = np.array((_starting_point)) #  Observed datapoints (features x samples)
                self.function = _function #  True, deterministic function
                self.dfunction = _dfunction #  Derivative function (e.g. f(x)=x*sin(x) -> df(x) = sin(x) + x*cos(x))
                self.y = self.function(self.X) #  Observed targets (samples x 1)
                self.dy = self.dfunction(self.X)
                self.sigma_kernel = 10
                self.noise = 1e-5

                # Construct kernel matrix with the given observation points
                self.K = self.RBF_Kernel(self.X, self.X.T)
                self.K_inv = self.invert_matrix(self.K, _noise= 1e-10)
                self.K_inv_noise = self.invert_matrix(self.K, _noise= 1e-10)

                # Construct the kernel matrix with the given observation points and their derivatives
                self.K_with_derivatives = self.construct_K_with_derivatives()
                self.K_with_derivatives_inv = self.invert_matrix(self.K_with_derivatives, _noise=1e-10)

                # Values relevant to plotting, xmin, xmax values and resolution of the evaluation points between xmin and xmax
                self.plotting_xmin = 0
                self.plotting_xmax = 10
                self.plotting_resolution = 100
                self.plotting_xs = np.linspace(self.plotting_xmin, self.plotting_xmax, self.plotting_resolution).reshape((-1, 1))

                # Values and variables relevant to the optimization part
                self.mean = None
                self.sigma = None
                self.EI = None


        def construct_K_with_derivatives(self):

                K = self.RBF_Kernel(self.X, self.X.T)
                K_d = self.RBF_Kernel_onederivative(self.X, self.X.T)
                K_dd = self.RBF_Kernel_twoderivative(self.X, self.X.T)

                K_upperhalf = np.concatenate((K, K_d), axis=1)
                K_lowerhalf = np.concatenate((K_d, K_dd), axis=1)

                K_with_derivatives = np.concatenate((K_upperhalf, K_lowerhalf), axis=0)

                return K_with_derivatives

        def search_input_space_for_maximum(self):
                self.predict_with_derivatives()
                optimum = np.max(self.function(self.X))

                Z = (self.mean - optimum)/self.sigma
                EI = (self.mean - optimum) * norm.cdf(Z) + self.sigma * norm.pdf(Z)
                self.EI = EI
                point_with_max_EI = np.array([np.argmax(EI)/(self.plotting_xmax - self.plotting_xmin)]).reshape((1,1))

                # Update internal variables
                self.X = np.concatenate((self.X, point_with_max_EI), axis=0)
                self.y = np.concatenate((self.y, self.function(point_with_max_EI)), axis=0)
                self.dy = np.concatenate((self.dy, self.dfunction(point_with_max_EI)), axis=0)

                self.K_with_derivatives = self.construct_K_with_derivatives()
                self.K_with_derivatives_inv = self.invert_matrix(self.K_with_derivatives, _noise=1e-10)

        def predict(self):

                k = self.RBF_Kernel(self.plotting_xs, self.X.T)
                k_starstar = self.RBF_Kernel(self.plotting_xs, self.plotting_xs.T)

                mean = k.dot(self.K_inv).dot(self.y)
                sigma = np.diag(k_starstar - k.dot(self.K_inv).dot(k.T)).reshape((-1,1))

                return mean, sigma

        def predict_with_derivatives(self):

                k = self.RBF_Kernel(self.plotting_xs, self.X.T)
                k_derivative = self.RBF_Kernel_onederivative(self.plotting_xs, self.X.T)
                k_with_derivatives = np.concatenate((k,k_derivative), axis=1)
                k_starstar = self.RBF_Kernel(self.plotting_xs, self.plotting_xs.T)

                y_with_derivatives = np.concatenate((self.y, self.dy),axis=0)

                mean_with_derivatives = k_with_derivatives.dot(self.K_with_derivatives_inv).dot(y_with_derivatives)
                # print(k_with_derivatives.shape, self.K_with_derivatives_inv.shape, y_with_derivatives.shape, '=', mean_with_derivatives.shape)
                sigma_with_derivatives = np.diag(k_starstar - k_with_derivatives.dot(self.K_with_derivatives_inv).dot(k_with_derivatives.T)).reshape((-1,1))

                self.mean = mean_with_derivatives
                self.sigma = sigma_with_derivatives

        def plot(self):

                # mean, sigma = self.predict(xs)
                # fig = plt.figure(figsize=(10,10))
                # plt.title('GP')
                # plt.plot(xs, self.function(xs), 'r:', label=u'$f(x) = x\,\sin(x)$') #  The true function
                # plt.plot(self.X, self.y, 'r.', markersize=10)#, label=u'Observations') #  Observations so far
                # # plt.plot(_x_pred, _utility, 'b:', label=u'utility') #  Utility function: predicted_mean - currently_best_value + 1.96*standard_deviation
                # # # plt.plot([np.argmax(utility)/100], [utility[np.argmax(utility)]], 'r*', markersize=10) #  New predicted max which will be evaluated
                # plt.plot(xs, mean, 'b-', label=u'Mean Prediction') #  Predicted mean
                # plt.fill(np.concatenate([xs, xs[::-1]]),
                #          np.concatenate([(mean - 1.9600 * sigma), (mean + 1.9600 * sigma)[::-1]]),
                #          alpha=.5,
                #          fc='b',
                #          ec='None',
                #          label='95% confidence interval')
                # plt.xlabel('$x$')
                # plt.ylabel('$f(x)$')
                # plt.ylim(-10, 20)
                # plt.xticks(np.arange(0,10,1))
                # plt.yticks(np.arange(-5,15,1))
                # plt.grid()
                # plt.legend(loc='upper left')


                self.predict_with_derivatives()
                fig = plt.figure(figsize=(10,10))
                plt.title('GP with Derivative Information')
                plt.plot(self.plotting_xs, self.function(self.plotting_xs), 'r:', label=u'$f(x) = x\,\sin(x)$') #  The true function
                plt.plot(self.X, self.y, 'r.', markersize=10)#, label=u'Observations') #  Observations so far
                plt.plot(self.plotting_xs, self.mean, 'b', label=u'Mean with Derivatives Prediction') #  Predicted mean
                plt.fill(np.concatenate([self.plotting_xs, self.plotting_xs[::-1]]),
                         np.concatenate([(self.mean - 1.9600 * self.sigma), (self.mean + 1.9600 * self.sigma)[::-1]]),
                         alpha=.5,
                         fc='b',
                         ec='None',
                         label='95% confidence interval')
                plt.xlabel('$x$')
                plt.ylabel('$f(x)$')
                plt.ylim(-10, 20)
                plt.xticks(np.arange(self.plotting_xmin,self.plotting_xmax,1))
                plt.yticks(np.arange(-5,15,1))
                plt.grid()
                plt.legend(loc='upper left')
                plt.show()

        def plot_with_optimization_info(self):

                fig = plt.figure(figsize=(10,10))
                plt.title('GP with Derivative Information')
                plt.plot(self.plotting_xs, self.function(self.plotting_xs), 'r:', label=u'$f(x) = x\,\sin(x)$') #  The true function
                plt.plot(self.plotting_xs, self.EI, 'g', label=u'$EI$') #  The true function
                plt.plot(self.X[:-1,:], self.y[:-1,:], 'r.', markersize=10)#, label=u'Observations') #  Observations so far
                plt.plot(self.X[-1,:], self.y[-1,:], 'g*', markersize=10)#, label=u'Observations') #  Observations so far
                plt.plot(self.plotting_xs, self.mean, 'b', label=u'Mean with Derivatives Prediction') #  Predicted mean
                plt.fill(np.concatenate([self.plotting_xs, self.plotting_xs[::-1]]),
                         np.concatenate([(self.mean - 1.9600 * self.sigma), (self.mean + 1.9600 * self.sigma)[::-1]]),
                         alpha=.5,
                         fc='b',
                         ec='None',
                         label='95% confidence interval')
                plt.xlabel('$x$')
                plt.ylabel('$f(x)$')
                plt.ylim(-10, 20)
                plt.xticks(np.arange(self.plotting_xmin,self.plotting_xmax,1))
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
                kernel = 2*np.exp(-0.5*np.square(_x-_y))

                return np.multiply(derivative_part, kernel)

        def RBF_Kernel_twoderivative(self, _x, _y):
                '''
                See 'GPDerivative_Solak.pdf' for information on the derivatives of the RBF-Kernel
                :param _x: 
                :param _y: 
                :return: 
                '''

                derivative_part = 1-np.square((_x-_y))
                kernel = 2*np.exp(-0.5*np.square(_x-_y))

                return np.multiply(derivative_part, kernel)

        def invert_matrix(self, _matrix, _noise=1e-5):

                return np.linalg.inv(_matrix + _noise*np.eye(_matrix.shape[0]))



###########################################################

starting_point = np.array([[1]]).reshape((-1, 1))

gpo = GPO(starting_point, f, df)
#gpo.plot()
for _ in range(10):
        gpo.search_input_space_for_maximum()
        gpo.plot_with_optimization_info()

