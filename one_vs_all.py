import numpy as np
from sklearn.datasets import load_digits
import scipy.optimize as op
from sklearn.model_selection import train_test_split

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def out(X,theta):
    return sigmoid(np.dot(X,theta))

def cost_function(theta,*args):
    X,y,reg = args
    m,n = X.shape
    theta = theta.reshape((n,1))
    theta_1 = theta[1:,:]
    h = out(X,theta)
    cost = -1 * np.sum(y * np.log(h) + (1 - y) * np.log((1 - h))) / m  + 0.5 * reg * theta_1.T.dot(theta_1) / m
    return cost

def gradf(theta,*args):
    X, y, reg = args
    m, n = X.shape
    theta = theta.reshape((n, 1))
    theta_1 = theta[1:, :]
    h = out(X, theta)
    grad = X.T.dot((h - y)) / m
    grad[1:,:] += reg * theta_1 / m
    g = grad.ravel()
    return g

def one_vs_all(X,y,reg):
    m,n = X.shape
    thetas = np.zeros((n,10))
    for i in range(10):
        init_theta = np.zeros(n)
        args = (X, y==i, reg)
        result = op.fmin_cg(f=cost_function,x0=init_theta,fprime=gradf,args=args)
        thetas[:,i] = result
    return thetas

def predict(x,thetas):
    h = out(x,thetas)
    a = sigmoid(h)
    pred = np.argmax(a,axis=1) #选出数值最大的下标最为分类
    return pred

def main():
    digits = load_digits()
    X = digits.data
    y = digits.target
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3)
    y_train = y_train.reshape((X_train.shape[0], 1))
    X_train_mean = np.mean(X_train, axis=0)
    X_train -= X_train_mean
    m = X_train.shape[0]
    X_train = np.hstack((np.ones((m, 1)), X_train))
    thetas = one_vs_all(X_train,y_train,1)
    print(thetas)
    y_test = y_test.reshape((X_test.shape[0], 1))
    X_test_mean = np.mean(X_test, axis=0)
    X_test -= X_test_mean
    m = X_test.shape[0]
    X_test = np.hstack((np.ones((m, 1)), X_test))
    y_pred = predict(X_test, thetas)
    print("train accuracy is :", np.mean(y_test.ravel() == y_pred))




if __name__ =="__main__":
    main()
