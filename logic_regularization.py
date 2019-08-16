import numpy as np
import scipy.optimize as op

def load_data(filename):
    '''
    这里我的读取想法是，将文件按行str->list,然后lists->array
    '''
    data = [ ]
    with open(filename,"r") as f :
        line = f.readline()#读取文本文件中的一行
        while line:
            each_line = line.strip().split(',')#将读取出来的一行字符串以“，”为间隔返回一个列表
            data_line = [ float(x) for x in each_line ]#将数列里的元素转换为浮点型
            data_line[2] = int(data_line[2])
            f.readline()#为了跳过两行中空的一行
            data.append(data_line)
            line = f.readline()
    return data#返回生成的列表

def X_polynomial(X1,X2):#求多项式矩阵
    out = np.ones((X1.shape[0], 1))
    for i in np.arange(1, 6 + 1, 1):
        for j in np.arange(0, i + 1, 1):
            temp = X1 ** (i - j) * X2 ** (j)
            out = np.hstack((out, temp))
    return out

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost_function(theta,X,y):
    num_X = X.shape[0]
    theta = theta.reshape(X.shape[1], 1)
    cost = np.sum(np.dot((-y).T, np.log(sigmoid(np.dot(X,theta)))) - np.dot((1 - y).T,\
    np.log(1 - sigmoid(np.dot(X,theta))))) / num_X + 0.5 * 1 * np.dot(theta.T,theta) / num_X
    return cost

def gradient(theta,X,y):
    num_X = X.shape[0]
    theta = theta.reshape(X.shape[1],1)#这里添加reshape一开始我想不明白，本来以为它一开始不就是这个shape吗？为什么还要重新reshape一次，后来发现
                                       #由于在minimize中theta的维度每次迭代都flatten一次，因此前面必须加一个reshape来保证参数向量回归原来运算需要的维度。
    theta = np.dot(X.T,(sigmoid(np.dot(X,theta))-y)) / num_X + 1 * theta / num_X
    return theta.flatten()



def main():
    data = np.array(load_data("ex2data2.txt"))
    X1 = data[:,[0]]
    X2 = data[:,[1]]
    y = data[:,[2]]
    X = X_polynomial(X1,X2)#按题目要求将多项式特征矩阵求出
    theta = np.zeros((X.shape[1], 1)).flatten()
    result = op.minimize(fun=cost_function, x0=theta, args=(X, y), method='BFGS', jac=gradient)
    theta = np.array(result["x"]).reshape(X.shape[1],1)
    X_test = np.array([[0.3, 0.3]])
    print(X_test)
    X_norm = X_polynomial(X_test[:,[0]],X_test[:,[1]])
    predict = sigmoid(X_norm.dot(theta))
    print(predict)

if __name__ == "__main__":
    main()