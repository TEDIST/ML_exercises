'''
程序基本功能实现，后续还可以添加一个matplotlib模块，用J_history绘制一个代价函数关于迭代次数的曲线图以观察它的收敛情况
'''

import numpy as np


def load_data(filename):
    '''
    这里我的读取想法是，将文件按行str->list,然后lists->array
    '''
    data = [ ]
    with open(filename,"r") as f :
        line = f.readline()#读取文本文件中的一行
        while line:
            each_line = line.strip().split(',')#将读取出来的一行字符串以“，”为间隔返回一个列表
            data_line = [ int(x) for x in each_line ]#将数列里的元素转换为整数型
            f.readline()#为了跳过两行中空的一行
            data.append(data_line)
            line = f.readline()
    return data#返回生成的列表

#由于观看数据可以发现每个特征的取值范围的差别还是很大的因此这里定义一个特征缩放函数
def feature_scaling(X):
    mu = np.zeros((1,X.shape[1]))
    x_range = np.zeros((1,X.shape[1]))#生成两个初值均为0的1*2维数组
    mu = np.mean(X, axis=0)#mu里存放各特征的均值
    x_range = np.array([X[:,[0]].max() - X[:,[0]].min() , X[:,[1]].max() - X[:,[1]].min()])
    x_norm = (X - mu) / x_range
    return x_norm,mu,x_range

#代价函数
def compute_cost(X_test,y_test,theta):
    num_X = X_test.shape[0]
    cost = 0.5 * (X_test.dot(theta) - y_test).T.dot(X_test.dot(theta) - y_test) / num_X
    return cost

#梯度下降函数
def gradient_descent(X_test,y_test,theta,alpha = 0.4,iters = 1000):#在这里可以调节学习率和迭代次数
    J_history = []#存储代价函数的每次爹迭代的返回值
    num_X = X_test.shape[0]
    for i in range(iters):
        theta = theta - alpha * X_test.T.dot(X_test.dot(theta) - y_test) / num_X#通过矩阵运算来使得theta值同步变化
        cost = compute_cost(X_test,y_test,theta)
        J_history.append(cost)
    return theta,J_history



def main():
    content = load_data("ex1data2.txt")
    data = np.array(content)
    X = data[:,[0,1]]
    y = data[:,[2]]
    X_norm,mu,X_range = feature_scaling(X)#将所有的特征变量放入特征缩放函数中进行缩放得到一个新的特征矩阵
    one = np.ones((X.shape[0],1))
    X = np.hstack((one, X_norm))#x0都默认为1，所以这里要加上一个X.shape[0]*1维的矩阵
    W = np.zeros((X.shape[1], 1))#将seta全部初始化为0,初始化参数
    print('开始运算')
    theta, J_history = gradient_descent(X, y, W)
    print('预测参数为\n', theta)
    X_t = ([[1650,3]] - mu) / X_range
    X_test = np.hstack((np.ones((1, 1)), X_t))
    predict = X_test.dot(theta)
    print("预测结果为")
    print(predict)

if __name__ == '__main__':
    main()