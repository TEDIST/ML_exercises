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
            data_line = [ float(x) for x in each_line ]#将数列里的元素转换为浮点型
            data_line[2] = int(data_line[2])
            f.readline()#为了跳过两行中空的一行
            data.append(data_line)
            line = f.readline()
    return data#返回生成的列表


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost(theta, X, y):
    num_X = X.shape[0]
    cost = np.sum(np.multiply(-y,np.log(sigmoid(X.dot(theta)))) - np.multiply((1 - y),np.log(1 - sigmoid(X.dot(theta)))))/num_X
    return cost

def gradient(theta,X,y,alpha = 0.004,iters = 1000000):#这里经过多次测试这个迭代次数下得到的参数去预测才比较负荷，但这时每预测一次都要跑30秒左右，小本子不堪重负
    J_history = []
    num_X = X.shape[0]
    for i in range(iters):
        theta = theta - alpha * X.T.dot(sigmoid(X.dot(theta)) - y) / num_X
        cost_J = cost(theta,X,y)
        J_history.append(cost_J)
    return theta,J_history


def main():
    content = load_data("ex2data1.txt")
    data = np.array(content)
    X = data[:,[0,1]]
    y = data[:,[2]]
    one = np.ones((X.shape[0], 1))
    X = np.hstack((one, X))  # x0都默认为1，所以这里要加上一个X.shape[0]*1维的矩阵
    theta = np.zeros((X.shape[1], 1))  # 将theta全部初始化为0,初始化参数
    print("开始运算")
    theta,J_history = gradient(theta,X,y)
    print('预测参数为\n', theta)
    X_test = np.array([1,34,78])
    predict = sigmoid(X_test.dot(theta))
    print(predict)



if __name__ == "__main__":
    main()