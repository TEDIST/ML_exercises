from sklearn.datasets import load_digits
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split



def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# sigmoid的倒数函数
def dsigmoid(x):
    return x * (1 - x)


class NeuralNetwork(object):
    def __init__(self, input_size, hidden_size, output_size):#参数初始化，不能全为零
        self.W1 = 0.01 * np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = 0.01 * np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros(output_size)

    def loss(self, X, y, reg=0.3):
        num_train, num_feature = X.shape
        #前向传播
        a1 = X
        a2 = sigmoid(a1.dot(self.W1) + self.b1)
        a3 = sigmoid(a2.dot(self.W2) + self.b2)

        loss = - np.sum(y * np.log(a3) + (1 - y) * np.log((1 - a3))) / num_train
        loss += 0.5 * reg * (np.sum(self.W1 * self.W1) + np.sum(self.W2 * self.W2)) / num_train

        # 后向传播
        error3 = a3 - y
        dW2 = a2.T.dot(error3) + reg * self.W2
        db2 = np.sum(error3, axis=0)

        error2 = error3.dot(self.W2.T) * dsigmoid(a2)
        dW1 = a1.T.dot(error2) + reg * self.W1
        db1 = np.sum(error2, axis=0)

        dW1 /= num_train
        dW2 /= num_train
        db1 /= num_train
        db2 /= num_train

        return loss, dW1, dW2, db1, db2

    # 开始训练，这里用到后面要学习到的mini_batch梯度
    def train(self, X, y,learn_rate=0.1, num_iters=1000):
        batch_size = 150
        num_train = X.shape[0]
        loss_list = []

        for i in range(num_iters):
            batch_index = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[batch_index]
            y_batch = y[batch_index]

            loss, dW1, dW2, db1, db2 = self.loss(X_batch, y_batch)
            loss_list.append(loss)

            # update the weight
            self.W1 += -learn_rate * dW1
            self.W2 += -learn_rate * dW2
            self.b1 += -learn_rate * db1
            self.b2 += -learn_rate * db2


        return loss_list
    # 预测一下，多分类的预测，取得最大值的下标为分类结果
    def predict(self, X_test):
        a2 = sigmoid(X_test.dot(self.W1) + self.b1)
        a3 = sigmoid(a2.dot(self.W2) + self.b2)
        y_pred = np.argmax(a3, axis=1)
        return y_pred

    pass


def main():
    digits = load_digits()
    X = digits.data
    y = digits.target
    X_mean = np.mean(X, axis=0)
    X -= X_mean
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    classify = NeuralNetwork(X.shape[1], 100, 10)
    y_train_label = LabelBinarizer().fit_transform(y_train)
    classify.train(X_train,y_train_label)

    # 查看正确率
    y_pred = classify.predict(X_test)
    print(y_pred)
    accuracy = np.mean(y_pred == y_test)
    print("the accuracy is ", accuracy)


if __name__ == "__main__":
    main()
