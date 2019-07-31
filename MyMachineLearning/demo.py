import numpy as np
from sklearn.preprocessing import LabelBinarizer
from MyMachineLearning.Dataset import LabeledDatasetFromFile

#激活函数tanh
def tanh(x):
    return np.tanh(x)
#tanh的导函数，为反向传播做准备
def tanh_deriv(x):
    return 1-np.tanh(x)*np.tanh(x)
#激活函数逻辑斯底回归函数
def logistic(x):
    return 1/(1+np.exp(-x))
#激活函数logistic导函数
def logistic_deriv(x):
    return logistic(x)*(1-logistic(x))
#神经网络类
class NeuralNetwork:
    def __init__(self, layers, activation='tanh'):
    #根据激活函数不同，设置不同的激活函数和其导函数
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_deriv
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv
       #初始化权重向量，从第一层开始初始化前一层和后一层的权重向量
        self.weights = []
        for i in range(1, len(layers)-1):
         #权重的shape，是当前层和前一层的节点数目加１组成的元组
            self.weights.append((2*np.random.random((layers[i-1]+1,layers[i]+1))-1)*0.25)
            #权重的shape，是当前层加１和后一层组成的元组
            self.weights.append((2*np.random.random((layers[i]+1,layers[i+1]))-1)*0.25)
    #fit函数对元素进行训练找出合适的权重，X表示输入向量，y表示样本标签，learning_rate表示学习率
    #epochs表示循环训练次数
    def fit(self , X , y , learning_rate=0.2 , epochs=10000):
        X  = np.atleast_2d(X)#保证X是二维矩阵
        temp = np.ones([X.shape[0],X.shape[1]+1])
        temp[:,0:-1] = X
        X = temp #以上三步表示给Ｘ多加一列值为１
        y = np.array(y)#将y转换成np中array的形式
        #进行训练
        for k in range(epochs):
            i = np.random.randint(X.shape[0])#从0-epochs任意挑选一行
            a = [X[i]]#将其转换为list
            #前向传播
            for l in range(len(self.weights)):
                a.append(self.activation(np.dot(a[l],self.weights[l])))
            #计算误差
            error = y[i] - a[-1]
            deltas = [error * self.activation_deriv(a[-1])]
            #反向传播，不包括输出层
            for l in range(len(a)-2,0,-1):
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_deriv(a[l]))
            deltas.reverse()
            #更新权重
            for i in range(len(self.weights)):
                layer  = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate*layer.T.dot(delta)

    #进行预测
    def predict(self,x):
        x = np.array(x)
        temp = np.ones(x.shape[0]+1)
        temp[0:-1] = x
        a = temp
        for l in range(0,len(self.weights)):
            a = self.activation(np.dot(a,self.weights[l]))
        return a


if __name__ == '__main__':
    nn = NeuralNetwork([2, 5, 3, 2], 'tanh')
    data_address = r'D:\Project\Github\LearningMachineLearning\dataset\demodata.xls'
    train_data = LabeledDatasetFromFile(data_address).get_data_by_sheet(0)
    x_train = train_data[:100, :-1]
    y_train = train_data[:100, -1]

    x_test = train_data[100:, :-1]
    y_test = train_data[100:, -1]

    labels_train = LabelBinarizer().fit_transform(y_train)
    labels_test = LabelBinarizer().fit_transform(y_test)

    nn.fit(x_train, labels_train, epochs=2000)
    correct = 0
    for i in range(x_test.shape[0]):
        judge = nn.predict(x_test[i])
        correct += (y_test == np.argmax(judge))

    print(correct / x_test.shape[0])
