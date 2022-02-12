import numpy as np
# import pandas as pd
from random import shuffle

class Linearmodel():
    def __init__(self):
        pass
    

    ### 初始化模型参数
    def initialize_params(self, dims):
        # 初始化权重参数为零矩阵
        w = np.zeros((dims, 1))
        # 初始化偏差参数为零
        b = 0
        return w, b
    
    ### 定义模型主体部分
    ### 包括线性回归公式、均方损失和参数偏导三部分
    def linear_loss(self, X, y, w, b):
        num_train = X.shape[0]                  # 训练样本数量
        # num_feature = X.shape[1]                # 训练特征数量
        y_hat = np.dot(X, w) + b                # 线性回归预测输出
        loss = np.sum((y_hat-y)**2)/num_train   # 计算预测输出与实际标签之间的均方损失
        dw = np.dot(X.T, (y_hat-y)) /num_train  # 基于均方损失对权重参数的一阶偏导数
        db = np.sum((y_hat-y)) /num_train       # 基于均方损失对偏差项的一阶偏导数
        return y_hat, loss, dw, db
    
    ### 定义线性回归模型训练过程
    def linear_train(self, X, y, learning_rate=0.01, epochs=10000):
        loss_his = []  # 记录训练损失的空列表
        w, b = self.initialize_params(X.shape[1]) # 初始化模型参数
        # 迭代训练
        for i in range(1, epochs):
            # 计算当前迭代的预测值、损失和梯度
            y_hat, loss, dw, db = self.linear_loss(X, y, w, b)
            # 基于梯度下降的参数更新
            w += -learning_rate * dw
            b += -learning_rate * db
            # 记录当前迭代的损失
            loss_his.append(loss)
            # 每1000次迭代打印当前损失信息
            if i % 10000 == 0:
                print('epoch %d loss %f' % (i, loss))
            # 将当前迭代步优化后的参数保存到字典
            params = {
                'w': w,
                'b': b
            }
            # 将当前迭代步的梯度保存到字典
            grads = {
                'dw': dw,
                'db': db
            }     
        return loss_his, params, grads
    
    
    ### 定义线性回归预测函数
    def predict(self, X, params):
        # 获取模型参数
        w = params['w']
        b = params['b']
        # 预测
        y_pred = np.dot(X, w) + b
        return y_pred
    
    
    ### 定义R2系数函数
    def r2_score(self, y_test, y_pred):
        # 测试标签均值
        y_avg = np.mean(y_test)
        # 总离差平方和
        ss_tot = np.sum((y_test - y_avg)**2)
        # 残差平方和
        ss_res = np.sum((y_test - y_pred)**2)
        # R2计算
        r2 = 1 - (ss_res/ss_tot)
        return r2
    
    def k_fold_cross_validation(self, items, k, randomize=True):
        if randomize:
            items = list(items)
            shuffle(items)
        slices = [items[i::k] for i in range(k)]
        for i in range(k):
            validation = slices[i]
            training = [item
                        for s in slices if s is not validation
                        for item in s]
            training = np.array(training)
            validation = np.array(validation)
            yield training, validation
            
            
if __name__ == "__main__":
    import matplotlib as mpl
    mpl.rcParams['font.sans-serif'] = ["SimHei"]
    mpl.rcParams["axes.unicode_minus"] = False
    
    # import pandas as pd
    from sklearn.datasets import load_diabetes
    from sklearn.utils import shuffle            # 导入sklearn打乱数据函数
    model = Linearmodel()
    diabetes = load_diabetes()                    # 获取diabetes数据集
    data, target = diabetes.data, diabetes.target # 获取输入和标签
    X, y = shuffle(data, target, random_state=13) # 打乱数据集
    offset = int(X.shape[0] * 0.8)                # 按照8/2划分训练集和测试集
    X_train, y_train = X[:offset], y[:offset]     # 训练集
    X_test, y_test = X[offset:], y[offset:]       # 测试集
    y_train = y_train.reshape((-1,1))             # 将训练集改为列向量的形式
    y_test = y_test.reshape((-1,1))               # 将验证集改为列向量的形式
    
    # 打印训练集和测试集维度
    print("X_train's shape: ", X_train.shape)
    print("X_test's shape: ",  X_test.shape)
    print("y_train's shape: ", y_train.shape)
    print("y_test's shape: ",  y_test.shape)
    
    # 线性回归模型训练
    loss_his, params, grads = model.linear_train(X_train, y_train, 0.01, 200000)
    # 打印训练后得到模型参数
    print(params)
    
    # 基于测试集的预测
    y_pred = model.predict(X_test, params)
    # 打印前五个预测值
    print(y_pred[:5])
    # 打印前5个实际值
    print(y_test[:5])
    
    
    # 计算并打印决定系数R2
    print(model.r2_score(y_test, y_pred))
    
    
    import matplotlib.pyplot as plt
    f = X_test.dot(params['w']) + params['b']
    plt.scatter(range(X_test.shape[0]), y_test) # 散点部分，真实值
    plt.plot(f, color = 'darkorange')           # 折线部分，预测值
    plt.xlabel('X_test')                        # X轴没有实际意义
    plt.ylabel('y_test')
    plt.title("预测数据图")
    plt.show()
    
    # 绘制迭代次数与损失函数关系图
    plt.plot(loss_his, color = 'blue')
    plt.xlabel('epochs') # 迭代次数
    plt.ylabel('loss') # 损失函数走势
    plt.title("迭代次数与损失函数关系图")
    plt.show()
    
    
    for training, validation in model.k_fold_cross_validation(data, 5): 
        # 训练集，分离变量和标签
        X_train = training[:, :10]
        y_train = training[:, -1].reshape((-1,1))
        # 测试集，分离变量和标签
        X_valid = validation[:, :10]
        y_valid = validation[:, -1].reshape((-1,1))
        print("查看数据信息：")
        print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape)
        
        loss5 = []
        loss, params, grads = model.linear_train(X_train, y_train, 0.001, 100000) # linear_train拼写错了
        loss5.append(loss)
        score = np.mean(loss5)
        print('five kold cross validation score is', score)  # 5折交叉验证得分
        
        y_pred = model.predict(X_valid, params)
        valid_score = np.sum(((y_pred - y_valid)**2))/len(X_valid)
        print('valid score is', valid_score)
        print()