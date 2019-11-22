#!/usr/bin/env python
# coding: utf-8

# In[81]:


import numpy as np
import pandas as pd
from collections import Counter

#树节点类，分为叶子节点，非叶子节点
class TreeNode(object):
    def __init__(self,is_leaf=False,score=None,
               feature=None,threshold=None,
               left=None,right=None):
        self.is_leaf=is_leaf
        self.score=score
        self.feature=feature
        self.threshold=threshold
        self.left=left
        self.right=right


# In[83]:


#测试集只有单条的预测
def predict_single(treenode,test):
    if treenode.is_leaf:
         return treenode.score
    else:
        #测试集当前节点特征的值小于域值，向训练好的treenode左子树查找
        if test[treenode.feature]<treenode.threshold:
            return predict_single(treenode.left,test)
        else:
            return predict_single(treenode.right,test)


# In[84]:


#测试集多条特征预测
def predict(treenode,test):
    result = []
    for i in test:
        result.append(predict_single(treenode,i))
    return result


# In[116]:


#决策树类，基于Top-Down思路，
#predict->construct_tree->calculate_score->find_best_feature_threshold_and_gain
class Tree(object):
    #estimator为训练好的决策树
    def __init__(self,estimator=None,max_depth=3,
               min_sample_split=10,gamma=0):
        self.estimator=estimator
        self.max_depth=max_depth
        self.min_sample_split=min_sample_split
        self.gamma=gamma
        
    #测试集只有单条的预测
    def predict_single(self,treenode,test):
        if treenode.is_leaf:
            return treenode.score
        else:
            print(' return self.predict_single(treenode.left,test)')
            #测试集当前节点特征的值小于域值，向训练好的treenode左子树查找
            if test[treenode.feature]<treenode.threshold:
                return self.predict_single(treenode.left,test)
            else:
                return self.predict_single(treenode.right,test)
            
    #测试集多条特征预测
    def predict(self,test):
        result = []
        for i in test:
            result.append(predict_single(self.estimator,i))
        return result
    
    #构建，construct_tree方法构建实例变量训练决策树estimator
    def fit(self,train):
        self.estimator = self.construct_tree(train,label,depth_left = self.max_depth)
        
    def construct_tree(self,train,label,depth_left):
        #决策树终止条件,剩余树深=0或者节点中样本数<self.min_sample_split
        if depth_left==0 or len(train) < self.min_sample_split:
            return TreeNode(is_leaf=True,score=self.calculate_class(label))
        
        feature,threshold,gain = self.find_best_feature_threshold_and_gain(train,label)
        #增益<=最大增益
        if gain<=self.gamma:
            return TreeNode(is_leaf=True,score=self.calculate_class(label))
        
        index = train[:,feature]<threshold
        left = self.construct_tree(train[index],label[index],depth_left-1)
        right = self.construct_tree(train[~index],label[~index],depth_left-1)
        return TreeNode(feature=feature,threshold=threshold,left=left,right=right)
    
    #输入训练数据和标签，返回最佳分裂特征、最佳分裂点、增益
    def find_best_feature_threshold_and_gain(self,train,label):
        best_feature = None
        best_threshold = None
        best_gain = 0 
        for feature in range(train.shape[1]):
            print('feature:' + str(feature))
            threshold,gain = self.find_best_threshold_and_gain(train[:,feature],label)
            print(gain)
            if gain > best_gain:
                best_feature = feature
                best_threshold = threshold
                best_gain = gain
            print('best_feature:' + str(best_feature))
            print('best_threshold:' + str(best_threshold))
            print('best_gain:' + str(best_gain))
            print('--------------------------')
        return best_feature,best_threshold,best_gain
    
    #输入一个特征和对应标签，返回在这个特征上的最佳分裂点、增益
    def find_best_threshold_and_gain(self,feature_array,label):
        
        original_loss = self.calculate_cross_entropy(label)
        best_threshold = None
        best_gain = 0
        #去重并排序
        sorted_feature_values = np.unique(feature_array)
        #计算所有可能分裂点
        for i in range(1,len(sorted_feature_values)):
            threshold = (sorted_feature_values[i-1] + sorted_feature_values[i])/2
            index = feature_array < threshold
            left_loss = self.calculate_cross_entropy(label[index])
            right_loss = self.calculate_cross_entropy(label[~index])
            gain = original_loss - left_loss - right_loss
            print('threshold:' + str(threshold))
            print('gain=original_loss-left_loss-right_loss='+
                  str(original_loss)+'-'+str(left_loss)+'-'+str(right_loss)+
                  '='+str(gain))
            if gain > best_gain:
                best_threshold = threshold
                best_gain = gain
            print('best_gain:' + str(best_gain))
            print('----------------------------------')    
        return best_threshold,best_gain
    
    #回归树:输入标签，返回损失函数的值,最小二乘法
    def calculate_loss(self,label):
        print(label)
        return np.sum(np.square(label-self.calculate_score(label)))
    
    #回归树:输入标签，返回预测值，平均值
    def calculate_score(self,label):
        #取平均值作为结果
        return np.mean(label)
    

    #分类树:输入标签，返回真实值(0,1,0),以及预测值概率(0.4,0.5,0.1)
    def realClass(self,label):
        class_set = np.unique(label)
        count = np.array([0]*class_set.size)
        #计算每种类别数量
        for i in label:
            for j in class_set:
                if i == j:
                    index = np.where(class_set == j)[0]
                    count[index] += 1      
        real_value = np.max(count)
        real_list = np.array([0]*class_set.size)
        index = count == np.max(count)
        probability = count/np.sum(count)
        return index.astype(int),probability

    #分类树：输入标签，输出预测值
    def calculate_class(self,label):
        class_set = np.unique(label)
        count = np.array([0]*class_set.size)
        #计算每种类别数量
        for i in label:
            for j in class_set:
                if i == j:
                    index = np.where(class_set == j)[0]
                    count[index] += 1 
        print(np.where(count == np.max(count)))
        return class_set[np.where(count == np.max(count))][0]

    #分类树：输入标签，输出交叉熵和
    def calculate_cross_entropy(self,label):
        print(label)
        true_value,predict_value = realClass(label)
        if true_value.size < 2:
            return 0
        a = np.log10(predict_value)*true_value 
        b = np.log10(1-predict_value)*(1-true_value)
        return np.sum(-(a+b))




# In[105]:


#分类树:输入标签，返回真实值(0,1,0),以及预测值概率(0.4,0.5,0.1)
def realClass(label):
    class_set = np.unique(label)
    count = np.array([0]*class_set.size)
    #计算每种类别数量
    for i in label:
        for j in class_set:
            if i == j:
                index = np.where(class_set == j)[0]
                count[index] += 1      
    real_value = np.max(count)
    real_list = np.array([0]*class_set.size)
    index = count == np.max(count)
    probability = count/np.sum(count)
    return index.astype(int),probability

#输入标签，输出预测值
def calculate_class(label):
    class_set = np.unique(label)
    count = np.array([0]*class_set.size)
    #计算每种类别数量
    for i in label:
        for j in class_set:
            if i == j:
                index = np.where(class_set == j)[0]
                count[index] += 1 
    print(np.where(count == np.max(count)))
    return class_set[np.where(count == np.max(count))][0]

#输入标签，输出交叉熵和
def calculate_cross_entropy(label):
    true_value,predict_value = realClass(label)
    if true_value.size < 2:
        return 0
    a = np.log10(predict_value)*true_value 
    b = np.log10(1-predict_value)*(1-true_value)
    return np.sum(-(a+b))


# In[106]:




# In[128]:

