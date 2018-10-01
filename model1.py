# -*- coding: utf-8 -*-
"""
Created on Mon Oct 01 14:28:20 2018

@author: lenovo
"""
import pandas as pd
import jieba
import jieba.analyse
import random

class chooseSub:
    def __init__(self,train_X,train_y,test_X):
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.subjects = ['动力','价格','内饰','配置','安全性','外观','操控','油耗','空间','舒适性']
        
    
    
    def getMetricResult(self,y_test,metrics='acc'):
        if metrics == 'acc':
            c = 0
            for i in range(len(y_test)):
                if self.y_pred[i] == y_test[i]:
                    c += 1
            acc = float(c) / len(y_test)
            print 'acc is ...',acc
    
    def get_tags(self,text,method='tfidf'):
        if method == 'tfidf':
            return " ".join(jieba.analyse.extract_tags(text))
        if method == 'pagerank':
            return " ".join(jieba.analyse.textrank(text))
        if method == 'cut':
            return " ".join(jieba.cut(text))
        if method == 'cut_all':
            return " ".join(jieba.cut(text,cut_all=True))
    
    def choose(self,arr,adopt = 0):
        '''
        模拟刚刚说的投票机制
        arr: tfidf_features[i] like object
        '''
        temp_dic = {}
        for line in arr:
            for item in line[1]:
                if item[0] not in temp_dic:
                    temp_dic[item[0]] = item[-1]
                else:
                    temp_dic[item[0]] += item[-1]
        #倒排
        temp_dic = sorted(temp_dic.items(),key = lambda item:item[1],reverse=True)
        #选择需要的特征值
        try:
            subject = temp_dic[adopt][0]
        except:
            subject = 0
        return subject

    def choose2(self,arr,adopt = 0):
        '''
        模拟刚刚说的投票机制
        arr: tfidf_features[i] like object
        '''
        temp_dic = {}
        for line in arr:
            for item in line[1]:
                if item[0] not in temp_dic:
                    temp_dic[item[0]] = [1,item[-1]]
                else:
                    temp_dic[item[0]][1] += item[-1]
                    temp_dic[item[0]][0] += 1
        #倒排
        temp_dic = sorted(temp_dic.items(),key = lambda item:item[1],reverse=True)
        #选择需要的特征值
        try:
            subject = temp_dic[adopt][0]
        except:
            subject = 0
        return subject
    
    def train(self,method='tfidf',count=50):
        self.train_X['words'] = self.train_X['content'].apply(self.get_tags,args=(method,))
        
        words_dic = {}
        for word in self.subjects:
            words_dic[word] = []
            
        words = self.train_X['words'].values
        for i in range(len(words)):
            li = words[i].split()
            subject = self.train_y[i]
            for word in li:
                words_dic[subject].append(word)
        
        words_dic_count = {}
        for word in self.subjects:
            words_dic_count[word] = {}
            for value in words_dic[word]:
                if value not in words_dic_count[word]:
                    words_dic_count[word][value] = 1
                else:
                    words_dic_count[word][value] += 1
                    
        for key in words_dic_count:
            words_dic_count[key] = sorted(words_dic_count[key].items(),key = lambda item:item[1],reverse=True)

        subjects_features = []
        for key in words_dic_count:
            try:
                temp = words_dic_count[key][:count]
            except:
                print 'count index error ',count,'out of ',len(words_dic_count[key])
                temp = words_dic_count[key][:]
            subjects_features.append([key,temp])
            
        subjects_dic = {}
        for item in self.train_y:
            if item not in subjects_dic:
                subjects_dic[item] = 1
            else:
                subjects_dic[item] += 1
    
        features_long = {}
        for line in subjects_features:
            for item in line[1]:
                if item[0] not in features_long:
                    features_long[item[0]] = [[line[0],item[1]]]
                else:
                    features_long[item[0]].append([line[0],item[1]])
        
        for key in features_long:
            temp = features_long[key]
            for j in range(len(temp)):
                count = subjects_dic[temp[j][0]]
                temp[j].append(float(temp[j][1])/count)
                
        return features_long
    
    def test(self,features_long,method='tfidf'):
        self.test_X['words'] = self.test_X['content'].apply(self.get_tags,args=(method,))
        content_idt = self.test_X.content_id.values
        id_dict= {}
        for content in content_idt:
            if content not in id_dict:
                id_dict[content] = 1
            else:
                id_dict[content] += 1
        
        words = self.test_X['words'].values
        words_features = []
        for i in range(len(words)):
            li = words[i].split()
            temp = []
            for word in li:
                if word in features_long:
                    temp.append([word,features_long[word]])
            words_features.append(temp)
            
        final_test = []
        reported = {}
        for i in range(len(content_idt)):
            count = id_dict[content_idt[i]]
            if count == 1:
                temp_subject = self.choose(words_features[i],0)
            elif count > 1:
                if content_idt[i] not in reported:
                    reported[content_idt[i]] = 0
                else:
                    reported[content_idt[i]] += 1
                temp_subject = self.choose(words_features[i],reported[content_idt[i]])
            final_test.append(temp_subject)
        
        #给为0的项随机分配一个主题
        count = 0
        for i in range(len(final_test)):
            if final_test[i] == 0:
                final_test[i]=random.choice(self.subjects)
                count += 1
        print 'value 0 ',count
        
        self.y_pred = final_test
        return final_test
    

def main():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test_public.csv')
    choose = chooseSub(train[['content_id','content']],train['subject'].values,test[['content_id','content']])
    model = choose.train(method='tfidf',count=1000)
    y_pred = choose.test(model,method='cut_all')
    #y_test = test.subject.values
    #choose.getMetricResult(y_test)
    return y_pred

def submit(y,name):
    test = pd.read_csv('test_public.csv')
    test['subject'] = y
    test['sentiment_value'] = 0
    test['sentiment_word'] = ""
    new = test.drop(['content'],axis=1)
    new.to_csv(name,index=False)
    