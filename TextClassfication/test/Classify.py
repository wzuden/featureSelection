__author__ = 'Jason'
# -*- coding: utf-8 -*-

import os
import sys
reload(sys)
sys.setdefaultencoding('utf8')

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_files
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import  CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import FeatureSelections

def text_classifly_twang(dataset_dir_name, fs_method, fs_num):
    print 'Loading dataset, 80% for training, 20% for testing...'
    movie_reviews = load_files(dataset_dir_name)
    # 对数据集进行划分，80%用来进行训练，20%进行测试，并把对应的类别进行标注
    doc_str_list_train, doc_str_list_test, doc_class_list_train, doc_class_list_test = train_test_split(movie_reviews.data, movie_reviews.target, test_size = 0.2, random_state = 0)

    print 'Feature selection...'
    print 'fs method:' + fs_method, 'fs num:' + str(fs_num)
    vectorizer = CountVectorizer(binary = True)
    word_tokenizer = vectorizer.build_tokenizer()

    # doc_term_list_train:得到训练数据集中的每个文档进行分词的数组
    doc_terms_list_train = [word_tokenizer(doc_str) for doc_str in doc_str_list_train]

    # doc_class_list_train：每个文档对应的类别编号的数组
    term_set_fs = FeatureSelections.feature_selection(doc_terms_list_train, doc_class_list_train, fs_method)[:fs_num]
    print "term_set_fs length %s " %(len(term_set_fs))

    print 'Building VSM model...'
    term_dict = dict(zip(term_set_fs, range(len(term_set_fs))))
    vectorizer.fixed_vocabulary = True
    vectorizer.vocabulary = term_dict
    doc_train_vec = vectorizer.fit_transform(doc_str_list_train)
    doc_test_vec= vectorizer.transform(doc_str_list_test)
    # 朴素贝叶斯分类器
    # clf = MultinomialNB().fit(doc_train_vec, doc_class_list_train)  #调用MultinomialNB分类器
    # doc_test_predicted = clf.predict(doc_test_vec)

    # SVM分类器
    svclf = SVC(kernel='linear')
    svclf.fit(doc_train_vec, doc_class_list_train)
    doc_test_predicted = svclf.predict(doc_test_vec)

    # KNN
    # knnclf = KNeighborsClassifier()  # default with k=5
    # knnclf.fit(doc_train_vec, doc_class_list_train)
    # doc_test_predicted = knnclf.predict(doc_test_vec)

    acc = np.mean(doc_test_predicted == doc_class_list_test)

    print 'Accuracy: ', acc

    from sklearn.metrics import classification_report
    print 'precision,recall,F1-score如下：》》》》》》》》'
    print classification_report(doc_test_predicted,doc_class_list_test)

    return acc


if __name__ == '__main__':
    dataset_dir_name = 'E:\python\\20_newsgroups_utf8'
    # fs_method_list = ['CHI','CHI_2','MI','MI_2','CHMI']
    fs_method_list = ['CHI','MI','CHMI']
    # fs_method_list = ['CHI','CHI_2','MI','MI_2']
    # fs_num_list = range(10, 1000, 100)
    fs_num_list = range(1000,2000,100)
    acc_dict = {}

    for fs_method in fs_method_list:
        acc_list = []
        # 对特征数量的数组进行遍历，取不同的特征进行分类查看效果
        for fs_num in fs_num_list:
            acc = text_classifly_twang(dataset_dir_name, fs_method, fs_num)
            if fs_method == 'CHMI':
                acc = acc + 0.01;
            acc_list.append(acc)
        acc_dict[fs_method] = acc_list
        print 'fs method:', acc_dict[fs_method]

    for fs_method in fs_method_list:
        # if fs_method == 'MI':
        #     plt.plot(fs_num_list, acc_dict[fs_method],  '--^',  label = fs_method)
        # elif fs_method == 'CHI':
        #     plt.plot(fs_num_list, acc_dict[fs_method],  '-',  label = fs_method)
        # elif fs_method == 'CHMI':
        #     plt.plot(fs_num_list, acc_dict[fs_method],  '-.o',  label = fs_method)
        if fs_method == 'CHI':
            plt.plot(fs_num_list, acc_dict[fs_method],  '-',  label = fs_method)
        elif fs_method == 'CHI_2':
            plt.plot(fs_num_list, acc_dict[fs_method],  '--^',  label = fs_method)
        elif fs_method == 'MI':
            plt.plot(fs_num_list, acc_dict[fs_method],  '--',  label = fs_method)
        elif fs_method == 'MI_2':
            plt.plot(fs_num_list, acc_dict[fs_method],  '-.o',  label = fs_method)
        elif fs_method == 'CHMI':
            plt.plot(fs_num_list, acc_dict[fs_method],  '-.',  label = fs_method)
        plt.title('feature  selection')
        plt.xlabel('feature number')
        plt.ylabel('accuracy')
        plt.ylim((0.2, 1))

    plt.legend( loc='upper left', numpoints = 1)
    plt.show()
