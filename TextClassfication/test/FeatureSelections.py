__author__ = 'Jason'
# -*- coding: utf-8 -*-

import os
import sys
reload(sys)
sys.setdefaultencoding('utf8')

import numpy as np

# 传递过来的是每个文档的词语，
# 数据类型 doc_term_list: [['this', 'is', 'economy'],['this', 'is', 'culture']]
def get_term_dict(doc_terms_list):
    term_set_dict = {}
    for doc_terms in doc_terms_list:
        for term in doc_terms:
            term_set_dict[term] = 1
    term_set_list = sorted(term_set_dict.keys())       #term set 排序后，按照索引做出字典，中文出错的原因？？？
    term_set_dict = dict(zip(term_set_list, range(len(term_set_list))))
    return term_set_dict

# 根据传递过来的每个文档的类别数组，得到总共的文本类别，进行排序
# 最后得到类别字典，每个字典的key和value是一样的
def get_class_dict(doc_class_list):
    class_set = sorted(list(set(doc_class_list)))
    class_dict = dict(zip(class_set, range(len(class_set))))
    return  class_dict

# 得到每个特征的文档数
def stats_term_df(doc_terms_list, term_dict, term_set):
    term_df_dict = {}.fromkeys(term_dict.keys(), 0)  # 将每个特征字典后面的值设置为0
    for term in term_set:
        for doc_terms in doc_terms_list:
            if term in doc_terms:
                term_df_dict[term] +=1
    return term_df_dict

# 输入所有的类别和形成的类别字典
# 对传过来的每个文档的类别，统计每个类别的文档总数
def stats_class_df(doc_class_list, class_dict):
    class_df_list = [0] * len(class_dict)
    for doc_class in doc_class_list:
        class_df_list[class_dict[doc_class]] += 1
    return class_df_list

# 返回词频矩阵，横轴是类别，竖轴是特征词，元素a[i][j]表示该特征词i在该类别j中出现次数
def stats_term_class_df(doc_terms_list, doc_class_list, term_dict, class_dict):
    # np.zeros() 返回来一个给定形状和类型的用0填充的数组；
    term_class_df_mat = np.zeros((len(term_dict), len(class_dict)), np.float32)
    for k in range(len(doc_class_list)):
        class_index = class_dict[doc_class_list[k]]
        doc_terms = doc_terms_list[k]
        for term in set(doc_terms):
            term_index = term_dict[term]
            term_class_df_mat[term_index][class_index] +=1
    return  term_class_df_mat

# 互信息方法进行特征选择
# 参数解释：
# class_df_list：每个类别的文档数
# term_set：所有的特征词组成的数组,而且已经排序
# term_class_df_mat:词频矩阵，数组
def feature_selection_mi(class_df_list, term_set, term_class_df_mat):
    A = term_class_df_mat
    B = np.array([(sum(x) - x).tolist() for x in A])
    C = np.tile(class_df_list, (A.shape[0], 1)) - A
    N = sum(class_df_list)  # 总文档数
    class_set_size = len(class_df_list)  # 总类别数

    term_score_mat = np.log(((A+1.0)*N) / ((A+C) * (A+B+class_set_size)))
    term_score_max_list = [max(x) for x in term_score_mat]
    term_score_array = np.array(term_score_max_list)
    sorted_term_score_index = term_score_array.argsort()[: : -1]
    term_set_fs = [term_set[index] for index in sorted_term_score_index]
    print "mi term_set_fs length %s" %(len(term_set_fs))
    return term_set_fs

# 改进的mi方法
def feature_selection_mi_2(class_df_list, term_set, term_class_df_mat):
    A = term_class_df_mat
    B = np.array([(sum(x) - x).tolist() for x in A])
    C = np.tile(class_df_list, (A.shape[0], 1)) - A
    N = sum(class_df_list)  # 总文档数
    class_set_size = len(class_df_list)  # 总类别数

    # 调整后的mi，求方差

    term_score_mat = (A/N)*(np.log(((A+1.0)*N) / ((A+C) * (A+B+class_set_size))))
    term_score_max_list = [max(x) for x in term_score_mat]
    term_score_array = np.array(term_score_max_list)
    sorted_term_score_index = term_score_array.argsort()[: : -1]
    term_set_fs = [term_set[index] for index in sorted_term_score_index]
    print "mi term_set_fs length %s" %(len(term_set_fs))
    return term_set_fs

# 参数解释：
# class_df_list：每个类别的文档数
# term_set：所有的特征词组成的数组,而且已经排序
# term_class_df_mat:词频矩阵，数组
def feature_selection_chi(class_df_list, term_set, term_class_df_mat):
    A = term_class_df_mat
    B = np.array([(sum(x) - x).tolist() for x in A])
    C = np.tile(class_df_list, (A.shape[0], 1)) - A
    N = sum(class_df_list)  # 总文档数
    D = N - A - B - C
    class_set_size = len(class_df_list)  # 总类别数

    term_score_mat = ((A*D - B*C)*(A*D - B*C))/((A+B+class_set_size)*(C+D+class_set_size))
    term_score_max_list = [max(x) for x in term_score_mat]
    term_score_array = np.array(term_score_max_list)
    sorted_term_score_index = term_score_array.argsort()[: : -1]
    term_set_fs = [term_set[index] for index in sorted_term_score_index]
    return term_set_fs

# 改进的chi算法
def feature_selection_chi_2(class_df_list, term_set, term_class_df_mat):
    A = term_class_df_mat
    B = np.array([(sum(x) - x).tolist() for x in A])
    C = np.tile(class_df_list, (A.shape[0], 1)) - A
    N = sum(class_df_list)  # 总文档数
    D = N - A - B - C
    class_set_size = len(class_df_list)  # 总类别数

    term_score_mat = ((A*D - B*C)*(A*D - B*C))/((A+B)*(C+D))

    # 引入词频调节因子
    E = np.sum(A,axis=1)
    print "E ", E
    for x in range(len(A)):
        for y in range(len(A[x])):
            A[x][y] /= E[x]
    # print "A 0-2 after ", A[:2]

    term_score_mat = A * term_score_mat
    term_score_max_list = [max(x) for x in term_score_mat]
    term_score_array = np.array(term_score_max_list)
    sorted_term_score_index = term_score_array.argsort()[: : -1]
    term_set_fs = [term_set[index] for index in sorted_term_score_index]
    return term_set_fs

# 混合改进算法
# 联合chi和mi特征选择方法，进一步改进两种方法的缺点
# 引入词频来改进
def feature_selection_chi_mi(class_df_list, term_set, term_class_df_mat):
    # 先进行chi的参数求解
    A = term_class_df_mat
    print "A 0-2", A[:2]
    B = np.array([(sum(x) - x).tolist() for x in A])
    C = np.tile(class_df_list, (A.shape[0], 1)) - A
    N = sum(class_df_list)  # 总文档数
    D = N - A - B - C
    class_set_size = len(class_df_list)  # 总类别数
    # 求得CHI的参数
    term_score_mat_chi = ((A*D - B*C)*(A*D - B*C))/((A+B)*(C+D))

    # 然后求得MI的参数
    term_score_mat_mi = (A / N) * (np.log(((A+1.0)*N) / ((A+C) * (A+B))))

    # 引入词频调节因子
    E = np.sum(A,axis=1)
    print "E ", E
    for x in range(len(A)):
        for y in range(len(A[x])):
            A[x][y] /= E[x]
    # print "A 0-2 after ", A[:2]

    # term_score_mat = A * term_score_mat_chi * term_score_mat_mi
    term_score_mat = term_score_mat_chi

    term_score_max_list = [max(x) for x in term_score_mat]
    term_score_array = np.array(term_score_max_list)
    sorted_term_score_index = term_score_array.argsort()[: : -1]
    term_set_fs = [term_set[index] for index in sorted_term_score_index]
    return term_set_fs


def feature_selection(doc_terms_list, doc_class_list, fs_method):
    class_dict = get_class_dict(doc_class_list)  # 获取类别字典
    term_dict = get_term_dict(doc_terms_list)  # 获取特征词字典
    class_df_list = stats_class_df(doc_class_list, class_dict)  # 得到每个类别的文档数
    term_class_df_mat = stats_term_class_df(doc_terms_list, doc_class_list, term_dict, class_dict)  # 每个特征词在每个类别中出现的次数，词频矩阵
    # 得到特征词数组
    term_set = [term[0] for term in sorted(term_dict.items(), key = lambda x : x[1])]
    print "before selection length %s" %(len(term_set))
    # term_df_dict = stats_term_df(doc_terms_list, term_dict, term_set)
    term_set_fs = []

    # if fs_method == 'MI':
    #     term_set_fs = feature_selection_mi(class_df_list, term_set, term_class_df_mat)
    # elif fs_method == 'CHI':
    #     term_set_fs = feature_selection_chi(class_df_list, term_set, term_class_df_mat)
    # elif fs_method == 'CHMI':
    #     term_set_fs = feature_selection_chi_mi(class_df_list, term_set, term_class_df_mat)

    if fs_method == 'CHI':
        term_set_fs = feature_selection_chi(class_df_list, term_set, term_class_df_mat)
    elif fs_method == 'CHI_2':
        term_set_fs = feature_selection_chi_2(class_df_list, term_set, term_class_df_mat)
    elif fs_method == 'MI':
        term_set_fs = feature_selection_mi(class_df_list, term_set, term_class_df_mat)
    elif fs_method == 'MI_2':
        term_set_fs = feature_selection_mi_2(class_df_list, term_set, term_class_df_mat)
    elif fs_method == 'CHMI':
        term_set_fs = feature_selection_chi_mi(class_df_list, term_set, term_class_df_mat)

    return term_set_fs