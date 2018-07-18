# coding=utf-8
from multiprocessing.dummy import Pool

__author__ = 'jellyzhang'
'''
预测新的script到结果队列中
'''
import collections
from sklearn.datasets import load_svmlight_file
from sklearn.externals import joblib
import time
from collections import OrderedDict
import tensorflow as tf
from tensorflow.python.framework import graph_util
import json
import pika
import requests
import json
import os
import os.path
import re
from preprocessing import  *
from predict_textrnn import predict
import  hashlib
import pickle
from config import *
import logging


logging.basicConfig(level=logging.WARNING,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename='predict.log',
                filemode='w')
readPath=os.path.join(os.path.split(os.path.realpath(__file__))[0],'tmp')
process_number=3
version=1
#加载固化模型
def load_pb(frozen_graph_filename):
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(frozen_graph_filename, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")
            # We load the graph_def in the default graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            output_graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None
        )
    return graph

#获取文件时间戳
def Get_timeprefix_file(filename):
    statinfo = os.stat(filename)
    return statinfo.st_mtime
#内存中加载模型
rnn_graph = load_pb(getConfig('predict-params','model_path'))
#获取文件时间戳
timeprefix=Get_timeprefix_file(getConfig('predict-params','model_path'))
#加载字典索引
with open('vocab_to_int.pkl', 'rb') as pickle_file:
    vocab_to_int = pickle.load(pickle_file)

'''
判断是否为色情url算法
    根据title  prob>=8则porn=1
    如果title<0.8 则判断body中porn所占比例是否达到40%(prob>=0.8) porn=1
    否则porn=0
'''
def detect_porn(title_prob,body_detect_counts,body_detect_porn_counts):
    threshold=float(getConfig('predict-params','threshold'))
    body_rate=float(getConfig('predict-params','body_rate'))
    ret_porn = 0
    if title_prob >= threshold:
        ret_porn = 1
    elif body_detect_counts>0:
        if body_detect_porn_counts*1.0/body_detect_counts>=body_rate:
            ret_porn=1
        else:
            ret_porn=0
    return  ret_porn



def readFile(filename):
                file=os.path.join(readPath,filename)
                print(file)
                #filename = os.path.split(file)[1]
                obj = json.load(open(file, 'r'))
                retDic=obj.copy()
                predict_result=[]
                if len(obj['predict_items'])>=1:
                    title_porn_prob=0
                    body_detect_counts=0
                    body_detect_porn_counts=0
                    inputs=[]
                    booleans=[]
                    plugins=[]
                    porn_items=[]
                    for item in obj['predict_items']: #plug遍历
                            input=item['input']
                            r1 = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
                            input=re.sub(r1,'',input)
                            input=input.replace(' ','')
                            """
                             长度》10的body 或者有匹配的关键字 或者title的都通过
                            """
                            if (len(input)>=10 and item['plugin']=='body') or ((item['blooean']==1 and len(input)>0) or (item['plugin']=='title')):
                                inputs.append(input)
                                booleans.append(item['blooean'])
                                plugins.append(item['plugin'])


                    #批量预测
                    probs, pclasses = predict(rnn_graph, inputs,vocab_to_int)
                    with  open('corpus_porn.txt','a')as fwrite_porn,open('corpus_unporn.txt','a')as fwrite_unporn:
                        for index,text in enumerate(inputs):
                            result = {}
                            porn_item={}
                            if plugins[index] == 'title':
                                title_porn_prob = probs[index]
                                if text=="":
                                    retDic['title_exist']=0
                                else:
                                    retDic['title_exist']=1
                            else:
                                body_detect_counts += 1
                                if probs[index]>= float(getConfig('predict-params','threshold')):
                                    body_detect_porn_counts += 1
                            result['input'] =text
                            result['boolean']=booleans[index]
                            result['plugin'] = plugins[index]
                            result['prob']=float(probs[index])
                            result['pclass']=pclasses[index]
                            predict_result.append(result)
                            #是否添加待筛选语料
                            # if booleans[index]==0 and probs[index]>=0.8: #no keyword ---->predict porn
                            #     fwrite_unporn.write('{}\n'.format(text))
                            # if booleans[index]==1 and probs[index]<0.75:#keyword ---->predict normal
                            #     fwrite_porn.write('{}\n'.format(text))
                            if probs[index]>=0.8:
                                porn_item['input']=text
                                porn_item['plugin'] = plugins[index]
                                porn_item['prob'] = float(probs[index])
                                porn_items.append(porn_item)
                    #delete porn_items
                    del retDic['predict_items']
                    #body 统计信息
                    retDic['body_detect_counts']=body_detect_counts
                    retDic['body_detect_porn_counts']=body_detect_porn_counts
                    #是否为porn标识
                    retDic['porn']=detect_porn(title_porn_prob,body_detect_counts,body_detect_porn_counts)
                    #预测结果
                    retDic['model_version']='v{}'.format(version)
                    retDic['predict_result']=predict_result
                    retDic['porn_items'] = porn_items

                    #只含有url和porn结果
                    result={}
                    result['url']=retDic['url']
                    result['porn']=retDic['porn']
                    #结果打印到日志文件 
                    logging.warning('url:{}\tporn:{}'.format(result['url'],result['porn']))
                    # 重新发布到mq上
                    connection = pika.BlockingConnection(pika.URLParameters(getConfig('predict-params','rabbituri_to')))
                    channel = connection.channel()
                    channel.publish('amq.direct',getConfig('predict-params','queue_result') ,json.dumps(result, separators=(',', ':')))
                    channel.publish('amq.direct', getConfig('predict-params', 'queue_info'),json.dumps(retDic, separators=(',', ':')))
                os.remove(file)



# def readDir(dir):
#     fileList=os.listdir(dir)
#     if len(fileList)>0:
#         pool=Pool(process_number)
#         pool.map(readFile,fileList)
#         pool.close()
#         pool.join()


def main():
    while (True):
        try:

            fileList = os.listdir(readPath)
            if len(fileList) > 0:
                pool = Pool(process_number)
                pool.map(readFile, fileList)
                pool.close()
                pool.join()
            #引入热更新
            if timeprefix!=Get_timeprefix_file('freeze_model/textrnn.pb'):  #更新内存中全局变量
                rnn_graph = load_pb('freeze_model/textrnn.pb')
                with open('vocab_to_int.pkl', 'rb') as pickle_file:
                    vocab_to_int = pickle.load(pickle_file)
                version=version+1  #添加model  version信息
                time.sleep(1) #模型有更新，预测程序sleep 1s，防止文件没有同时上传成功报错



            #time.sleep(60)
        except Exception as ex:
            print(ex)
            logging.error(ex)
if __name__=='__main__':
    main()