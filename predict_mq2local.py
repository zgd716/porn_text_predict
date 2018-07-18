# coding=utf-8
__author__ = 'jellyzhang'

import os
import pika
import json
import time
import os
from multiprocessing.dummy import Process
from datetime import datetime
import re
import logging
from logging.handlers import RotatingFileHandler
from logging import Formatter
import sys
from config import *
#本地可存储最大数量
max_store_number=int(getConfig('predict-params','max_store_localfiles'))

domaincache={}
reip=re.compile(r'^(?:[12]?\d?\d\.){3}[12]?\d?\d$')
curdir,pyfile=os.path.split(sys.argv[0])
def main():
    workers=[]
    workers.append(Process(target=find_data,args=(getConfig('predict-params','rabbituri_from'),getConfig('predict-params','queue_from'))))
    for p in workers:
        p.start()
    for p in workers:
        p.join()





def find_data(rabbituri,src):
    lasttime=datetime.now()
    while(True):
        try:
            if len(os.listdir('tmp'))>max_store_number:
                time.sleep(60)
            else:
                print('init rabbitmq connection...[{0}]'.format(rabbituri))
                conn = pika.BlockingConnection(pika.URLParameters(rabbituri))
                channel = conn.channel()
                while(True):
                    if len(os.listdir('tmp')) > max_store_number:
                        time.sleep(60)
                    else:
                        (getok,properties,body)=channel.basic_get(src,no_ack=False)
                        if not body:
                            print('no tasks...[{0}]'.format(src))
                            if (datetime.now()-lasttime).total_seconds()>600:
                                lasttime=datetime.now()
                                conn.close()
                                break
                            time.sleep(5)
                            continue
                        lasttime=datetime.now()
                        obj=None
                        try:
                            obj = json.loads(body.decode())
                            #obj=json.loads(body.decode(errors='ignore'),encoding='utf-8')
                            #print(str(obj))
                        except Exception as e:
                            print(e)
                            print('invalid msg,{0}'.format(body.decode()))
                        if obj:  #处理数据逻辑
                            temp={}
                            #if 'result' in obj:
                            #保存到本地，避免文件太大时，mq连接断开
                            #if  obj['result']['score']!=0:
                            path = os.path.split(os.path.realpath(__file__))[0]
                            filename = os.path.join(os.path.join(path, 'tmp'),
                                                             time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))+ '.txt')
                            with open(filename, 'w') as fwrite:
                                            fwrite.write(json.dumps(obj))
                                    #time.sleep(3)
                            channel.basic_ack(getok.delivery_tag)  #队列消息确认操作，测试时可暂时注释掉

        except Exception as e:
            print(e)
            time.sleep(30)


if __name__=='__main__':
    main()