import configparser
import os


def getConfig(section,key):
    config=configparser.ConfigParser()
    path=os.path.join(os.path.split(os.path.realpath(__file__))[0],'config.ini')
    config.read(path)
    return  config.get(section,key)

