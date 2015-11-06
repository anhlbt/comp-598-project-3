
import os, time, platform
from constants import *

def add_filename_prefix_to_path(prefix,source):
    #given prefix="test-" and source="/path/to/thingy.csv", returns "/path/to/test-thingy.csv
    split=source.split(os.sep)
    pieces=split[:-1]+[prefix+split[-1]]
    return os.sep.join(pieces)

class Timer:
    #easy way to show updates every X number of seconds for big jobs.
    def __init__(self,interval):
        self.start_time=time.time()
        self.last_time=self.start_time
        self.interval=interval

    def tick(self,text):
        if time.time()>self.last_time+self.interval:
            self.last_time=time.time()
            print_color(text,COLORS.YELLOW)

    def stop(self,label):
        print_color("%s took %s seconds."%(label,round(time.time()-self.start_time,1)),COLORS.YELLOW)

def print_color(text,color=0,end="\n"):
    if platform.system()!="Linux":
        print(text,end=end)
    prefix=""
    if color:
        prefix+="\033[%sm"%(color-10)

    print(prefix+text+"\033[0m",end=end)

