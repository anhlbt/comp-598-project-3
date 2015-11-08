
import os, time, platform, random
from constants import *

def neuronize(Y):
    #convert Y=[[2],[0],[1]...] to Y=[[0,0,1],[1,0,0],[0,1,0]...], or does nothing if Y=[[1],[0] ...]
    highest=-1
    for i in Y:
        highest=i[0] if i[0]>highest else highest
    if highest==1:
        return Y
    
    new_y=[[1 if i[0]==j else 0 for j in range(highest+1)] for i in Y]
    return new_y

def get_data(csv_name,validation_ratio,has_header=True):
    if validation_ratio<0 or validation_ratio>1:
        raise ValueError("bad validation ratio")
    X,Y=[],[]
    with open(csv_name,"r") as f:
        lines= f.readlines()
    if has_header:
        lines=lines[1:]
    random.shuffle(lines)
    X=[[float(i) for i in line.strip().split(",")[1:]] for line in lines]
    Y=[[float(line[0])] for line in lines]
    
    count=round(validation_ratio*len(X))
    X_train=X[:count]
    Y_train=Y[:count]
    X_valid=X[count:]
    Y_valid=Y[count:]

    return X_train,Y_train,X_valid,Y_valid

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

