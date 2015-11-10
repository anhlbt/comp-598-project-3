
b="python3 main.py ../../data/data_and_scripts/train_inputs.csv ../../data/data_and_scripts/train_outputs.csv --sizes=2304,{hidden}10 --validate --trials={trials} --learn-rate={lr} --validation-ratio=0.9 --random {normalize} --report --final-learn-rate={flr} --timer=120"

import random, math

def magic(number,power_range,is_int=False):
    r=random.random()*number
    while r<number:
        r=r*10
    while r>number:
        r=r/10
    power=random.randint(-power_range,power_range)
    result=r*10**(power)
    if is_int:
        result=int(result)
    return result

def get_bash():
    return b.format(**get_hyper())

def get_hyper():
    hyper={"trials":random.randint(1,100)*10000,
            "lr":magic(0.01,3),
            "normalize":"--normalize" if random.random()>0.1 else ""}
    hyper["flr"]=hyper["lr"]*random.random()

    retry=1
    hidden=""
    while retry>0.95:
        hidden+="%s,"%(magic(100,1,is_int=1)*100)
        retry=random.random()
    hyper["hidden"]=hidden
    return hyper

def debug():
    all=[]
    for i in range(100):
        m=magic(100,1,is_int=1)*100
        print(m)
        all.append(m)
    print("highest: %s"%max(all))
    print("lowest: %s"%min(all))

if __name__=="__main__":
    lines=[get_bash() for i in range(100)]
    with open("batch_nn.bs","w") as f:
        for line in lines:
            f.write("echo %s\n%s\n\n"%(line,line))
