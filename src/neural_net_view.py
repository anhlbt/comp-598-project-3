
import os
from utilities import *
from constants import *

class NeuralNetView:  
    def setup_logging(self):
        names=("weights","outputs","activations","corrections")
        self.log_folder="logs"
        if self.logging:
            try:
                os.mkdir(self.log_folder)
            except OSError:
                pass
            try:
                for name in names:
                    os.remove(self.log_folder+os.sep+name+"-log.csv")
            except FileNotFoundError:
                pass 
            
    def show(self,weights=False,outputs=False,activations=False,corrections=False,all=False):
        #this is a convenient way to show some or all of the NN info
        def show_np_list(label,np_list):
            print_color("%s:"%label,COLORS.YELLOW)
            for i,item in enumerate(np_list):
                s="" if type(item) is int else str(item.shape)
                print("%s:"%i,s,item)
        
        if weights or all:
            show_np_list("weights",self.weights)
        if outputs or all:
            show_np_list("outputs",self.outputs)
        if activations or all:
            show_np_list("activations",self.activations)
        if corrections or all:
            show_np_list("corrections",self.corrections)

    def log(self):
        #appends the current state of the NN to log files
        items={"weights":self.weights,
                "outputs":self.outputs,
                "activations":self.activations,
                "corrections":self.corrections}
        for key in items:
            text=[]
            item=items[key]
            for w in item:
                if type(w) is int:
                    text+=[w]
                else:
                    text+=w.flatten().tolist()
            text=",".join([str(round(i,3)) for i in text])
            with open(self.log_folder+os.sep+key+".csv","a") as f:
                f.write("\n"+text)


    def get_report(self,X,Y):
        #gets predictions for all of X, compares to Y, returns a report
        errors=[]
        error_squared=0
        results={}
        success_count=0
        for i in range(len(X)):
            self.forward(X[i])

            output=self.get_output()[0]
            result=1 if output>0.5 else 0
            if result in results:
                results[result]+=1
            else:
                results[result]=1
            expected=Y[i][0]
            error_squared+=(output-expected)**2
            if result == expected:
                success_count+=1
            else:
                errors.append("result=%s expected=%s case=%s"%(result,expected,str(X[i])))
        
        report={"errors":errors,
                "error squared":error_squared,
                "results":results,
                "success count":success_count}
        return report

    def show_report(self,X,Y):
        report=self.get_report(X,Y)
        keys=sorted(list(report.keys()))
        for key in keys:
            print_color(key.upper(),COLORS.GREEN)
            data=report[key]
            if type(data) is list:
                print("    ",end="")
                print_color("\n    ".join(data),COLORS.ORANGE)
            else:
                print_color(str(data),COLORS.YELLOW)

