
import os
from utilities import *
from constants import *

class NeuralNetView:  
    def setup_logging(self):
        names=("weights","outputs","activations","corrections")
        if self.logging:
            try:
                os.mkdir(LOGFOLDER)
            except OSError:
                pass
            try:
                for name in names:
                    os.remove(self.get_log_path(name))
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
            digits=20 if key in ("corrections","weights") else 3
            text=",".join([str(round(i,digits)) for i in text])
            with open(self.get_log_path(key),"a") as f:
                f.write("\n"+text)

    def get_log_path(self,label):
        return LOGFOLDER+label+"-log.csv"

    def get_prediction(self,output):
        #if output is just one neuron, then return 0 or 1 based on its weight
        #if output is more than one neuron, prediction is the index of the highest activated neuron
        if len(output)==1:
            result=1 if output>0.5 else 0
        else:
            result=output.tolist().index(max(output))
        return result

    def get_error_report(self,label,X,Y):
        #gets predictions for all of X, compares to Y
        errors=[]
        predictions={}
        success_count=0
        for i in range(len(X)):
            self.forward(X[i])
            output=self.get_output()
            prediction=self.get_prediction(output)
            if prediction in predictions:
                predictions[prediction]+=1
            else:
                predictions[prediction]=1
            expected=Y[i][0]

            if prediction == expected:
                success_count+=1
            else:
                errors.append("prediction=%s expected=%s case=%s"%(prediction,expected,str(X[i])))

        accuracy=success_count/len(X)
        
        return {label+" errors":errors,
                label+" predictions":predictions,
                label+" success count":success_count,
                label+" fail count":len(X)-success_count,
                label+" accuracy":accuracy}

    def get_report(self,train_X,train_Y,valid_X,valid_Y):
        #gets predictions for all of X, compares to Y, returns a report

        report={"validation size":len(valid_X),
                "validation ratio":-1,
                "nn size":"-".join([str(i) for i in self.sizes]),
                "learning rate":self.learning_rate,
                "final learning rate":self.final_learning_rate,
                "back count":self.back_count,
                "batch size":self.batch_size}

        train_report=self.get_error_report("train",train_X,train_Y)
        valid_report=self.get_error_report("valid",valid_X,valid_Y)

        report.update(train_report)
        report.update(valid_report)
        
        return report

    def show_report(self,train_X,train_Y,valid_X,valid_Y):
        report=self.get_report(train_X,train_Y,valid_X,valid_Y)
        keys=sorted(list(report.keys()))
        for key in keys:
            print_color(key.upper(),COLORS.GREEN)
            data=report[key]
            if type(data) is list:
                text="    "+"\n    ".join(data)
                if len(text)>100:
                    text=text[:100]+" ..."
                print_color(text,COLORS.ORANGE)
            else:
                print_color(str(data),COLORS.YELLOW)
        return report

