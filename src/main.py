"""
COMP598 Project 3 Neural Network

Usage:
  main.py <train-csv> <prediction-csv> [<target-csv>] [options]

Options:
    --trials=<count>       Backpropagate this many times [default: 10000]
    --learn-rate=<lr>      Set learning rate to this [default: 0.1]
    --normalize            Subtract the mean and divide by the standard deviation for all of X.
    --random               Do not use seed, make trials actually random each time.
    --timer=<interval>     Wait this many seconds before printing an update during big jobs. [default: 10]
    --logging              Writes the weights, outputs, etc to csvs in the logs folder for every backpropagation step.

    --validate
    --validation-ratio=<r>  Number from 0 to 1. 0.8 means 80% of data is used for training, 20% for validation. If value is 1 and --validate is specified, then training=validation for basic testing purposes [default: 0.8]

    --sizes=<sizes>        Describes the number of nodes per layer: input, hidden(s), and output. [default: 2,2,1]

    --verbose

    -h --help              Show this screen.
    -v --version           Show version.
"""

import os.path
import numpy as np
from docopt import docopt
from constants import *
from utilities import *
from neural_net import NeuralNet

def get_csv_path():
    #scans all existing data csvs, returns the name with the lowest number suffix that is unused
    folder="results"
    if not os.path.isdir(folder):
        os.mkdir(folder)
    index=1
    def get_path(index,folder):
        filename="results-%s.csv"%str(index).zfill(3)
        return os.path.join(folder,filename)
    while os.path.isfile(get_path(index,folder)):
        index+=1
    return get_path(index,folder)

def is_file(path):
    if not os.path.isfile(path):
        print_color("Not a file: '%s'"%path,color=COLORS.RED)
        return 0
    return 1

def main(args):
    if not args["--random"]:
        random.seed(123)
        np.random.seed(123)

    train_csv=args["<train-csv>"]
    if not is_file(train_csv):
        return

    prediction_csv=args["<prediction-csv>"]
    if not is_file(prediction_csv):
        return

    target=args["<target-csv>"]
    if target and not is_file(target):
        return

    try:
        trials=int(args["--trials"])
    except ValueError:
        print_color("Bad value for trials.",COLORS.RED)
        return

    try:
        learn_rate=float(args["--learn-rate"])
    except ValueError:
        print_color("Bad value for learn rate.",COLORS.RED)
        return

    try:
        interval=int(args["--timer"])
    except ValueError:
        print_color("Bad value for timer interval.",COLORS.RED)
        return

    try:
        sizes=[int(i) for i in args["--sizes"].split(",")]
    except ValueError:
        print_color("Bad value for sizes.",COLORS.RED)
        return

    try:
        validation_ratio=float(args["--validation-ratio"])
    except ValueError:
        print_color("Bad value for validation ratio.",COLORS.RED)
        return

    start_time=time.time()

    print_color("Opening file: %s"%train_csv,COLORS.YELLOW)

    X_train,Y_train,X_valid,Y_valid=get_data_2csv(train_csv,prediction_csv,validation_ratio,normalize=True)
    if validation_ratio==1 and args["--validate"]:
        X_valid,Y_valid=X_train,Y_train

    if sizes[0]!=len(X_train[0]):
        print_color("Bad 'sizes' parameter for this input data. sizes[0]=%s len(X[0])=%s"%(sizes[0],len(X_train[0])),COLORS.RED)
        return

    print_color("Initializing neural net.",COLORS.GREEN)
    nn=NeuralNet(sizes,learning_rate=learn_rate,
            verbose=args["--verbose"],timer_interval=interval,
            logging=args["--logging"])
    nn.train(X_train,Y_train,trials)

    if args["--validate"]:
        print_color("Starting validation.",COLORS.GREEN)
        nn.show_report(X_valid,Y_valid)
    if target:
        raise NotImplementedError
        print_color("Making predictions.",COLORS.GREEN)
        nn.make_predictions_csv(target)

    print_color("Done after %s seconds."%round(time.time()-start_time,1),COLORS.GREEN)

if __name__ == "__main__":
    args = docopt(__doc__, version="1.0")
    main(args)
















