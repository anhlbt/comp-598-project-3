"""
COMP598 Project 3 Neural Network

Usage:
  main.py <training-csv> [<target-csv>] [options]

Options:
    --trials=<count>       Backpropagate this many times [default: 10000]
    --learn-rate=<lr>      Set learning rate to this [default: 0.1]
    --random               Do not use seed, make trials actually random each time.
    --timer=<interval>     Wait this many seconds before printing an update during big jobs. [default: 10]
    --validate
    --validation-ratio=<r>  Number from 0 to 1. 0.8 means 80% of data is used for training, 20% for validation. [default: 0.8]
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

    training=args["<training-csv>"]
    if not is_file(training):
        return

    target=args["<target-csv>"]
    if target and not is_file(target):
        return

    try:
        trials=int(args["--trials"])
    except ValueError:
        print_color("Bad value for trials.")
        return

    try:
        learn_rate=int(args["--learn-rate"])
    except ValueError:
        print_color("Bad value for learn rate.")
        return

    try:
        interval=int(args["--timer"])
    except ValueError:
        print_color("Bad value for timer interval.")
        return

    try:
        sizes=[int(i) for i in args["--sizes"].split(",")]
    except ValueError:
        print_color("Bad value for sizes.")
        return

    try:
        validation_ratio=float(args["--validation-ratio"])
    except ValueError:
        print_color("Bad value for validation ratio.")
        return

    start_time=time.time()

    X_train,Y_train,X_valid,Y_valid=get_data(training,validation_ratio)

    nn=NeuralNetwork(sizes,learning_rate=learn_rate,verbose=args["--verbose"])
    nn.train(X_train,Y_train,trials,timer_interval=interval)

    if args["--validate"]:
        print_color("Starting validation.",COLORS.GREEN)
        report=nn.get_report(X_valid,Y_valid)
    if target:
        print_color("Making predictions.",COLORS.GREEN)
        nn.make_predictions_csv(target)

    print("Done after %s seconds."%round(time.time()-start_time,1))

if __name__ == "__main__":
    args = docopt(__doc__, version="1.0")
    main(args)
















