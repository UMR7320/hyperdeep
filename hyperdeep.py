#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Created on 16 nov. 2017
@author: laurent.vanni@unice.fr
'''
import sys
import os
import json

from preprocess.general import PreProcessing
from classifier.main import train, predict
from preprocess.w2vec import create_vectors, get_most_similar

def print_help():
    print("usage: python hyperdeep.py <command> <args>\n")
    print("The commands supported by deeperbase are:\n")
    print("\tword2vec\ttrain a word2vec model")
    print("\tnn\tquery for nearest neighbors\n")
    print("\ttrain\ttrain a CNN model for sentence classification\n")
    print("\tpredict\tpredict most likely labels")
    
def print_invalidArgs_mess():
    print("Invalid argument detected!\n")

def get_args():
    args = {}
    for i in range(2, len(sys.argv[1:])+1):
        if sys.argv[i][0] == "-":
            try:
                args[sys.argv[i]] = sys.argv[i+1]
            except:
                args[sys.argv[i]] = True
        else:
            args[i] = sys.argv[i]
    return args

if __name__ == '__main__':

    # GET COMMAND
    try:
        command = sys.argv[1]
        if command not in ["word2vec", "nn", "train", "predict"]:
            raise
    except:
        print_help()
        exit()

    # GET CONFIG FILE
    try:
        config = json.loads(open("config.json", "r").read())
        config["plot"] = True # to print example output (tds, wtds, lime)
    except:
        print("Error: no config file found")
        exit()

    # EXECT COMMAND
    if command == "word2vec":
        try:
            args = get_args()
            corpus_file = args["-input"]
            model_file = args["-output"]

            preprocessing = PreProcessing(model_file, config)
            preprocessing.loadData(corpus_file)
            create_vectors(preprocessing.channel_texts, model_file, config)
        except:
            print_invalidArgs_mess()
            print("The following arguments are mandatory:\n")
            print("\t-input\ttraining file path")
            print("\t-output\toutput file path\n")
            print_help()
            exit()
            
    if command == "nn": # nearest neighbors
        try:
            args = get_args()
            model = args[2]
            word = args[3]
            most_similar_list = get_most_similar(word, model)
            for neighbor in most_similar_list:
                print(neighbor[0], neighbor[1])

            # save predictions in a file
            result_path = "results/" + os.path.basename(model) + ".nn"
            results = open(result_path, "w")
            results.write(json.dumps(most_similar_list))
            results.close()

        except:
            print_invalidArgs_mess()
            print("usage: python hyperdeep.py nn <model> <word>\n")
            print("\tmodel\tmodel filename")
            print("\tword\tinput word\n")
            print_help()
            exit()
            
    if command == "train":
        try:
            args = get_args()
            corpus_file = args["-input"]
            model_file = args["-output"]

            # TRAIN
            scores = train(corpus_file, model_file, config)

        except:
            raise
            print_invalidArgs_mess()
            print("The following arguments are mandatory:\n")
            print("\t-input\ttraining file path")
            print("\t-output\toutput file path\n")
            print("The following arguments for training are optional:\n")
            print("\t-w2vec\tword vector representations file path\n")
            print("\t-tg\tuse tagged inputs (TreeTagger format)\n")
            print_help()
            exit()

    if command == "predict":
        try:
            args = get_args()
            model_file = args[2]
            text_file = args[3]

            config["ANALYZER"] = []
            if "-lime" in args.keys():
                config["ANALYZER"] += ["LIME"]
            
            if "-tds" in args.keys():
                config["ANALYZER"] += ["TDS"]

            if "-wtds" in args.keys():
                config["ANALYZER"] += ["wTDS"]

            predictions = predict(text_file, model_file, config)

            # save predictions in a file
            result_path = "results/" + os.path.basename(text_file) + ".pred"
            results = open(result_path, "w")
            results.write(json.dumps(predictions))
            results.close()

        except:
            print_invalidArgs_mess()
            print("usage: hyperdeep predict <model> <test-data>:\n")
            print("\t<model>\tmodel file path\n")
            print("\t<vec>\tword vector representations file path\n")
            print("\t<test-data>\ttest data file path\n")
            print_help()
            exit()
