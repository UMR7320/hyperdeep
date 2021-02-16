#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Created on 20 dec. 2020
@author: laurent

t.vanni@unice.fr
'''

import sys
import os
from generator.main import train, generate

def print_help():
    print("usage: python hypergen.py <command> <args>\n")
    print("The commands supported by deeperbase are:\n")
    print("\ttrain\ttrain a generator model")
    print("\tgenerate\ta new text")
    
def print_invalidArgs_mess():
    print("Invalid argument detected!\n")

def get_args():
    args = {}
    for i in range(2, len(sys.argv[1:])+1):
        if sys.argv[i][0] == "-":
            args[sys.argv[i]] = sys.argv[i+1]
        else:
            args[i] = sys.argv[i]
    return args

if __name__ == '__main__':

    # GET COMMAND
    try:
        command = sys.argv[1]
        if command not in ["train", "generate"]:
            raise
    except:
        raise
        print_help()
        exit()

    if command == "train":
        try:
            args = get_args()
            corpus_file = args["-input"]
            model_file = args["-output"]

            # GET SPEC FILE
            #spec = json.loads(open(corpus_file + ".spec", "r").read())

            # TRAIN
            train(corpus_file, model_file)

        except:
            raise
            print_invalidArgs_mess()
            print("The following arguments are mandatory:\n")
            print("\t-input\ttraining file path")
            print("\t-output\toutput file path\n")
            print_help()
            exit()

    if command == "generate":
        try:
            args = get_args()
            model_file = args[2]
            text_file = args[3]
            new_text = generate(model_file, text_file)

            # save predictions in a file
            result_path = "results/" + os.path.basename(text_file) + ".res"
            results = open(result_path, "w")
            results.write(new_text)
            results.close()

        except:
            raise
            print_invalidArgs_mess()
            print("usage: hypergen generate <model> <input-text>:\n")
            print("\t<model>\tmodel file path\n")
            print("\t<vec>\tword vector representations file path\n")
            print("\t<test-data>\ttest data file path\n")
            print_help()
            exit()