# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 08:40:05 2020

@author: NBOLLIG
"""
import os
import datetime
import pickle
import pathlib
from pathlib import Path




class History():
    def __init__(self):
        pass
    
    def set_dir(self, dir_name):
        """
        dir_name - Path object
        """
        self.dir_name = dir_name
    
    def save_tables(self, prefix = ""):
        save_output(self.set_summary, self.dir_name, prefix + "set_summary")
        save_output(self.instance_summary, self.dir_name, prefix + "instance_summary")

    def save(self, prefix = ""):
        timestamp = str(int(datetime.datetime.now().timestamp()))
        path = os.path.join(self.dir_name, prefix + "_" + timestamp +".p")
        pickle.dump(self, open(path, "wb"))

# =============================================================================
# Utilities
# =============================================================================
def create_history_from_file(file):
    try:
        h = pickle.load( open(file, "rb" ) )
        h.dir_name = os.path.dirname(file)
    except NotImplementedError:
        temp = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath
        h = pickle.load( open(file, "rb" ) )
        h.dir_name = os.path.dirname(file)
        pathlib.PosixPath = temp
    return h    

def save_output(output, dir_name, exp_name):
    timestamp = str(int(datetime.datetime.now().timestamp()))
    path = os.path.join(dir_name, exp_name + "_" + timestamp +".csv")
    output.to_csv(path)

def save_image(plt, dir_name, name):
    timestamp = str(int(datetime.datetime.now().timestamp()))
    path = os.path.join(dir_name, name + "_" + timestamp +".jpg")
    plt.savefig(path, dpi = 400)
    return


if __name__ == "__main__":
    FILE_NAME = '_1606316466.p'
    dir_name = Path('data/')
    file = os.path.join(dir_name, FILE_NAME)
    h = create_history_from_file(file)