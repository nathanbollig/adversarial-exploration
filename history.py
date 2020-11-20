# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 08:40:05 2020

@author: NBOLLIG
"""
import os
import datetime
import pickle


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
def save_output(output, dir_name, exp_name):
    timestamp = str(int(datetime.datetime.now().timestamp()))
    path = os.path.join(dir_name, exp_name + "_" + timestamp +".csv")
    output.to_csv(path)

def save_image(plt, dir_name, name):
    timestamp = str(int(datetime.datetime.now().timestamp()))
    path = os.path.join(dir_name, name + "_" + timestamp +".jpg")
    plt.savefig(path, dpi = 400)
    return