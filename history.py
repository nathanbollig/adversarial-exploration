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
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class History():
    def __init__(self):
        pass
    
    def set_dir(self, dir_name):
        """
        dir_name - Path object
        """
        self.dir_name = dir_name
    
    def _get_conf_cuts(self, perturb_set_idx=None):
        """
        Gets a list of conf values to use as cutoff points for different visualizations.
        """
        IS = self.instance_summary.copy()
        
        # Perturb set filter
        if perturb_set_idx != None:
            IS = IS.loc[IS['perturb_set_idx'] == perturb_set_idx]
            
        output = IS['conf'].tolist()
        output.sort(reverse = True)
        
        # Output a log scale up to the max value
        max_conf_cut = max(output)
        conf_cuts = np.logspace(np.log10(1.5), np.log10(max_conf_cut + 1), 150, endpoint = True) - 1
        
        return conf_cuts
    
    def _filter_instance_summary(self, confidence_threshold, perturb_set_idx=None):
        """
        Creates an instance summary filtered to mimic a stopping criterion that would have used a 
        particular confidence threshold. I.e. for each instance, only include up to the first mutation 
        whose confidence exceeded the given threshold.
        
        Optionally also filtered to given perturb_set_idx.
        
        Returns a pandas dataframe.
        """
        IS = self.instance_summary.copy()
        
        # Perturb set filter
        if perturb_set_idx != None:
            IS = IS.loc[IS['perturb_set_idx'] == perturb_set_idx]
        
        # Confidence threshold filter
        last_instance = int(IS.iloc[-1]['instance'])               
        for i in range(1, last_instance + 1):
            # List of indices with confidence above threshold
            higher_conf_idx = list(IS.index[(IS['instance']==i) & (IS['conf'] >= confidence_threshold)])
            
            if len(higher_conf_idx) > 0:
                # Find row index of first mutation whose conf exceeds confidence_threshold
                first_high_conf_idx = min(higher_conf_idx) 
                
                # Find row indices of additional mutations
                del_idx = list(IS.index[(IS['instance']==i) & (IS.index > first_high_conf_idx)])
            
                # Delete indices
                IS = IS.drop(del_idx)
        
        return IS
    
    # SUMMARY (scalar values) over range of confidence thresholds
    def summary_at_conf(self, confidence_threshold, perturb_set_idx=None):
        """
        For a given confidence threshold, run over instances and return the following:
            Actual label flip rate
            Average number of mutations
            Median number of mutations
            Average number of mutations for successful label flip
            Median number of mutations for successful label flip
            Average number of mutations for unsuccessful label flip
            Median number of mutations for unsuccessful label flip
            
        """
        IS = self._filter_instance_summary(confidence_threshold, perturb_set_idx)
        
        actual_labs = []
        model_labs = []
        mut_counts = []
        mut_counts_successful = []
        mut_counts_unsuccessful = []
        
        for _, g in IS.groupby('instance'):
            last_line = g.tail(1)
            actual_lab = last_line['actual_label'].values[0]
            actual_labs.append(actual_lab)
            model_labs.append(int(last_line['pred_proba'].values[0] > 0.5))
            mut_count = last_line['change_number'].values[0]
            mut_counts.append(mut_count)
            if actual_lab == 1:
                mut_counts_successful.append(mut_count)
            else:
                mut_counts_unsuccessful.append(mut_count)
                
        return np.mean(actual_labs), np.mean(model_labs), np.mean(mut_counts), np.median(mut_counts), np.mean(mut_counts_successful), np.median(mut_counts_successful), np.mean(mut_counts_unsuccessful), np.median(mut_counts_unsuccessful)
    
    def _get_summary_over_confs(self, round_to = 2, perturb_set_idx=None):
        """
        Run over a range of confidence thresholds and compute summary at each value.
        
        Returns:
            conf_cuts - list of confidence thresholds
            actual_flip_rates - list of actual label flip rates
            avg_mut_list - list of average mutations
            median_mut_list - list of median mutations
            avg_mut_successful_list - list of average mutations for successful label flip
            median_mut_successful_list - list of median mutations for successful label flip
            avg_mut_unsuccessful_list - list of average mutations for unsuccessful label flip
            median_mut_unsuccessful_list - list of median mutations for unsuccessful label flip
        """
        conf_cuts = self._get_conf_cuts(perturb_set_idx=perturb_set_idx)
        
        actual_flip_rates = []
        model_flip_rates = []
        avg_mut_list = []
        median_mut_list = []
        avg_mut_successful_list = []
        median_mut_successful_list = []
        avg_mut_unsuccessful_list = []
        median_mut_unsuccessful_list = []
        
        for c in conf_cuts:
            rate, model_rate, avg_mut, median_mut, avg_mut_successful, median_mut_successful, avg_mut_unsuccessful, median_mut_unsuccessful = self.summary_at_conf(c, perturb_set_idx=perturb_set_idx)
            actual_flip_rates.append(rate)
            model_flip_rates.append(model_rate)
            avg_mut_list.append(avg_mut)
            median_mut_list.append(median_mut)
            avg_mut_successful_list.append(avg_mut_successful)
            median_mut_successful_list.append(median_mut_successful)
            avg_mut_unsuccessful_list.append(avg_mut_unsuccessful)
            median_mut_unsuccessful_list.append(median_mut_unsuccessful)
        
        return conf_cuts, actual_flip_rates, model_flip_rates, avg_mut_list, median_mut_list, avg_mut_successful_list, median_mut_successful_list, avg_mut_unsuccessful_list, median_mut_unsuccessful_list
    
    def plot_summaries_over_confs(self, perturb_set_idx=None):
        conf_cuts, actual_flip_rates, model_flip_rates, avg_mut_list, median_mut_list, avg_mut_successful_list, median_mut_successful_list, avg_mut_unsuccessful_list, median_mut_unsuccessful_list= self._get_summary_over_confs(perturb_set_idx)
        
        # Actual label flip rates
        plt.plot(conf_cuts, actual_flip_rates)
        plt.xlabel('Confidence threshold')
        plt.ylabel('Actual label flip rate')
        save_image(plt, self.dir_name, "actual_flip_rate_confs")
        plt.show()     
        plt.clf()
        
        # Actual and model label flip rates
        plt.plot(conf_cuts, model_flip_rates, linestyle=':', color = 'k', label = "Model flip rate")
        plt.plot(conf_cuts, actual_flip_rates, linestyle='-', color = 'k', label = "True label flip rate")
        plt.plot([0,0], [1,1], linestyle='-', color = 'grey')
        plt.xlabel('Confidence threshold')
        plt.ylabel('Flip rate')
        plt.xlim(0.5, 1)
        plt.ylim(0.5, 1)
        
        plt.legend(loc='lower right')
        save_image(plt, self.dir_name, "actual_flip_rate_confs")
        plt.show()     
        plt.clf()
        
        # Avg mutation
        plt.plot(conf_cuts, avg_mut_list)
        plt.title("Average number of mutations")
        plt.xlabel('Confidence threshold')
        plt.ylabel('Average number of mutations')
        save_image(plt, self.dir_name, "avg_mut_confs")
        plt.show()
        plt.clf()
    
        # Median mutation
        plt.plot(conf_cuts, median_mut_list)
        plt.title("Average number of mutations")
        plt.xlabel('Confidence threshold')
        plt.ylabel('Median number of mutations')
        save_image(plt, self.dir_name, "median_mut_confs")
        plt.show()
        plt.clf()
        
        # Avg mutation for successful label flip
        plt.plot(conf_cuts, avg_mut_unsuccessful_list, linestyle='--', color = 'k', label="Unsuccessful attempts")
        plt.plot(conf_cuts, avg_mut_successful_list, linestyle='-', color = 'k', label="Successful attempts")
        plt.legend(loc="upper left")
        plt.title("Average number of mutations")
        plt.xlabel('Confidence threshold')
        plt.ylabel('Average number of mutations')
        save_image(plt, self.dir_name, "avg_mut_confs_success")
        plt.show()
        plt.clf()
    
        # Median mutation for successful label flip
        plt.plot(conf_cuts, median_mut_unsuccessful_list, linestyle='--', color = 'k', label="Unsuccessful attempts")
        plt.plot(conf_cuts, median_mut_successful_list, linestyle='-', color = 'k', label="Successful attempts")
        plt.legend(loc="upper left")
        plt.title("Median number of mutations")
        plt.xlabel('Confidence threshold')
        plt.ylabel('Median number of mutations')
        save_image(plt, self.dir_name, "median_mut_confs_success")
        plt.show()
        plt.clf()
        
        # Save raw data
        data = {}
        data['conf_cuts'] = conf_cuts
        data['actual_flip_rates'] = actual_flip_rates
        data['model_flip_rates'] = model_flip_rates
        data['avg_mut_list'] = avg_mut_list
        data['median_mut_list'] = median_mut_list
        data['avg_mut_successful_list'] = avg_mut_successful_list
        data['median_mut_successful_list'] = median_mut_successful_list
        data['avg_mut_unsuccessful_list'] = avg_mut_unsuccessful_list
        data['median_mut_unsuccessful_list'] = median_mut_unsuccessful_list
        pd.DataFrame(data).to_csv('data\data_over_conf.csv', index=False)
    
    # NUMBER OF MUTATION VIOLIN PlOTS
    def mut_dist_at_conf(self, confidence_threshold, perturb_set_idx=None):
        """
        For a given confidence threshold, run over instances and return the 
        distribution of numbers of mutations.
            
        """
        IS = self._filter_instance_summary(confidence_threshold, perturb_set_idx)
        
        mut_counts = []
        
        for _, g in IS.groupby('instance'):
            last_line = g.tail(1)
            mut_counts.append(last_line['change_number'].values[0])
        
        return mut_counts
    
    
    def mutations_over_confs_violin(self, perturb_set_idx=None):
        conf_cuts = [0.5, 0.7, 0.9, 0.95, 0.98, 0.995]
        
        count_distribution_list = []
        
        for c in conf_cuts:
            counts = self.mut_dist_at_conf(c, perturb_set_idx=perturb_set_idx)
            count_distribution_list.append(counts)
        
        plt.violinplot(count_distribution_list, conf_cuts, widths=0.015, showextrema=False, showmeans=False, showmedians=True)
        plt.ylim(0, 8)
        plt.xlabel('Confidence threshold')
        plt.ylabel('Number of mutations')
        save_image(plt, self.dir_name, "violin")
        plt.show()     
        plt.clf()
    
    # POSITIONAL DISTRIBUTION
    def all_positions_hist(self, perturb_set_idx = None):
        IS = self.instance_summary.copy()
        
        # Perturb set filter
        if perturb_set_idx != None:
            IS = IS.loc[IS['perturb_set_idx'] == perturb_set_idx]
        
        # Plot histogram
        positions = IS['pos_to_change'].to_numpy()
        _, self.bins, _ = plt.hist(positions, 60, density=False, facecolor='g', edgecolor='k', alpha=0.75)
        plt.xlabel('Position')
        plt.ylabel('Count')
        plt.title('Mutation location')
        save_image(plt, self.dir_name, "all_positions")
        plt.clf()
    
    def positions_by_success(self, perturb_set_idx = None):
        IS = self.instance_summary.copy()
        
        # Perturb set filter
        if perturb_set_idx != None:
            IS = IS.loc[IS['perturb_set_idx'] == perturb_set_idx]
        
        # Gather position lists for successful and not successful
        success_positions = []
        non_success_positions = []
        
        for _, g in IS.groupby('instance'):
            actual_lab = g.tail(1)['actual_label'].values[0]
            if actual_lab == 1:
                success_positions.extend(g['pos_to_change'].to_numpy())
            else:
                non_success_positions.extend(g['pos_to_change'].to_numpy())
        
        # Plot histogram
        plt.hist([non_success_positions, success_positions], self.bins, density=False, color=['b', 'g'], edgecolor='k', alpha=0.75, label=["Unsuccessful attempts","Successful attempts"], stacked=True)
        plt.xlabel('Position')
        plt.ylabel('Count')
        plt.title('Mutation location')
        plt.legend(loc="upper left")
        save_image(plt, self.dir_name, "positions_by_success")
        plt.clf()
    
    def first_subsequent_positions(self, perturb_set_idx = None):
        IS = self.instance_summary.copy()
        
        # Perturb set filter
        if perturb_set_idx != None:
            IS = IS.loc[IS['perturb_set_idx'] == perturb_set_idx]
        
        # Gather position lists for first and subsequent
        first_positions = []
        subsequent_positions = []
        
        for _, g in IS.groupby('instance'):
            first_pos = g.head(1)['pos_to_change'].values[0]
            first_positions.append(first_pos)
            subsequent_pos = g[1:]['pos_to_change'].to_numpy()
            subsequent_positions.extend(subsequent_pos)
        
        # Plot histogram
        plt.hist([first_positions, subsequent_positions], self.bins, density=False, color=['b', 'g'], edgecolor='k', alpha=0.75, label=["Initial mutations","Subsequent mutations"], stacked=True)
        plt.xlabel('Position')
        plt.ylabel('Count')
        plt.title('Mutation location')
        plt.legend(loc="upper left")
        save_image(plt, self.dir_name, "positions_by_initial_v_subsequent")
        plt.clf()
    
    def positions_activating_mut(self, perturb_set_idx = None):
        IS = self.instance_summary.copy()
        
        # Perturb set filter
        if perturb_set_idx != None:
            IS = IS.loc[IS['perturb_set_idx'] == perturb_set_idx]
        
        # Gather position lists
        positions = IS['pos_to_change'].to_numpy()
        activating_pos = []
        
        for _, g in IS.loc[IS['actual_label'] == 1].groupby('instance'):
            pos = g.head(1)['pos_to_change'].values[0]
            activating_pos.append(pos)
        
        # Plot histogram
        plt.hist(positions, self.bins, density=False, alpha=0)
        plt.hist(activating_pos, self.bins, density=False, facecolor='red', edgecolor='k', alpha=0.75)
        plt.xlabel('Position')
        plt.ylabel('Count')
        plt.title('Locations of activating mutations')
        save_image(plt, self.dir_name, "activating_mut")
        plt.clf()

    def compute_effect_of_errant_mut_at_conf(self, confidence_threshold = 0.999, active_site = [25, 34], pad = 5, perturb_set_idx = None):
        """
        Compute the effect of errant/background mutations (at positions outside of active_site) on actual 
        label flip rate.
        
        pad is a length of characters in which mutations are allowed before or after active site without being
        considered errant
        
        Saves text to h.text_output.
        """
        IS = self._filter_instance_summary(confidence_threshold, perturb_set_idx)        

        errant_successes = 0
        errant_total = 0
        non_errant_successes = 0
        non_errant_total = 0
        
        for _, g in IS.groupby('instance'):
            # is there errant mutation in this instance's trajectory?
            positions = np.sort(g['pos_to_change'].to_numpy())
            min_pos = np.min(positions)
            max_pos = np.max(positions)
            errant = (min_pos < active_site[0] - pad or max_pos > active_site[1] + pad)
            
            # is the attempt successful
            success = (g.tail(1)['actual_label'].values[0] == 1)
            
            # cache result
            if errant == True:
                errant_total += 1
                if success == True:
                    errant_successes += 1
            else:
                non_errant_total += 1
                if success == True:
                    non_errant_successes += 1
        
        result = "With confidence_threshold = %.3f: success rate is %.2f (%i/%i) when errant mutation occurs vs. %.2f (%i/%i) when errant mutation does not occur.\n" % (confidence_threshold, errant_successes/errant_total, errant_successes, errant_total, non_errant_successes/non_errant_total, non_errant_successes, non_errant_total)
        
        if hasattr(self, 'text_output'):
            self.text_output += result
        else:
            self.text_output = result
            
        return result
    
    def compute_effect_of_errant_mut(self, active_site = [25, 34], pad = 5, perturb_set_idx = None):
        conf_cuts = [0.5, 0.7, 0.9, 0.95, 0.98, 0.995]
        
        for c in conf_cuts:
            self.compute_effect_of_errant_mut_at_conf(confidence_threshold = c, active_site = active_site, pad = pad, perturb_set_idx = perturb_set_idx)
    
    # two distributions - model conf at initial in successful flip vs. unsuccessful flip
    def success_and_initial_conf(self):
        conf_success = []
        conf_non_success = []
        for _, g in self.instance_summary.groupby('instance'):
            actual_lab = g.tail(1)['actual_label'].values[0]
            initial_conf = g.head(1)['init_pred_proba'].values[0]
            if actual_lab == 1:
                conf_success.append(initial_conf)
            else:
                conf_non_success.append(initial_conf)
        
        # Plot histogram
        plt.hist([conf_success, conf_non_success], 50, density=True, label=["Successful attempts", "Unsuccessful attempts"])
        plt.xlabel('Initial confidence')
        plt.ylabel('Normalized count')
        plt.title('Initial confidence split by success')
        plt.legend(loc="upper right")

        save_image(plt, self.dir_name, "initial_conf")
        plt.clf()
    
    # SAVING
    def save_tables(self, prefix = ""):
        save_output(self.set_summary, self.dir_name, prefix + "set_summary")
        save_output(self.instance_summary, self.dir_name, prefix + "instance_summary")

    def save(self, prefix = ""):
        timestamp = str(int(datetime.datetime.now().timestamp()))
        path = os.path.join(self.dir_name, prefix + "_" + timestamp +".p")
        pickle.dump(self, open(path, "wb"))
    
    def save_txt(self):
        timestamp = str(int(datetime.datetime.now().timestamp()))
        path = os.path.join(self.dir_name, "output_" + timestamp +".txt")
        with open(path, "w") as text_file:
            text_file.write(self.text_output)

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
    FILE_NAME = '_1608566313.p'
    dir_name = Path('data/')
    file = os.path.join(dir_name, FILE_NAME)
    h = create_history_from_file(file)
    h.plot_summaries_over_confs()
    h.mutations_over_confs_violin()
    h.all_positions_hist()
    h.positions_by_success()
    h.first_subsequent_positions()
    h.positions_activating_mut()
    h.compute_effect_of_errant_mut()
    h.save_txt()
    