import pandas as pd
import os
import time
import datetime
FILE_PATH = './log_files/'
class CSVLogger:
    def __init__(self, name):
        ts = time.gmtime()
        time_stamp = time.strftime("%Y-%m-%d-%H:%M:%S", ts)
        self.name = name
        if not os.path.exists(FILE_PATH):
            os.mkdir(FILE_PATH)
        self.filename = FILE_PATH + time_stamp + '_' + name + '.csv'

    def to_dict_entry(self,success, losses, steps_taken, cumulated_reward):
        d = dict()
        d['success'] = [success]
        d['steps'] = [steps_taken]
        d['cumulated_reward'] = [cumulated_reward]
        if losses != None:
            for i in range(len(losses)):
                for k,v in losses[i].items():
                    d[str(k)+'_'+str(i)] = [v]
        return d

    def log_to_file(self, d):
        df = pd.DataFrame.from_dict(d)
        df.to_csv(self.filename, mode='a', header=True, index=True)

