from __future__ import (absolute_import, division, print_function)
import os
import sys
import threading
import time
import pickle


python = 'python'
#reduce_one_run_script = '/home/ntv/ml_peak_integration/integrate_peak_set_keras_hkl.py'
reduce_one_run_script = '/home/ntv/ml_peak_integration/integrate_peak_set_knn.py'
run_nums = [9113, 9114, 9115, 9116, 9117]
max_processes = 3

#Define a class for threading (taken from ReduceSCD_Parallel.py) and set up parallel runs
class ProcessThread ( threading.Thread ):
    command = ""

    def setCommand( self, command="" ):
        self.command = command

    def run ( self ):
        print('STARTING PROCESS: ' + self.command)
        os.system( self.command )

procList=[]
index = 0
for r_num in run_nums:
    procList.append( ProcessThread() )
    cmd = '%s %s %s' % (python, reduce_one_run_script, r_num)
    procList[index].setCommand( cmd )
    index = index + 1

all_done = False
active_list=[]
while not all_done:
    if  len(procList) > 0 and len(active_list) < max_processes :
        thread = procList[0]
        procList.remove(thread)
        active_list.append( thread )
        thread.start()
    time.sleep(2)
    for thread in active_list:
        if not thread.isAlive():
            active_list.remove( thread )
    if len(procList) == 0 and len(active_list) == 0 :
        all_done = True

