import multiprocessing as mp
import multiprocessing.managers as managers
import time
import numpy as np
import pandas as pd
import utils
import datasets
import os
import errno
import signal
from collections import Sequence
from itertools import product
import progressbar


##############################
#    Experiment Launchers    #
##############################

def run(n_trials, fname, evaluators, load_datasetf, tparams, mparams, n_workers, seed=None):
    n_workers = max(n_workers, 1)
    print('Computing %d trials.' % n_trials)
    print()
    return _run_experiment(n_trials, fname, evaluators, load_datasetf, tparams, mparams, n_workers, seed)

def run_existing(n_trials, fname, evaluators, load_datasetf, n_workers, status_delay=1.0, seed=None):
    n_workers = max(n_workers, 1)
    all_results = []

    # Cleanup in case there were existing backup or worker files from a previous run
    for i in range(n_workers):
        wfname = fname.replace('.h5','.worker_%d.h5'%i)
        if os.path.exists(wfname):
            os.remove(wfname)
    basedir = os.path.dirname(fname)
    if os.path.exists(basedir):
        for fn in os.listdir(basedir):
            if fn.endswith('.bak'):
                os.remove(os.path.join(basedir, fn))

    # Get existsing parameters and results
    with pd.HDFStore(fname) as store:
        # Get existing parameters
        mnames  = [ k.split('/')[-1] for k in store.keys() if k.startswith('/method_parameters/') ]
        tparams = [ {k:s[k] for k in s.keys() } for _,s in store['/task_parameters'].iterrows() ]
        mparams = {nm:[ {k:s[k] for k in s.keys()} for _,s in store['/method_parameters/%s'%nm].iterrows() ] for nm in mnames}
        # Get existing results and adjust the total number of trials
        n_trials_prev = 0
        if '/results' in store.keys():
            n_trials_prev = store['results'].groupby(['name','tid','pid'], as_index=False).size().min()
            n_trials_prev = 0 if np.isnan(n_trials_prev) else n_trials_prev
            all_results.append(store['results'])
        if n_trials_prev >= n_trials:
            print('%d trials were requested, but there were already %d trials computed. Exiting.' % (n_trials,n_trials_prev))
            return None
        else:
            print('%d trials were requested, with %d trials already computed. Computing %d more trials.' % (n_trials,n_trials_prev,n_trials-n_trials_prev))
            n_trials = n_trials - n_trials_prev
    # Get existing incomplete results
    partials = None
    i_fname  = fname.replace('.h5', '.incomplete.h5')
    if os.path.exists(i_fname):
        with pd.HDFStore(i_fname) as store:
            if '/results' in store.keys():
                partials = store['results'][['name','tid','pid','seed']]
    # utils.keyboard()
    return _run_experiment(n_trials, fname, evaluators, load_datasetf, tparams, mparams, n_workers, seed=seed, partials=partials)


###############################
#    Parallel: Task Object    #
###############################

class Task:
    def __init__(self, name, tid, pid, seed):
        self.name = name
        self.tid  = tid
        self.pid  = pid
        self.seed = seed

    def __repr__(self):
        return 'Task(name=\'%s\', tid=%d, pid=%d, seed=%d)' % (self.name, self.tid ,self.pid, self.seed)

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        if self.name != other.name:
            return False
        if self.tid != other.tid:
            return False
        if self.pid != other.pid:
            return False
        if not(np.isclose(self.seed, other.seed)):
            return False
        return True

    def to_dict(self):
        return {'name':self.name, 'tid':self.tid, 'pid':self.pid, 'seed':self.seed}


#################################
#    Parallel: Task Iterator    #
#################################

class TaskIterator:
    def __init__(self, n_trials, tparams, mparams, seed=None, partials=None, load_datasetf=None):
        self._lock     = mp.Lock()
        self._method_names = list(mparams.keys())
        self._n_tconfs_pt  = len(tparams)
        self._n_mconfs_pt  = [ len(mparams[k]) for k in self._method_names ]
        self._base_seed = make_seed() if seed is None else seed
        self._n_trials  = n_trials
        self._loadf = load_datasetf
        self._tparams = tparams
        # Record any trials that are partially completed and their corresponding seeds
        if partials is None:
            self._seeds    = []
            self._partials = []
        else:
            self._seeds    =  partials.seed.unique().tolist()
            self._partials = [ Task(r['name'],r['tid'],r['pid'],r['seed']) for _,r in partials.iterrows() ]
        # Set up for the first trial
        self.seed = None
        self.reset()

    def __iter__(self):
        return self

    def __next__(self):
        if self._terminate:
            raise StopIteration
        with self._lock:
            return self._next()

    def _next(self):
        if self.tnum >= self._n_trials:
            raise StopIteration 
        try:
            # Get the next task in this trial
            return self._next_in_trial()
            # if self.tnum <= 1:
            #     return self._next_in_trial()
            # else:
            #     raise StopIteration
        except StopIteration:
            # We've run out of tasks for this trial - start a new one
            self.setup_next_trial()
            return self._next()

    def _next_in_trial(self):
        rejects = []
        while True:
            # If this method's param confs have been tested, go to the next method
            if self.pid >= self._n_mconfs_pt[self.nid]:
                self.pid  = 0
                self.nid += 1
            # If all methods have been tested, move to the next task configuration
            if self.nid >= len(self._method_names):
                self.nid  = 0
                self.tid += 1
            # If all task confs have been tested, the iteration is complete
            if self.tid >= self._n_tconfs_pt:
                raise StopIteration
            # Create a task object
            task = self.make_task()
            if not(task in self._partials):
                if len(rejects) > 0:
                    print('rejected %d previously.' % len(rejects))
                return task
            else:
                rejects.append(task)

    def close(self):
        self._terminate = True

    def n_tasks_per_trial(self):
        return self._n_tconfs_pt * sum(self._n_mconfs_pt)

    def get_status(self):
        with self._lock:
            n_tpt = self.n_tasks_per_trial()
            n_t   = self._n_trials * n_tpt
            ipct = 100.0 * self.cnum / n_tpt
            opct = 100.0 * (n_tpt*self.tnum + self.cnum) / (n_t*n_tpt)
            return ipct, opct, self.tnum+1

    def reset(self):
        self._terminate = False
        self._random = np.random.RandomState(self._base_seed)
        self.tnum = None
        self.setup_next_trial()

    def setup_next_trial(self):
        self.tid = 0
        self.pid = 0
        self.nid = 0
        self.cnum = 0
        tnum = self.tnum
        self.tnum = 0 if (self.tnum is None) else self.tnum+1
        attempts = 0
        # Keep generating seeds until all splits have at least one of each label and T value
        while attempts < 10:
            # attempts += 1
            if len(self._seeds) > 0:
                seed = self.seed
                self.seed = self._seeds.pop()
            else:
                seed = self.seed
                self.seed = make_seed(self._random)
            if not(self._loadf is None):
                valid = True
                for tp in self._tparams:
                    dataset = self._loadf(tp, seed=self.seed) 
                    if isinstance(dataset, datasets.Dataset):
                        if len(np.unique(dataset._T)) > 1:
                            _, Y, T = dataset.safety_splits()
                            if len(np.unique(Y)) == 1 or len(np.unique(T)) == 1:
                                valid = False
                                continue
                            _, Y, T = dataset.optimization_splits()
                            if len(np.unique(Y)) == 1 or len(np.unique(T)) == 1:
                                valid = False
                                continue
                            _, Y, T = dataset.training_splits()
                            if len(np.unique(Y)) == 1 or len(np.unique(T)) == 1:
                                valid = False
                                continue
                            _, Y, T = dataset.testing_splits()
                            if len(np.unique(Y)) == 1 or len(np.unique(T)) == 1:
                                valid = False
                                continue
                    elif isinstance(dataset, datasets.RLDataset):
                        if len(np.unique(dataset._T)) > 1:
                            _, _, R, T, _ = dataset.safety_splits()
                            if len(np.unique(T)) == 1:
                                valid = False
                                continue
                            _, _, R, T, _ = dataset.optimization_splits()
                            if len(np.unique(T)) == 1:
                                valid = False
                                continue
                            _, _, R, T, _ = dataset.training_splits()
                            if len(np.unique(T)) == 1:
                                valid = False
                                continue
                            _, _, R, T, _ = dataset.testing_splits()
                            if len(np.unique(T)) == 1:
                                valid = False
                                continue
                if valid:
                    break

    def make_task(self):
        name = self._method_names[self.nid]
        # print(self.seed)
        task = Task(name, self.tid, self.pid, self.seed)
        self.pid  += 1
        self.cnum += 1
        return task


#################################################
#    Parallel: Task Iterator Proxy & Manager    #
#################################################

class IteratorProxy(managers.BaseProxy):
    _exposed_ = ('__next__', 'close', 'n_tasks_per_trial', 'get_status')
    def __iter__(self):
        return self
    def __next__(self, *args):
        return self._callmethod('__next__', args)
    def close(self, *args):
        return self._callmethod('close', args)
    def n_tasks_per_trial(self, *args):
        return self._callmethod('n_tasks_per_trial', args)
    def get_status(self, *args):
        return self._callmethod('get_status', args)

class ExperimentManager(managers.SyncManager): pass
ExperimentManager.register('TaskIterator', TaskIterator, proxytype=IteratorProxy)


##################################
#    Parallel: Worker Process    #
##################################
import sys
def process_tasks(wid, tasks, n_trials, fname, evaluators, load_datasetf, all_tparams, all_mparams, result_lock):
    ignore_signals()
    save_data = []
    prev_tid  = None
    prev_save = 0
    try:
        for task in tasks:
            name = task.name
            tparams = all_tparams[task.tid]
            mparams = all_mparams[name][task.pid]

            if not(prev_tid == task.tid): # only load the dataset if it's not loaded already
                dataset = load_datasetf(tparams, seed=task.seed)
                prev_tid = task.tid
            results = evaluators[task.name](dataset, mparams)
            results.update(task.to_dict())
            save_data.append(pd.Series(results))

            if time.time()-prev_save > 5: # save once per five seconds
                if save_worker_results(fname, wid, save_data, result_lock):
                    prev_save = time.time()
                    save_data = []
    finally:
        save_worker_results(fname, wid, save_data, result_lock, block=True)


########################################
#    Parallel: Experiment Execution    #
########################################

def _run_experiment(n_trials, fname, evaluators, load_datasetf, tparams, mparams, n_workers, seed=None, partials=None, task_iterator=None):
    if n_trials <= 0:
        return None
    manager = ExperimentManager()
    manager.start(initializer=ignore_signals)
    result_locks  = [ manager.Lock() for _ in range(n_workers) ]
    terminate = manager.Event()
    
    if task_iterator is None:
        task_iterator = manager.TaskIterator(n_trials, tparams=tparams, mparams=mparams, seed=seed, partials=partials, load_datasetf=load_datasetf)

    # Create the workers
    workers = []
    for wid in range(n_workers):
        args = (wid, task_iterator, n_trials, fname, evaluators, load_datasetf, tparams, mparams, result_locks[wid])
        proc = mp.Process(target=process_tasks, args=args)
        workers.append(proc)
    # Start the workers
    for w in workers:
        w.start()

    # Create and start the results consolidator process
    c_args = (n_workers, task_iterator, fname, result_locks, terminate, False)
    c_proc = mp.Process(target=consolidate, args=c_args)
    c_proc.start()

    # Wait for the workers to finish and print progress
    err = None
    n_interrupts = 0
    prev_tnum = 1
    inum_digits = 2*np.floor(np.log10(n_trials)).astype(int)+1
    inum_fmt = lambda t: '%d/%d'.rjust(inum_digits) % (t, n_trials)
    bar = progressbar.ProgressBar(max_value=100, widgets=['Trial: %s  '%inum_fmt(1), 
                                                          progressbar.Bar(), '  ', 
                                                          progressbar.Timer()])
    while any([ w.is_alive() for w in workers ]):
        try:
            if not(terminate.is_set()):
                ipct, opct, trialnum = task_iterator.get_status()
                bar.update(np.floor(ipct).astype(int))
                if trialnum != prev_tnum and trialnum <= n_trials:
                    prev_tnum = trialnum
                    bar.finish()
                    bar = progressbar.ProgressBar(max_value=100, widgets=['Trial: %s  '%inum_fmt(trialnum), 
                                                          progressbar.Bar(), '  ', 
                                                          progressbar.Timer()])

            time.sleep(1.0);
        except KeyboardInterrupt:
            task_iterator.close()
            terminate.set()
            try:
                del bar
            except:
                pass
            print('\nKeyboardInterrupt')
            n_interrupts += 1
            if n_interrupts >= 50:
                print('Spam detected. Exiting.')
                for w in workers:
                    w.terminate()
                break
            continue
        except Exception as e:
            err = e
            break

    # Cleanup before exit
    task_iterator.close()
    terminate.set()
    c_proc.join()
    print()
    consolidate_results(n_workers, task_iterator, fname, result_locks, debug=True)
    if not(err is None):
        raise err
    return task_iterator


##################################################
#    Parallel: Saving & Consolidating Results    #
##################################################

def save_worker_results(fname, wid, save_data, lock, block=False):
    if len(save_data) == 0:
        return False
    if lock.acquire(block):
        # Save results to a worker-specific file
        fname = fname.replace('.h5', '.worker_%d.h5'%wid)
        df = pd.DataFrame(save_data)
        with pd.HDFStore(fname) as store:
            if '/results' in store.keys():
                df = pd.concat([store['results'], df], ignore_index=True)
            store.put('results', df)
        lock.release()
        return True
    else:
        return False

def consolidate(n_workers, task_iterator, fname, result_locks, exit, debug=False):
    ignore_signals()
    while not(exit.is_set()):
        # print('\nCONSOLIDATING\n')
        consolidate_results(n_workers, task_iterator, fname, result_locks, debug=debug)
        time.sleep(1.0)
    # print('\nCONSOLIDATOR EXITING...\n')

def consolidate_results(n_workers, task_iterator, fname, result_locks, debug=False):
    # Create a helper for extracting files and making backups
    backups, all_results = [], []
    def _append_results(fname, move_existing=True, all_results=all_results, backups=backups, lock=None):
        if os.path.exists(fname):
            if (lock is None) or (lock.acquire()):
                # Get results if they exist
                store = pd.HDFStore(fname)
                if '/results' in store.keys():
                    all_results.append(store['results'])
                store.close()
                # Move the file to a backup if required
                if move_existing:
                    bak_fname = fname + '.bak'
                    # print('\nDoes %s already exist? %r\n' % (bak_fname,os.path.exists(bak_fname)))
                    # if os.path.exists(bak_fname):
                    #     print('\'%s\' already exists.' % bak_fname)
                    #     os.remove(bak_fname)
                    try:
                        os.rename(fname, bak_fname)
                    except Exception as e:
                        os.remove(bak_fname)
                        os.rename(fname, bak_fname)
                    backups.append(bak_fname)
                # Release the lock if one was specified
                if not(lock is None):
                    lock.release()

    # Get existing completed results
    _append_results(fname, move_existing=False)

    # Get existing incomplete results
    i_fname = fname.replace('.h5', '.incomplete.h5')
    _append_results(i_fname, move_existing=False)

    # Get results from each worker's record file
    for wid, lock in enumerate(result_locks):
        w_fname = fname.replace('.h5','.worker_%d.h5'%wid)
        _append_results(w_fname, lock=lock)

    # Exit if there are no results
    if len(all_results) == 0:
        return

    # Add the results to a single, new dataframe
    df = pd.concat(all_results, ignore_index=True)      

    # Split the results into samples for completed and incompleted trials
    gb = df.groupby(['seed']).size() == task_iterator.n_tasks_per_trial()
    gb = gb.reset_index(level=['seed'])
    gb = gb.rename(columns={0:'_is_complete'})
    df = df.merge(gb, on=['seed'])
    cresults = df.loc[ df._is_complete, df.columns!='_is_complete']
    iresults = df.loc[~df._is_complete, df.columns!='_is_complete']


    if any(iresults.groupby(['seed','name','tid','pid']).size() > 1):
        print([len(v) for v in all_results])
        with pd.HDFStore(i_fname.replace('incomplete','reference')) as store:
            print(i_fname.replace('incomplete','reference'))
            store.put('results', iresults)
        utils.keyboard()
    else:
        # Save completed trials
        if len(cresults) > 0:
            with pd.HDFStore(fname) as store:
                store.put('results', cresults)

        # Save incomplete trials
        if len(iresults) > 0:
            with pd.HDFStore(i_fname) as store:
                store.put('results', iresults)
        else:
            if os.path.exists(i_fname):
                os.remove(i_fname)

        # Delete backups
        for bak in backups:
            os.remove(bak)

    # Print a debug message if requred
    if debug:
        print('...............................................')
        # with pd.HDFStore(fname) as store:
        #     if '/results' in store.keys():
        #         print('Completed results:')
        #         print( store['results'].groupby('seed').size())
        #     else:
        #         print('No complete results.')
        # print()
        with pd.HDFStore(i_fname) as store:
            if '/results' in store.keys():
                print('Incomplete results:')
                print( store['results'].groupby('seed').size())
            else:
                print('No incomplete results.')
        print('...............................................')



########################
#    IO Preparation    #
########################

def prepare_paths(dirname, tparams, mparams, smla_names, root='results', filename=None):
    # Prepare the paths needed to save results
    print('Preparing results directory.')
    is_new  = True
    basedir = os.path.join(root, dirname)
    for subdir in utils.subdir_incrementer(basedir):
        if not(os.path.isdir(subdir)):
            break
        is_new = False
    
    message = {True  : 'Saving to \'%s\'.',
               False : 'Path \'.\\%s\' already exists.\n  Saving results to \'.\\%%s\' instead.' % basedir}
    print('  %s' % (message[is_new]%subdir))
    if not(filename is None):
        filename = filename if filename.endswith('.h5') else filename+'.h5'
    else:
        filename = dirname + '.h5'
    os.makedirs(subdir)
    save_path = os.path.join(subdir, filename)  

    # Save the parameters to the hdf5 store
    with pd.HDFStore(save_path) as store:
        store.append('task_parameters', pd.DataFrame(utils.stack_all_dicts(*tparams)))
        for k,mps in mparams.items():
            store.append('method_parameters/%s' % k, pd.DataFrame(utils.stack_all_dicts(*mps)))
        store.append('meta', pd.DataFrame({'smla_names':smla_names}))

    return save_path

################################
#    Parameter Construction    #
################################

def make_parameters(task_params, method_params, expand=None):
    tparams = _expand_parameters(task_params, expand)
    mparams = { k:_expand_parameters(v, expand) for k,v in method_params.items() }
    return tparams, mparams

def _expand_parameters(params, expand=None):
    vnames, vparams = [], []
    sdict = {}
    for k,v in params.items():
        if isinstance(v, (np.ndarray, Sequence)) and ((expand is None) or (k in expand)):
            vnames.append(k)
            vparams.append(v)
        else:
            sdict[k] = v
    out_dicts = []
    for _vparams in product(*vparams):
        _d = dict(sdict)
        for k,v in zip(vnames, _vparams):
            _d[k] = v
        out_dicts.append(_d)
    return out_dicts


#######################
#    Misc. Helpers    #
#######################

def ignore_signals():
    # children ignore SIGINT to allow graceful termination
    sig = signal.signal(signal.SIGINT, signal.SIG_IGN) 
    return [sig]

def make_seed(random=np.random, digits=8):
    return np.floor(random.rand()*10**digits).astype(int)