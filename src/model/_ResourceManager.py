'''
Greg Schuette 2024 (Note: Buggy)
'''
from concurrent.futures import ThreadPoolExecutor
import time
from tqdm.auto import tqdm

# Essentially a wrapper for ThreadPoolExecutor. Intelligently assigns jobs in the queue
# to specific resources (replicas attached to individual GPUs) as they become available. 
class ResourceManager:

    def __init__(self,replicas,check_interval=.05):
        self.replicas = replicas
        self.__len = len(self.replicas)
        self.running_tasks = {k:None for k in range(len(self))}
        self.completed_tasks = {}
        self.queue = []
        self.curr_job_id = 0 
        self.__run = False
        self.__manager = None
        self.executor = None
        assert isinstance(check_interval,float) and check_interval>0, \
        'check_interval should be a positive-valued float.'
        self.run(check_interval)

    def __len__(self):
        return self.__len

    def __cleanup(self):
        for k,task in self.running_tasks.items():
            if task and task[-1].done():
                self.completed_tasks[task[0]] = task[1].result()
                self.running_tasks[k] = None

    def n_active_tasks(self,cleanup=True):
        if cleanup:
            self.__cleanup()
        return sum(task is not None for task in self.running_tasks.values())

    def __available_resources(self):
        # Manger calls this AFTER n_active_tasks, so we've already
        # checked for completed processes & handled them and can be lazy here. 
        return [(k,self.replicas[k]) for k,task in self.running_tasks.items() if task is None]

    def __manage(self,check_interval):
        '''
        Manage the queue, completed processes, etc., on one looping thread. 
        Importantly, assigns tasks to GPUs in the order they become available
        to avoid wasting resources. 
        '''
        if check_interval is not None:
            try:
                self.__check_interval = float(check_interval)
            except:
                raise Exception('check_interval should be a positive-valued float.')
        check_interval = self.__check_interval
        while self.__run:
            if self.n_active_tasks() < len(self) and self.queue:
                for replica_id, replica in self.__available_resources():
                    process_id, f_name, args, kwargs = self.queue.pop(0)
                    
                    self.running_tasks[replica_id] = (
                        process_id,
                        self.executor.submit(
                            getattr(replica,f_name), *args, **kwargs
                        )
                    )
                    if not self.queue:
                        # In case we've run out of tasks to assign before we run out of available resources
                        break
            
            time.sleep(check_interval)

    def run(self,check_interval=None):
        if self.__run and self.running():
            return # should probably allow check_interval to update...

        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True) # Just to be safe
            del self.executor

        self.executor = ThreadPoolExecutor(max_workers=len(self)+1)
        self.__run = True
        if self.__manager:
            if self.__manager.running():
                return
            self.kill()
        self.__manager = self.executor.submit(self.__manage,check_interval)

    def kill(self,kill_subprocesses=True):
        self.__run = False
        if self.running():
            self.__manager.result() 
            del self.__manager
            self.__manager = None
        if self.executor is not None:
            self.executor.shutdown()
        if kill_subprocesses:
            for pid,task in self.running_tasks.items():
                if task:
                    task[1].cancel()
                    self.running_tasks
                    self.running_tasks[pid] = None

    def running(self):
        return self.__manager is not None and self.__manager.running()
        
    def wait(self,check_interval=.05,verbose=False,desc=None):
        n_total_tasks = self.curr_job_id
        n_completed_tasks = len(self.completed_tasks)

        if n_completed_tasks == n_total_tasks:
            return

        pbar = tqdm(initial = n_completed_tasks, total = n_total_tasks, disable = not verbose, desc=desc)
        while True:
            time.sleep(check_interval)
            n_completed_tasks1 = len(self.completed_tasks)
            if n_completed_tasks1 == n_completed_tasks:
                continue

            pbar.update(n_completed_tasks1-n_completed_tasks)
            if verbose:
                pbar.display()
            n_completed_tasks = n_completed_tasks1
            if n_completed_tasks == n_total_tasks:
                return

            # Check that the manager task hasn't died; if it has, get the result to raise the relevant exception
            if not self.running():
                if self.__manager is not None:
                    self.__manager.result()
                # If we're here, the process must have been purposefully killed at some point... 
                # Worth raising an exception for the user so they aren't stuck in an infinite loop
                raise Exception('The process seems to have been killed, so you will wait forever!')

    def results(self,check_interval=.05,verbose=False,desc=None):
        self.wait(check_interval,verbose,desc=desc)
        process_ids = list(self.completed_tasks)
        process_ids.sort()
        return [self.completed_tasks[pid] for pid in process_ids]

    def clear(self):
        was_running = self.running()
        self.kill()
        time.sleep(.05) # in case racing with __manage
        for pid,task in self.running_tasks.items():
            if task:
                task[1].cancel()
                self.running_tasks
                self.running_tasks[pid] = None
        self.queue.clear()
        self.completed_tasks.clear()
        self.curr_job_id = 0
        if was_running:
            self.run()
    
    def submit(self,f_name,*args,**kwargs):
        self.queue.append((self.curr_job_id, f_name, args, kwargs))
        self.curr_job_id+= 1

    def __del__(self):
        self.kill()
        self.executor.shutdown(wait=True,cancel_futures=True)
        del self.__manager
        del self.executor
        self.replicas.clear()
        del self.replicas
        self.running_tasks.clear()
        del self.running_tasks
        self.completed_tasks.clear()
        del self.completed_tasks
        self.queue.clear()
        del self.queue
        del self.__len
        del self.curr_job_id
        del self.__run
        super().__del__()