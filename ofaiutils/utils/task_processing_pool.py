#!/usr/bin/env python

import sys, re, time
import numpy as np
import logging
from multiprocessing import (Process, Manager, cpu_count, Lock,
                             current_process, Queue)
from multiprocessing.managers import BaseManager, DictProxy, ListProxy
from queue import Queue, Empty
from collections import Counter, defaultdict

LOGGER = logging.getLogger(__name__)

class Task(object): pass
            
class StopTask(Task): pass

class TaskProcessor(object):
    def __init__(self,tasks,manager): pass
    def do_task(self,task): pass

class Worker(Process):

    def __init__(self, tasks, processor_dict):
        Process.__init__(self)
        self.tasks = tasks
        self.pr_dict = processor_dict
        self.daemon =True
        self.go = True
        self.start()

    def run(self):
        while self.go:
            task = self.tasks.get()
            processor = self.pr_dict.get(type(task), False)

            if processor:

                processor.do_task(task)
                self.tasks.task_done()

            elif isinstance(task, StopTask):


                self.go = False
            else:

                LOGGER.info(('do not know how to do {0}, ' +
                             'removing it from queue') 
                            .format(type(task).__name__))

                self.tasks.task_done()
        LOGGER.debug('worker {0} stopping'.format(current_process().name))
        return True

class WorkerPool(object):

    def __init__(self, n_processes = None):
        if n_processes is None:
            self.n_processes = cpu_count()
        else:
            self.n_processes = n_processes
        if False:
            self.manager = Manager()
        else:
            self.manager = BaseManager()
            #self.manager.register('Counter',Counter,DictProxy)
            self.manager.register('Counter',Counter,DictProxy)
            self.manager.register('list',list, ListProxy)
            self.manager.register('Lock', Lock)
            self.manager.register('dict', dict, DictProxy)
            self.manager.register('set',set)
            self.manager.register('Queue', Queue)
            self.manager.start()

        self.tasks = self.manager.Queue()
        self.task_processors = {}
    
    def add_task_processor(self, taskType, processor):
        self.task_processors[taskType] = processor
    
    def start(self):
        """
        Spawn the worker processes

        """
        self.workers = [Worker(self.tasks, self.task_processors)
                        for _ in range(self.n_processes)]

    def add_task(self, task):
        self.tasks.put(task)

    def join(self):
        """
        Wait until the task queue is empty.
        """
        self.tasks.join()

    def finalize(self):
        """
        Wait until the task queue is empty and 
        stop the worker processes

        """
        # wait till the task queue is empty
        self.join()

        # tell all workers to stop
        for worker in self.workers:
            self.add_task(StopTask())

        # join the worker processes
        for worker in self.workers:
            worker.join()

# # EXAMPLE OF HOW TO USE THE WORKER POOL:

# INTERVAL = .1#.1

# class SnTask(object): pass

# class LineTask(object):
#     def __init__(self,l):
#         self.l = l

# class LineProcessor(object):
#     def __init__(self,tasks,manager):
#         self.tasks = tasks
#         self.lock = manager.Lock()
#         self.word_dict = manager.dict()

#     def do_task(self,task):
#         line = task.l
#         print('doing line in {0}'.format(current_process().name))
#         words = line.strip().split()
#         for w in words:
#             time.sleep(INTERVAL)
#             self.lock.acquire()
#             if not self.word_dict.has_key(w):
#                 self.word_dict[w] = True
#                 self.tasks.put(WordTask(w))
#             self.lock.release()


# class WordTask(object):
#     def __init__(self,w):
#         self.w = w

# class WordProcessor(object):
#     def __init__(self,tasks,manager):
#         self.tasks = tasks
#         self.lock = manager.Lock()
#         self.letter_dict = manager.dict()

#     def do_task(self,task):
#         word = task.w
#         print('doing word in {0}'.format(current_process().name))
#         letters = list(word)
#         for l in letters:
#             time.sleep(INTERVAL)
#             self.lock.acquire()
#             if not self.letter_dict.has_key(l):
#                 self.letter_dict[l] = True
#                 self.tasks.put(LetterTask(l))
#             self.lock.release()

# class LetterTask(object):
#     def __init__(self,l):
#         self.l = l

# class LetterProcessor(object):
#     def __init__(self,manager):
#         self.result = manager.list()

#     def do_task(self,task):
#         letter = task.l
#         print('doing letter in {0}'.format(current_process().name))
#         time.sleep(INTERVAL)
#         self.result.append(letter.upper())

# def main_new():
#     wp = WorkerPool()

#     letter_processor = LetterProcessor(wp.manager)
#     line_processor = LineProcessor(wp.tasks,wp.manager)
#     word_processor = WordProcessor(wp.tasks,wp.manager)

#     wp.add_task_processor(LineTask, line_processor)
#     wp.add_task_processor(WordTask, word_processor)
#     wp.add_task_processor(LetterTask, letter_processor)

#     wp.start()
    
#     data = ['a w rwn apwrna ls wn a wpna a a ',
#             '. w r .q qj qW Pmer P M m wan ; ',
#             ' fwer pw pw Wpap [a [a [',
#             ' w ww ; [sf w r q fd 3 14 sad er fwer pw pw pap [a [a [',
#             ' sf23 lk i m lk  m,  ls  amf ',
#             ' sf23 lk i masfp kaf adfm,  ls  amf s',
#             ' s3f23 k l lm lk l m,  sls  samf ',
#             'M w WR A P N M A Q'
#             ]

#     for l in data:
#         wp.add_task(LineTask(l))
#     wp.add_task(SnTask())
#     wp.finalize()
#     print(letter_processor.result)


# if __name__ == '__main__':
#     main_new()

