from _typeshed import Self
from asyncio import FastChildWatcher
from audioop import reverse
from base64 import decode
from bisect import bisect, bisect_left
from cmath import inf, pi
import collections
from curses.ascii import isdigit
from functools import lru_cache
import heapq
from operator import le
from pydoc import Helper
from re import I, M
import turtle
from typing import Collection, List, Optional
from collections import *
import sys
import math
from matplotlib.cbook import flatten
from numpy import diff, sort
from regex import R
import random

from threading import Lock

class Foo:
    def __init__(self):
        self.locks = (Lock(), Lock())
        self.locks[0].acquire()
        self.locks[1].acquire()


    def first(self, printFirst: 'Callable[[], None]') -> None:
        
        # printFirst() outputs "first". Do not change or remove this line.
        printFirst()
        self.locks[0].release()


    def second(self, printSecond: 'Callable[[], None]') -> None:
        with self.locks[0]:

            # printSecond() outputs "second". Do not change or remove this line.
            printSecond()
            self.locks[1].release()


    def third(self, printThird: 'Callable[[], None]') -> None:
        with self.locks[1]:
            # printThird() outputs "third". Do not change or remove this line.
            printThird()