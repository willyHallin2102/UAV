"""
    tools / timer.py
    ----------------
    Timer instance to measure the performance of various methods, functions or 
    operations. It does operate by conduct a function `f` within the timer instance
    which encapsulate the function operation designated within it:
    
    with Timer("Encapsulate Timer Operation"):
        f()
    
    It does in addition support a versatile set of different measures
"""
from __future__ import annotations

import time

from contextlib import ContextDecorator
from typing import Optional



class Timer(ContextDecorator):
    """
    A lightweight high-resolution execution timer to evaluate performance of whatever
    operation encapsulated or wrapped by this instance.
    """

    def __init__(self,
        name: Optional[str] = None, show: bool = True, auto_format: bool = True
    ):
        """
            Initialize Timer Instance
        """

        self.clock = time.perf_counter
        self.name = name
        self.show = show
        self.auto_format = auto_format

        self.time_src: Optional[float] = None
        self.time_dst: Optional[float] = None
    

    def __enter__(self):
        """
        Start the clock, and return the instance itself, it initialize the clock
        whenever clock is referred to within with Timer('Log Message'):
        """
        self.time_src = self.clock()
        return self
    

    def __exit__(self, *args):
        """ 
        Stop timer measurement and optionally display the result if show is True. 
        """
        self.time_dst = self.clock() - self.time_src
        if self.show:
            print(self.result())
    

    def __str__(self) -> str:
        """ String representation of the timer. """
        return self.result()
    

    def __repr__(self) -> str:
        """ Return a representation of the timer. """
        name = f" name={self.name!r}" if self.name else ""
        time = f" time={self.time_dst:.6} s" if self.time_dst else ""

        return f"Timer({name}{time})"
    

    @property
    def time(self) -> float:
        """ Get elapsed time in seconds """
        return self.time_dst or 0.0
    
    
    @property
    def time_ms(self) -> float:
        """ Get elapsed time in milliseconds """
        return self.time_dst * 1000 or 0.0
    

    @property
    def time_us(self) -> float:
        """ Get elapsed time in microseconds """
        return self.time_dst * 1_000_000 or 0.0
    

    def result(self) -> str:
        """ Format the result with auto-formatting based on duration """
        if self.time_dst is None:
            print("Timer not started or not stopped")
        
        if self.auto_format:
            if self.time_dst >= 1.0: 
                t = f"{self.time_dst:.3f} s"
            
            elif self.time_dst >= 1e-3:
                t = f"{self.time_ms:.3f} ms"
            
            else:
                t = f"{self.time_us:.3} µs"
        
        else:
            t = f"{self.time_dst:.6f} S"

        # Add the name if provided
        prefix = f"{self.name}: " if self.name else ""
        return f"{prefix}{t}"
    

    def reset(self):
        self.time_src, self.time_dst = None, None




def timer(name: Optional[str] = None, print_result: bool = True) -> Timer:
    """Convenience function to create a timer"""
    return Timer(name=name, print_result=print_result)



def time_it(f):
    """Decorator to time a function"""
    def wrapper(*args, **kwargs):
        with Timer(name=f.__name__, print_result=True):
            return f(*args, **kwargs)
    
    return wrapper
