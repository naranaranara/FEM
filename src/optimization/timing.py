from time import perf_counter

class PhaseTimer:
    def __init__(self):
        self.seconds={};self.counts={}
    def record(self,name,start):
        self.seconds[name]=self.seconds.get(name,0.)+perf_counter()-start
        self.counts[name]=self.counts.get(name,0)+1
