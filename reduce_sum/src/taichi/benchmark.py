from .reduce_sum import reduce_sum
import time

def benchmark(scale=list):
    results = []
    for i in scale:
        results.append(reduce_sum(i))

    return {"taichi":results}
