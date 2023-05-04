import os
import json
from contextlib import contextmanager

from subprocess import Popen, PIPE

@contextmanager
def pushd(path):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)

def run_binary(binary_file, argv):
    p = Popen(['./' + binary_file] + argv, stdout=PIPE)
    output, err = p.communicate()
    output = output.decode('utf-8')
    output = output.split('\n')
    print(output)
    results = []
    for line in output[:-1]:
        res_dict = None
        try:
            res_dict = json.loads(line)
        except:
            pass
        if res_dict:
            results.append(res_dict)
    return results

def compile_and_benchmark(source_name, flags=[]):
    workdir = os.path.dirname(os.path.abspath(__file__))

    with pushd(workdir):
        # Compile
        p = Popen(['make'] + flags, stdout=PIPE)
        output, err = p.communicate()
        rc = p.returncode
        if rc != 0:
            raise Exception("Cannot compile gemm")
        print("Successfully compiled gemm")
        output_binary_name = "sgemm_gpu"
        # Run Benchmark
        results = []
        if source_name == 'cublas':
            argv = ["0"]
            results += run_binary(output_binary_name, argv)
            print("{} test finished.".format(source_name))

        elif source_name == "handwritten_native":
            argv = ["2"]
            results += run_binary(output_binary_name, argv)
            print("{} test finished.".format(source_name))

        elif source_name == "handwritten_opt":
            argv = ["11"]
            results += run_binary(output_binary_name, argv)
            print("{} test finished.".format(source_name))
        return results

def benchmark():
    return {"cublas": compile_and_benchmark("cublas"), "handwritten_native": compile_and_benchmark("handwritten_native"),
            "handwritten_opt": compile_and_benchmark("handwritten_opt")}

if __name__ == '__main__':
    print(benchmark())
