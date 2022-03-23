import matplotlib.pyplot as plt
import sys
import os

cuda_sample_results = {'cuda_2d': [{'n_particles': 512, 'time_ms': 1.024374}, {'n_particles': 2048, 'time_ms': 1.033906}, {'n_particles': 4608, 'time_ms': 1.13101}, {'n_particles': 8192, 'time_ms': 1.24747}, {'n_particles': 12800, 'time_ms': 1.484127}, {'n_particles': 18432, 'time_ms': 1.791808}, {'n_particles': 25088, 'time_ms': 2.105247}, {'n_particles': 32768, 'time_ms': 2.370462}], 'cuda_3d': [{'n_particles': 8192, 'time_ms': 2.70262}, {'n_particles': 65536, 'time_ms': 8.443884}, {'n_particles': 221184, 'time_ms': 36.317848}, {'n_particles': 524288, 'time_ms': 174.325989}, {'n_particles': 1024000, 'time_ms': 565.163757}, {'n_particles': 1769472, 'time_ms': 1228.022095}, {'n_particles': 2809856, 'time_ms': 2252.36377}, {'n_particles': 4194304, 'time_ms': 3472.967773}]}

#cuda_sample_results = {'cuda_2d': [{'n_particles': 128, 'time_ms': 0.989771}, {'n_particles': 1152, 'time_ms': 1.019925}, {'n_particles': 3200, 'time_ms': 1.043498}, {'n_particles': 6272, 'time_ms': 1.144777}, {'n_particles': 10368, 'time_ms': 1.325348}, {'n_particles': 15488, 'time_ms': 1.604376}, {'n_particles': 21632, 'time_ms': 1.780423}, {'n_particles': 28800, 'time_ms': 2.219935}, {'n_particles': 36992, 'time_ms': 2.624425}], 'cuda_3d': [{'n_particles': 1024, 'time_ms': 2.175864}, {'n_particles': 27648, 'time_ms': 4.838743}, {'n_particles': 128000, 'time_ms': 17.925041}, {'n_particles': 351232, 'time_ms': 82.931763}, {'n_particles': 746496, 'time_ms': 332.263519}, {'n_particles': 1362944, 'time_ms': 900.967163}, {'n_particles': 2249728, 'time_ms': 2020.989258}, {'n_particles': 3456000, 'time_ms': 2922.075928}, {'n_particles': 5030912, 'time_ms': 4490.73584}]}

taichi_sample_results = {'taichi_2d': [{'n_particles': 512, 'time_ms': 0.4708201557548364}, {'n_particles': 2048, 'time_ms': 0.4732567070391269}, {'n_particles': 4608, 'time_ms': 0.5115689839101378}, {'n_particles': 8192, 'time_ms': 0.5668949448249805}, {'n_particles': 12800, 'time_ms': 0.6931407124000089}, {'n_particles': 18432, 'time_ms': 0.8494079414163025}, {'n_particles': 25088, 'time_ms': 1.0304002558427783}, {'n_particles': 32768, 'time_ms': 1.2111213139576193}], 'taichi_3d': [{'n_particles': 8192, 'time_ms': 1.9398335703044722}, {'n_particles': 65536, 'time_ms': 10.701219556153774}, {'n_particles': 221184, 'time_ms': 34.867725920889825}, {'n_particles': 524288, 'time_ms': 140.94088707861374}, {'n_particles': 1024000, 'time_ms': 549.7929778569244}, {'n_particles': 1769472, 'time_ms': 1286.6532287104633}, {'n_particles': 2809856, 'time_ms': 2374.8291689565235}, {'n_particles': 4194304, 'time_ms': 4005.1647448022436}]}

#taichi_sample_results = {'taichi_2d': [{'n_particles': 128, 'time_ms': 0.48161188865947224}, {'n_particles': 1152, 'time_ms': 0.4853399545936554}, {'n_particles': 3200, 'time_ms': 0.49222280321714607}, {'n_particles': 6272, 'time_ms': 0.5224119828994844}, {'n_particles': 10368, 'time_ms': 0.6324768173726625}, {'n_particles': 15488, 'time_ms': 0.758077704574589}, {'n_particles': 21632, 'time_ms': 0.9206238256922461}, {'n_particles': 28800, 'time_ms': 1.1050803203147552}, {'n_particles': 36992, 'time_ms': 1.347265234386441}], 'taichi_3d': [{'n_particles': 1024, 'time_ms': 0.870358452147002}, {'n_particles': 27648, 'time_ms': 5.206501572274647}, {'n_particles': 128000, 'time_ms': 21.61489226222102}, {'n_particles': 351232, 'time_ms': 65.09008040575281}, {'n_particles': 746496, 'time_ms': 337.4440998095736}, {'n_particles': 1362944, 'time_ms': 863.3175398461788}, {'n_particles': 2249728, 'time_ms': 1877.4295391049804}, {'n_particles': 3456000, 'time_ms': 3140.0475489834034}, {'n_particles': 5030912, 'time_ms': 5052.8972775903185}]}

def extract_perf(results):
    perf = []
    for record in results:
        perf.append(record["time_ms"])
    return perf

def extract_particles(results):
    particles = []
    for record in results:
        particles.append(record["n_particles"])
    return particles 

def plot_bar(cuda_results, taichi_results):
    fig, ax = plt.subplots(figsize=(12,9))
    plot_series = "3d"

    x_cuda = extract_particles(cuda_results["cuda_" + plot_series])
    y_cuda = extract_perf(cuda_results["cuda_" + plot_series])
    bar_pos = [i*3 for i in range(len(x_cuda))]
    ax.bar(bar_pos, y_cuda)

    x_taichi = extract_particles(taichi_results["taichi_" + plot_series])
    y_taichi = extract_perf(taichi_results["taichi_" + plot_series])
    bar_pos = [i*3+1 for i in range(len(x_taichi))]
    ax.bar(bar_pos, y_taichi)

    labels = ["{}".format(i) for i in x_cuda]
    ax.set_xticks(bar_pos, labels, rotation = 30)
    
    if plot_series == "3d":
        plt.yscale("log")  
    plt.grid('minor', axis='y')
    plt.xlabel("#Particles")
    plt.ylabel("Time per Frame (ms)")
    plt.legend(["CUDA", "Taichi"], loc='upper left')
    plt.title("MPM(" + plot_series.upper() + ") benchmark")
    plt.savefig("fig/bench_" + plot_series + ".png", dpi=150)

if __name__ == '__main__':
    try:
        os.makedirs('fig')
    except FileExistsError:
        pass
    cuda_results = cuda_sample_results
    taichi_results = taichi_sample_results
    plot_bar(cuda_results, taichi_results)
