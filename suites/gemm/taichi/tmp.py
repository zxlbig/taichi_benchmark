import taichi as ti

@ti.kernel
def use_vec4():
    vec = ti.math.vec4()
    vec = [4,4,4,4]
    
