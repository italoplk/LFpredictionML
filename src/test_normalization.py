import dataset_reader
import numpy
import einops
img = dataset_reader.read_LF("/scratch/Original_LFs/png/Urban/Bench_in_Paris.mat.png")
dataset_reader.write_LF_PMG(img.numpy(), "Bench_in_Paris.mat")
img2 = dataset_reader.read_LF("Bench_in_Paris.mat.png")

diff = img - img2
print(einops.reduce(diff, '... -> ', 'max'))
print(einops.reduce(diff, '... -> ', 'min'))