import dataset_reader
import numpy
import einops

def test(img, label):
    dataset_reader.write_LF_PMG(img.numpy(), "example")
    img2 = dataset_reader.read_LF("example.png")
    diff = img - img2
    MAX = (einops.reduce(diff, '... -> ', 'max'))
    MIN = (einops.reduce(diff, '... -> ', 'min'))
    print(f"{label}\t{MAX}\t{MIN}")

dataset = dataset_reader.training_dataset
for lf in dataset.lfs:
    img = dataset.read_original_mat(*lf)
    test(img, '_'.join(lf))

    bpps = dataset.bbps[lf[0], lf[1]]
    for bpp in bpps:
        (bpp, img) = dataset.read_decoded_mat(*lf, bpp)
        test(img, bpp)

