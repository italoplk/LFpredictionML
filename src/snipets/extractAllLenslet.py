import os
import multiviewExtractor as extractor
import sys

classe = sys.argv[1]
print(sys.argv[1])
lfr_path = "/home/machado/EPFLOriginal_LFRs/"
path = "/home/machado/"
# classe = "Buildings"

save_lensletGscale_path = os.path.join(path, "Lenslet_Gscale", classe)
save_mv_path = os.path.join(path, "MultiView_RGB", classe)
save_lensletRGB_path = os.path.join(path, "Lenslet_RGB", classe)

# for folder in os.listdir(lfr_path):
inner_path = os.path.join(lfr_path, classe)
os.makedirs(save_lensletGscale_path, exist_ok=True)
os.makedirs(save_mv_path, exist_ok=True)
os.makedirs(save_lensletRGB_path, exist_ok=True)
for lf in os.listdir(inner_path):
    if lf.split(".")[0] + ".png" not in os.listdir(save_lensletRGB_path):
        extractor.extract_lenslet(inner_path, save_lensletGscale_path, save_mv_path, save_lensletRGB_path, lf)
