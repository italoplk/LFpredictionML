import os
import multiviewExtractor as extractor
import sys
import threading
import time


def extract_all(classe):

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
        if lf.split(".")[0] + ".png" not in os.listdir(save_lensletRGB_path) and len(lf.split(".")) == 2:
            extractor.extract_lenslet(inner_path, save_lensletGscale_path, save_mv_path, save_lensletRGB_path, lf)
    print("finished ", classe)


extract_all(sys.argv[1])

#
# thread1 = threading.Thread(target=extract_all, args=("Grids",))
# thread2 = threading.Thread(target=extract_all, args=("ISO_and_Colour_Charts",))
# thread3 = threading.Thread(target=extract_all, args=("Landscapes",))
# thread4 = threading.Thread(target=extract_all, args=("Light",))
# thread5 = threading.Thread(target=extract_all, args=("Nature",))
# thread6 = threading.Thread(target=extract_all, args=("People",))
# thread7 = threading.Thread(target=extract_all, args=("Studio",))
# thread8 = threading.Thread(target=extract_all, args=("Urban",))
#
# # Start the threads
# thread1.start()
# thread2.start()
# thread3.start()
# thread4.start()
# thread5.start()
# thread6.start()
# thread7.start()
# thread8.start()
# # Wait for all threads to finish
# thread1.join()
# thread2.join()
# thread3.join()
# thread4.join()
# thread5.join()
# thread6.join()
# thread7.join()
# thread8.join()
