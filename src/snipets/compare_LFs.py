from einops import reduce
from dataset_reader import read_LF
import os

# decoded = "/home/machado/Decoded_LFs/png/decoded_32_noPartition/"
# original = "/home/machado/Original_LFs/png/"
#
# for folder in os.listdir(decoded):
#     for lf_f in os.listdir(os.path.join(decoded, folder)):
#         for lf in os.listdir(os.path.join(decoded, folder, lf_f)):
#             if "75" in lf:
#                 path = os.path.join(decoded, folder, lf_f, 'bpp_0.75.png')
#                 # print(lf)
#                 lf_decoded = read_LF(path)
#                 path2 = os.path.join(original, folder, lf_f + '.mat.png')
#                 # print(path2)
#                 lf_original = read_LF(path2)
#                 diff = lf_decoded - lf_original
#                 squared_error = diff * diff  # Isso é um produto ponto a ponto
#                 # MSE_by_view = reduce(squared_error, 'c u v s t -> u v', 'mean')
#                 # Como a leitura é em tons de cinza, os 3 canais são iguais
#                 MSE = float(reduce(squared_error, 'c u v s t -> c', 'mean')[0])
#
#                 print(lf_f, ' ', MSE)


string_counts = {}
string_counts_val = {}




import json
with open("../chosen_list.txt", "r") as foldfile, open("outputFolds.txt", "w") as output:
    folds = json.loads(foldfile.read())
    for lf in folds[0][0]:
        if lf[0] in string_counts:
            # If the string is already in the dictionary, increment its count by 1
            string_counts[lf[0]] += 1
        else:
            # If the string is not in the dictionary, add it with a count of 1
            string_counts[lf[0]] = 1

        # Print the string counts
    for string, count in string_counts.items():
        print(f"{string}: {count}")
    print("\nValidation: \n")
#fold0 step 1 = validation
    for lf in folds[0][1]:
        if lf[0] in string_counts_val:
            # If the string is already in the dictionary, increment its count by 1
            string_counts_val[lf[0]] += 1
        else:
            # If the string is not in the dictionary, add it with a count of 1
            string_counts_val[lf[0]] = 1

        # Print the string counts
    for string, count in string_counts_val.items():
        print(f"{string}: {count}")
