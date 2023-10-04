import os
import random as rd



directory_path ='/scratch/Decoded_LFs/png/decoded_32_noPartition/'
outPathTrain = '/scratch/train/'
outPathValidation = '/scratch/validation/'

os.system('python3 --version')

for folder in os.listdir(directory_path):
    for innerFolder in os.listdir(directory_path +'/' + folder):

        rand = rd.randint(1, 100)

        for file in os.listdir(os.path.join(directory_path, folder, innerFolder)):

            if rand < 75:
                os.system('cp ' + os.path.join( directory_path,  folder, innerFolder, file) + ' ' + os.path.join(outPathTrain, folder+'_'+innerFolder + '_' +file))
            else:
                os.system('cp ' + os.path.join(directory_path, folder, innerFolder, file) + ' ' + os.path.join(outPathValidation, folder+'_'+innerFolder + '_' +file))

        '''if '&' in file:
            oldName = (directory_path + folder + '/' + file)
            newName = (directory_path+folder+'/'+file.replace('&', '_'))
            print(newName)
            os.rename(oldName, newName)'''
