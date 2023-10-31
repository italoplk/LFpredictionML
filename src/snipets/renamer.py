import os
import random as rd



directory_path ='/home/machado/Decoded_LFs/png/decoded_32_noPartition/'
outPathTrain = '/scratch/train/'
outPathValidation = '/scratch/validation/'

os.system('python3 --version')

for file in os.listdir(outPathValidation):
    if '0.75' not in file:
        print(file)
        os.system('rm '+ os.path.join(outPathValidation,file))

for file in os.listdir(outPathTrain):
    if '0.75' not in file:
        print(file)
        os.system('rm ' + os.path.join(outPathTrain, file))

# for folder in os.listdir(directory_path):
#     for innerFolder in os.listdir(directory_path + '/' + folder):
#
#         rand = rd.randint(1, 100)
#         for file in os.listdir(os.path.join(directory_path, folder, innerFolder)):
#             if rand < 75:
#                 os.system('cp ' + os.path.join(directory_path, folder, innerFolder, file) + ' ' + os.path.join(outPathTrain,
#                                                                                                                folder + '_' + innerFolder + '_' + file))
#             else:
#                 os.system(
#                     'cp ' + os.path.join(directory_path, folder, innerFolder, file) + ' ' + os.path.join(outPathValidation,
#                                                                                                          folder + '_' + innerFolder + '_' + file))


    '''os.system('mkdir /home/machado/HBPP/'+folder)
    
        os.system('mkdir ' + os.path.join('/home/machado/HBPP/' , folder, innerFolder))
        for file in os.listdir(os.path.join(directory_path, folder, innerFolder)):
            if '0.75' in file:
                os.system('cp ' + os.path.join(directory_path, folder, innerFolder, file) + ' '
                          + os.path.join('/home/machado/HBPP/', folder, innerFolder))

        '''

    ''' RENAMER
        if '&' in file:
            oldName = (directory_path + folder + '/' + file)
            newName = (directory_path+folder+'/'+file.replace('&', '_'))
            print(newName)
            os.rename(oldName, newName)'''
