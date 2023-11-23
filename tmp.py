import shutil
import os

train_path = './data/HockeyFights/train/'
valid_path = './data/HockeyFights/validation/'
test_path = './data/HockeyFights/test/'


for folder_path in [train_path, valid_path, test_path]:
    file_list = os.listdir(folder_path)

    # make two new folders
    os.mkdir(folder_path+"fight")
    os.mkdir(folder_path+"no_fight")

    # loop through all files
    for i in range(len(file_list)):
        # if "fi" in train_list[i] move to "fight folder"
        if "avi" not in file_list[i]:
            continue
        if "fi" in file_list[i]:
            shutil.move(folder_path+file_list[i].rstrip(), folder_path+"fight/"+file_list[i].rstrip())
        else: # move to "no fight folder"
            shutil.move(folder_path+file_list[i].rstrip(), folder_path+"no_fight/"+file_list[i].rstrip())
