import os


def create_or_clean_existing(folder_path):
    if os.path.exists(folder_path):
        print(f"Folder {folder_path} already exists. Typing YES will CLEAN everything, typing anything else aborts procedure.")
        i = input()
        
        if i != "YES":
            raise Exception(f'Aborted; typed {i} != "YES" into prompt')
    else:
        os.mkdir(folder_path)
        
    # Clean everything in folder
    for f in os.listdir(folder_path):
        try:
            os.remove(folder_path + f)
        except:
            print(f'Failed to remove file {f}: Is A Directory')