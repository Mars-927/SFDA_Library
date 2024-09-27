
import os
import random
def get_label(path):
    label_list = ["agricultural","baseball_diamond","beach","dense_residential","forest","medium_residential","parking_lot","river","sparse_residential","storage_tank"]
    for index,item in enumerate(label_list):
        if item in path:
            return index

# get 'directory' folder all images filename
def write_file_paths(directory):
    files_dict = dict()
    for root, dirs, files in os.walk(directory):
        for file_name in files:
            file_path = os.path.join(os.path.basename(root), file_name)
            index = get_label(file_path)
            if index not in files_dict.keys():
                files_dict[index] = []
            files_dict[index].append(file_path)
    return files_dict

def divide(files_dict,train = 80,test = 20):
    select_train = dict()
    select_test = dict()
    select_all = dict()
    for key in list(files_dict.keys()):
        random.shuffle(files_dict[key])
        select_train[key] = files_dict[key][:train]
        select_test[key] = files_dict[key][train:train + test]
        select_all[key] = files_dict[key][:train + test]
    return select_train,select_test,select_all


def write(root_path,dataset,train,test,all):
    files_dataset = root_path + ".txt"
    content_str = ""
    for key,value in dataset.items():
        for item in value:
            content_str += f"{item} {key}\n"
    with open( files_dataset , "w" , encoding="utf-8" ) as f:
        f.write(content_str)

   
dataset = "UCMerced_LandUse"
select_dataset = write_file_paths(f'F:/HANS/!dataset/RS_DomainAdaptation_AIDUCMNWPU/{dataset}')
select_train,select_test,select_all = divide(select_dataset)
write(f"dataset/AID_NWPU_UCM/{dataset}",select_dataset,select_train,select_test,select_all)


# AID
# NWPU-RESISC45
# UCMerced_LandUse