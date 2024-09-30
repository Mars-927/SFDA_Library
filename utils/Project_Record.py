import datetime
import os
import shutil
import sys


class Project():
    is_init = False
    root_path = None
    folders = dict()

    def __init__(self,project_name = None):
        project_root = os.path.dirname(sys.argv[0])
        process_folder_path = os.path.join(project_root, "process")
        if not os.path.exists(process_folder_path):
            self.__create_folder(process_folder_path)
        
        if project_name is None:
            info_folder_path = os.path.join(process_folder_path,f"process-{str(int(datetime.datetime.now().timestamp() * 1000// 1))}")
        else:
            info_folder_path = os.path.join(process_folder_path,project_name)

        if os.path.exists(info_folder_path):
            input("\nexise log folder, press everything to clear it...")
            shutil.rmtree(info_folder_path)
        self.__create_folder(info_folder_path)


        Project.root_path = info_folder_path
        Project.folders['images'] = os.path.join(info_folder_path, 'images')
        Project.folders['result'] = os.path.join(info_folder_path, 'result')
        Project.folders['root'] = info_folder_path
        self.__create_folder(Project.folders['images'])
        self.__create_folder(Project.folders['result'])
        Project.is_init = True
        Project.global_save = dict()

    def __create_folder(self, folder_path):
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

    @classmethod
    def log(cls, message, filename = "run_log.txt",is_print=True):
        assert Project, "not initial class:Project"
        if ".txt" not in filename:
            filename = filename + ".txt"
        meg = f"【{datetime.datetime.now().strftime('%Y/%m/%d-%H:%M:%S')}】{message}"
        now_filepath = os.path.join(Project.root_path, filename)
        with open(now_filepath, 'a', encoding='utf-8') as file:
            file.write(meg + '\n')
        if is_print:
            print(message)

    @classmethod
    def create_folder(cls,name):
        # create folder
        assert Project, "not initial class:Project"
        assert name not in Project.folders.keys(), "folder exist!"
        Project.folders[name] = os.path.join(Project.root_path, name)
        if not os.path.exists(Project.folders[name]):
            os.mkdir(Project.folders[name])

    @classmethod
    def get_folder(cls,folder):
        # get folder path
        assert Project, "not initial class:Project"
        if folder not in Project.folders.keys():
            Project.create_folder(folder)
        return Project.folders[folder]

    @classmethod
    def get_item_path(cls,filename, folder=None):
        # get item path, return: ./process/info-xxxx/images
        assert Project, "not initial class:Project"
        assert "." in filename, "filename not tail"
        if folder is None:
            return os.path.join(Project.folders['images'],filename)
        else:
            if folder not in Project.folders.keys():
                Project.create_folder(folder)
            return os.path.join(Project.folders[folder], filename)
        


# backup code
def backup_code(source_folder, destination_folder):
    os.makedirs(destination_folder, exist_ok=True)
    items = os.listdir(source_folder)
    for item in items:
        if item in ['process','dataset','history','cpkg','README_About']:
            continue
        item_path = os.path.join(source_folder, item)
        if os.path.isdir(item_path):
            new_destination_folder = os.path.join(destination_folder, item)
            os.makedirs(new_destination_folder, exist_ok=True)
            backup_code(item_path, new_destination_folder)
        elif item.endswith('.py'):
            destination_path = os.path.join(destination_folder, item)
            shutil.copy(item_path, destination_path)



