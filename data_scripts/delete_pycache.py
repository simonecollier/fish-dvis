import os
import shutil

def delete_pycache_dirs(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if '__pycache__' in dirnames:
            pycache_path = os.path.join(dirpath, '__pycache__')
            print(f"Deleting: {pycache_path}")
            shutil.rmtree(pycache_path)
            # Remove from dirnames to prevent further walk into it
            dirnames.remove('__pycache__')

if __name__ == "__main__":
    # Change this to your target directory
    target_dir = "/home/simone/cloned_repos/DVIS_Plus"
    delete_pycache_dirs(target_dir)