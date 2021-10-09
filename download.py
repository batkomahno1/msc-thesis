import subprocess
from secrets import SERVER_NAME, REMOTE_LOCAL_PATHS, PROXY, DOWNLOADS_DIR, BKP_DIR
import os, shutil, time

REMOTE_ROOT_PATH = SERVER_NAME+':msc-thesis/'

# delete old folder
downloads_full_path = os.path.join(os.getcwd(), DOWNLOADS_DIR)
if os.path.exists(downloads_full_path):
    print('Deleting', downloads_full_path)
    shutil.rmtree(downloads_full_path)

# create dirs
for directory in [local for _,local in REMOTE_LOCAL_PATHS]:
    if not os.path.exists(directory):
        print('Creating dir', directory)
        os.makedirs(directory)

# download reporting data
print('Downloading all reports...')
for remote_path, local_path in REMOTE_LOCAL_PATHS:
    # sleep to prevent flooding!
    time.sleep(1)
    proc = subprocess.run(["scp", "-r"] + PROXY + [REMOTE_ROOT_PATH+remote_path, local_path])
    proc.check_returncode()

# backup old downloads
print('Backing up', downloads_full_path)
now = str(round(time.time()))
if not os.path.exists(BKP_DIR): os.makedirs(BKP_DIR)
output_filename = os.path.join(os.getcwd(), BKP_DIR + 'bkp_' + now)
shutil.make_archive(output_filename, 'zip', downloads_full_path)

print('Done.')
