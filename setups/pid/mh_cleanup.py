import os 
import shutil 
from glob import glob

folder = 'output'
for f in glob(os.path.join(folder, '*.txt')):
    os.remove(f)
    print('Removed file: {}'.format(f))

for f in glob(os.path.join(folder, '*')):
    if os.path.isdir(f):
        shutil.rmtree(f)
        print('Removed folder: {}'.format(f))