import os
import time


command = 'python -W ignore test.py --sr-scales multi --config-file config_files/dev_experiments/testing/{}.ini'

sota_methods = [
    'srgan',
    'edsr',
    'rdn',
    'srdensenet',
    'MDSR',
]

for m in sota_methods:
    os.system(command.format(m))
    time.sleep(20)

