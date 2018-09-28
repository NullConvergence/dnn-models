from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import re
import sys
import tarfile
from six.moves import urllib

class Data(object):
    def __init__(self, folder, url):
        self.download_extract(folder, url)
    
    def download_extract(self, dest_folder, url):
        """
            Download and extract ImageNet database
        """
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)
        
        tar_file = url.split('/')[-1]
        tar_path = os.path.join(dest_folder, tar_file)

        if not os.path.exists(tar_path):
            tar_path, _ = urllib.request.urlretrieve(url, tar_path, self.progress)
            print()
            stats = os.stat(tar_path)
            print('Successfuly downloaded', stats.st_size, 'bytes.')
        
        tarfile.open(tar_path, 'r:gz').extractall(dest_folder)
        
    
    def progress(self, count, block_size, total_size):
        """ Courtesy of TensorFlow - https://github.com/tensorflow """
        sys.stdout.write('\r>> Downloading %.1f%%' % (float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()
