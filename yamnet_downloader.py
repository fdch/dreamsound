###############################################################################
#
#  This file is part of the dreamsound package.
#  
#  YamnetDownloader class definition
#
###############################################################################

import os
import requests
import pathlib

class YamnetDownloader(object):

    url  = "https://raw.githubusercontent.com/fdch/models/master/research/audioset/yamnet/"
    weights = "https://storage.googleapis.com/audioset/yamnet.h5"
    files = [ 
            "export.py",
            "features.py",
            "inference.py",
            "params.py",
            "yamnet.py",
            "yamnet_class_map.csv",
            "yamnet_test.py",
            "yamnet_visualization.ipynb",
            ]

    def __init__(self, path=None):
        
        if path is not None:
            self.dir = path
        else:
            self.dir = "."
        
        self.path = str(pathlib.Path(self.dir).parent.absolute())

    def __call__(self):

        for p in self.files:
            self.__download__(self.url + p, self.path + "/" + p)

        w = self.weights.split("/")[-1]
        self.__download__(self.weights, self.path + "/" + w)

    def __download__(self, url, path):
        if not os.path.exists(path):
            print(f"Downloading {url} into {path} ...")
            try:
                r = requests.get(url, allow_redirects=True)
                open(path, 'wb').write(r.content)
            except Exception as e: print(e)

# end YamnetDownloader class
if __name__ == "__main__":
    yn = YamnetDownloader()()