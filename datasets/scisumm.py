import os, glob
import logging
import numpy as np
import xml.etree.ElementTree as ET

import torch.utils.data as data

from . import transforms

class ConcatDataset(data.Dataset):
    def __init__(self, datasets):
        _datasets = []
        for d in datasets:
            _datasets.append(d.data)
        self.data = np.concatenate(_datasets)
    
    def __getitem__(self, index):
        input, target = self.data[index]
        return input, target
    
    def __len__(self):
        return len(self.data)
    
    def sample(self, n, seed):
        if n > 0:
            idx = np.random.RandomState(seed).randint(len(self.data), size=n)
            self.data = self.data[idx]
    
    def split_xy(self):
        x, y = zip(*self.data)
        y = np.array(y)
        return x, y
        

class Scisumm(data.Dataset):
    
    base_dir = './datasets/scisummnet_release1.1__20190413/top1000_complete'
    
    def __init__(self, args, transforms=None):
        self.data = self._get_data(args)
        if transforms is not None:
            self._build(transforms)
            
    def _build(self, transforms):
        for t in transforms:
            self.data = t(self.data)
        
    def _get_data(self, args):
        folders = sorted(glob.glob('{}/*'.format(self.base_dir)))
        if not folders:
            raise SystemExit("Couldn't find Scisumm dataset")
        data = self._read_files(args, folders)
        return data
        
    def _read_files(self, args, folders):
        papers = []
        scores = []
        for folder in folders:
            paper_id = folder.split('/')[-1]
            file_path = os.path.join(folder, 'Documents_xml', paper_id+'.xml')
            root = ET.parse(file_path).getroot()
            content = []
            absract = root.find('ABSTRACT')
            if absract is not None:
                for s in absract.findall('S'):
                    content.append(s.text)
                for section in root.findall('SECTION'):
                    for s in section.findall('S'):
                        content.append(s.text)
            else:
                for s in root.findall('S'):
                    content.append(s.text)
            try:
                papers.append(' '.join(content))
                scores.append([0]*len(args.aspects))
            except:
                pass
        assert len(papers) == len(scores)
        data = np.array(list(zip(papers, scores)), dtype=object)
        return data
    
    def __getitem__(self, index):
        input, target = self.data[index]
        return input, target
    
    def __len__(self):
        return len(self.data)

        
def get_unlabeled_scisumm(args, tokenizer=None):    
    
    if tokenizer is not None:
        unlabeled_dataset = Scisumm(
            args, transforms=[transforms.Tokenize(tokenizer, args.max_length)])
    else:
        unlabeled_dataset = Scisumm(args)
    
    logging.info('Scisumm unlabeled: {}'.format(len(unlabeled_dataset)))
    
    return unlabeled_dataset