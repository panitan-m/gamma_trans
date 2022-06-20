import os, glob, re
import logging
import math
import numpy as np
from collections import defaultdict
from gensim.models import KeyedVectors
from sklearn.metrics import mean_squared_error

import torch
import torch.utils.data as data

from transformers import AutoTokenizer
from . import transforms

from .parsers.Paper import Paper
from .parsers.ScienceParseReader import ScienceParseReader
from .parsers.PaperSection import PaperSection

from models.metrics import METRICS

logger = logging.getLogger(__name__)

class PeerRead(data.Dataset):

    base_dir = './datasets/PeerRead/data/'

    def __init__(self, args,
                 split=None,
                 indexs=None,
                 unlabeled=False,
                 balance=False,
                 transforms=None):
        
        self.loss_fn = args.loss_fn
        self.score_type = 'major' if args.loss_fn == 'cross_entropy' else 'mean'
        self.split = split
        self.balance = balance
        self.data, self.paper_ids = self._get_original_split(args, split, unlabeled) if split else self._get_raw_data(args, unlabeled)
        if indexs is not None:
            self.data = self.data[indexs]
            self.paper_ids = self.paper_ids[indexs]
        if transforms is not None:
            self._build(transforms)
            
    def _build(self, transforms):
        for t in transforms:
            self.data = t(self.data)
        
    def __getitem__(self, index):
        input, target = self.data[index]
        return input, target
    
    def __len__(self):
        return len(self.data)
    
    def split_xy(self):
        x, y = zip(*self.data)
        y = np.array(y)
        return x, y
    
    def aspects_to_binary(self, threshold=4):
        for i in range(len(self.data)):
            x, y = self.data[i]
            y  = [0 if a < 4 else 1 for a in y]
            self.data[i] = x, y
    
    def set_by_idxs(self, idxs):
        self.data = self.data[idxs]
    
    def get_stats(self):
        if self.split_sentence:
            num_sentences = [len(x[0]) for x in self.data]
            lengths = [len(x.split(' ')) for y in self.data for x in y[0]]
            return num_sentences, lengths
        else:
            lengths = [len(x[0].split(' ')) for x in self.data]
            return lengths
        
    def _read_json_files(self, paper_json_filenames, scienceparse_dir=None):
        papers = []
        paper_ids = []
        
        for paper_json_filename in paper_json_filenames:
            
            paper = Paper.from_json(paper_json_filename)
            if scienceparse_dir is None:
                scienceparse_dir = os.path.dirname(paper_json_filename)+'/'
            paper.SCIENCEPARSE = ScienceParseReader.read_science_parse(paper.ID, paper.TITLE, paper.ABSTRACT, scienceparse_dir)
            paper_section = PaperSection.from_paper(paper)
            papers.append(paper_section)
            paper_ids.append('{}'.format(paper.ID))
            
        return papers, paper_ids
    
    def _clean_text(self, input):
        cleaned = input.strip()
        cleaned = re.sub("\n([0-9]*( *\n))+", "\n", cleaned)
        return cleaned
    
    def _get_section_key(self, section):
        return section[0].upper() + section[1:].replace('_', ' ')
    
    def _append_data(self, list, item):
        try:
            list.append(self._clean_text(item))
        except:
            list.append([self._clean_text(s) for s in item])
        return list
        
    
    def _get_raw_data(self, args, unlabeled=False):
        
        papers = []
        paper_ids = []
        
        datasets = args.unlabeled_datasets if unlabeled else args.datasets
        
        for dataset in datasets:
            if all(s in os.listdir(os.path.join(self.base_dir, dataset)) for s in ['train', 'dev', 'test']):
                for split in ['train', 'dev', 'test']:
                    review_dir = os.path.join(self.base_dir, dataset, split, 'reviews/')
                    scienceparse_dir = os.path.join(self.base_dir, dataset, split, 'parsed_pdfs/') 
                    paper_json_filenames = sorted(glob.glob('{}/*.json'.format(review_dir)))
                    _papers, _paper_ids = self._read_json_files(paper_json_filenames, scienceparse_dir)
                    papers.extend(_papers)
                    paper_ids.extend(_paper_ids)
            else:
                scienceparse_dir = os.path.join(self.base_dir, dataset, 'parsed_pdfs/')
                paper_json_filenames = sorted(glob.glob('{}/*.json'.format(scienceparse_dir)))
                _papers, _paper_ids = self._read_json_files(paper_json_filenames)
                papers.extend(_papers)
                paper_ids.extend(_paper_ids)  

        ids = []
        raw_data = []
        labels = []
        data = []

        for id, paper in zip(paper_ids, papers):
            if args.section == 'all':
                content = paper.SCIENCEPARSE.get_paper_content()
            else:
                content = paper.SECTION_SENTENCES[self._get_section_key(args.section)]
                
            if content and len(content) > 1:
                if args.section != 'all':
                    if self.split_sentence:
                        content = paper.SECTION_SENTENCES[self._get_section_key(args.section)]
                    else:
                        content = paper.SECTIONS[self._get_section_key(args.section)]
                score = []
                for aspect in args.aspects:
                    try:
                        s = paper.SCORE[aspect.upper()][self.score_type]
                        s = 0 if s < 4 else 1
                        score.append(s)
                    except:
                        score = None
                        continue
                if content:
                    if score:
                        ids.append(id)
                        self._append_data(raw_data, content)
                        labels.append(score)
                    elif unlabeled:
                        ids.append(id)
                        self._append_data(raw_data, content)
                        labels.append([0]*len(args.aspects))
                        
        

        if len(raw_data) > 0:
            raw_data = np.array(raw_data, dtype=object)
            if self.loss_fn == 'cross_entropy':
                labels = np.array(labels) - 1
            else:
                # labels = (np.array(labels) - 1)/4 if not unlabeled else np.array(labels)
                # assert ((labels >= 0) & (labels <=1)).all()
                labels = np.array(labels, dtype=np.float32)
                #balance
                if self.balance:
                    label_0_idx = np.where(labels == 0)[0]
                    label_1_idx = np.where(labels == 1)[0]
                    n_least = min(len(label_0_idx), len(label_1_idx))
                    label_0_idx = np.random.RandomState(seed=args.seed).permutation(label_0_idx)[:n_least]
                    label_1_idx = np.random.RandomState(seed=args.seed).permutation(label_1_idx)[:n_least]
                    idx = np.concatenate([label_0_idx, label_1_idx])
                    idx = np.sort(idx)
            if self.balance:
                data = np.array(list(zip(raw_data[idx], labels[idx])), dtype=object)
                ids = np.array(ids)[idx]
            else:
                data = np.array(list(zip(raw_data, labels)), dtype=object)
                ids = np.array(ids)
        else:
            raise SystemExit('No data')
        
        assert len(data) == len(ids)

        return data, ids
    
    def _get_original_split(self, args, split='train', unlabeled=False):
        
        papers = []
        paper_ids = []
        
        for dataset in args.datasets:
            if split in os.listdir(os.path.join(self.base_dir, dataset)):
                review_dir = os.path.join(self.base_dir, dataset, split, 'reviews/')
                scienceparse_dir = os.path.join(self.base_dir, dataset, split, 'parsed_pdfs/') 
                paper_json_filenames = sorted(glob.glob('{}/*.json'.format(review_dir)))
                _papers, _paper_ids = self._read_json_files(paper_json_filenames, scienceparse_dir)
                papers.extend(_papers)
                paper_ids.extend(_paper_ids)
            else:
                raise SystemExit("{} doesn't have {} set", dataset, split)
    
        ids = []
        data = []
        raw_data = []
        labels = []
            
        for id, paper in zip(paper_ids, papers):
            if args.section == 'all':
                content = paper.SCIENCEPARSE.get_paper_content()
            else:
                content = paper.SECTION_SENTENCES[self._get_section_key(args.section)]
                
            if content and len(content) > 1:
                if args.section != 'all':
                    if self.split_sentence:
                        content = paper.SECTION_SENTENCES[self._get_section_key(args.section)]
                    else:
                        content = paper.SECTIONS[self._get_section_key(args.section)]
                score = []
                for aspect in args.aspects:
                    try:
                        s = paper.SCORE[aspect.upper()][self.score_type]
                        score.append(s)
                    except:
                        score = None
                        continue
                if content:
                    if score:
                        ids.append(id)
                        self._append_data(raw_data, content)
                        labels.append(score)
                    elif unlabeled:
                        ids.append(id)
                        self._append_data(raw_data, content)
                        labels.append(0)
                        
        if len(raw_data) > 0:
            raw_data = np.array(raw_data, dtype=object)
            if self.loss_fn == 'cross_entropy':
                labels = np.array(labels) - 1
            else:
                # labels = (np.array(labels) - 1)/4 if not unlabeled else np.array(labels)
                # assert ((labels >= 0) & (labels <=1)).all()
                labels = np.array(labels, dtype=np.float32)
                
            data = np.array(list(zip(raw_data, labels)), dtype=object)
        else:
            raise SystemExit('No data in {} set', split)
                        
        ids = np.array(ids)
        
        return data, ids
    
def evaluate(y, y_):
    return math.sqrt(mean_squared_error(y, y_))
    
def evaluate_mean(args, dataset):
    evaluate_mean = defaultdict(list)
    _, y = dataset.split_xy()
    y = torch.tensor(y)
    logging.info('\t%13s\t%6s\t%6s\t%7s\t%7s'%('Mean (Test)', 'RMSE', 'MAE', 'SPR', 'PRS'))
    for aid, (a, y_aspect) in enumerate(zip(args.aspects, y.T)):
        mean_aspect = torch.mean(y_aspect)
        evaluate_mean_aspect = {}
        for m in args.eval_metrics:
            evaluate_mean_aspect[m] = METRICS[m](y_aspect, torch.tensor([mean_aspect]*len(y_aspect)))
            evaluate_mean[m].append(evaluate_mean_aspect[m])
        logging.info('\t%13s\t%.4f\t%.4f\t%7.4f\t%7.4f'%(
            a.upper(),
            evaluate_mean_aspect['rmse'],
            evaluate_mean_aspect['mae'],
            evaluate_mean_aspect['spr'],
            evaluate_mean_aspect['prs']))
    logging.info('\t%13s\t%.4f\t%.4f\t%7.4f\t%7.4f'%(
        'TOTAL',
        np.average(evaluate_mean['rmse']),
        np.average(evaluate_mean['mae']),
        np.average(evaluate_mean['spr']),
        np.average(evaluate_mean['prs'])))
    
def get_original_peerread(args, split_sentence=False, binary=False):
    
    tokenizer = AutoTokenizer.from_pretrained(args.bert)
    
    train_dataset = PeerRead(
        args, split='train', 
        transforms=[transforms.Tokenize(tokenizer, args.max_length)])
    dev_dataset = PeerRead(
        args, split='dev',
        transforms=[transforms.Tokenize(tokenizer, args.max_length)])
    test_dataset = PeerRead(
        args, split='test',
        transforms=[transforms.Tokenize(tokenizer, args.max_length)]) 
    
    logging.info('Train: {}'.format(len(train_dataset)))
    logging.info('Dev: {}'.format(len(dev_dataset)))
    logging.info('Test: {}'.format(len(test_dataset)))
    
    evaluate_mean(args, train_dataset)
    
    return train_dataset, dev_dataset, test_dataset, tokenizer


def get_unlabeled_peerread(args, test_ids=None, tokenizer=None):    
    unlabeled_dataset = PeerRead(args, unlabeled=True)
    test_idxs = []
    for i, d in enumerate(unlabeled_dataset.paper_ids):
        if d in test_ids:
            test_idxs.append(i)
    indexs = np.delete(np.arange(len(unlabeled_dataset)), test_idxs)
    if tokenizer is not None:
        unlabeled_dataset = PeerRead(
            args, indexs=indexs, unlabeled=True,
            transforms=[transforms.Tokenize(tokenizer, args.max_length)])
    else:
        unlabeled_dataset = PeerRead(
            args, indexs=indexs, unlabeled=True) 
    logging.info('PeerRead unlabeled: {}'.format(len(unlabeled_dataset)))
    return unlabeled_dataset


def get_binary_balanced_peerread(args):
    tokenizer = AutoTokenizer.from_pretrained(args.bert)
    dataset = PeerRead(args, balance=True)
    return dataset, tokenizer

    
def get_binary_balanced_peerread_by_indexs(args, indexs, tokenizer=None):
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(args.bert)
    dataset = PeerRead(
        args, indexs=indexs,balance=True,
        transforms=[transforms.Tokenize(tokenizer, args.max_length)])
    return dataset, tokenizer