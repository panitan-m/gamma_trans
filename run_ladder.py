import argparse
import logging
import random
import math
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler, SequentialSampler

from transformers import get_cosine_schedule_with_warmup

from datasets.peerread import get_binary_balanced_peerread, get_unlabeled_peerread, get_binary_balanced_peerread_by_indexs
from datasets.scisumm import get_unlabeled_scisumm, ConcatDataset
from models.ladder_net import LadderNetworkFC
from models.metrics import METRICS
from models.evaluator import EvaluatorKFold as Evaluator

import utils as U
from utils import AverageMeter

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', default=['acl_2017'], type=str, nargs="+",
                    help='dataset name {acl_2017|iclr_2017|conll_2016}')
parser.add_argument('--unlabeled-datasets', default=['acl_2017'], type=str, nargs="+",
                    help='unlabeled dataset name {acl_2017|iclr_2017|conll_2016}')
parser.add_argument('--section', default='all', type=str,
                    help='section title {all|abstract|introduction|related_work|experiment|conclusion}')
# parser.add_argument('--aspects', default=['recommend', 'substance', 'appropriate', 'comparison', 'soundness', 'originality', 'clarity', 'impact'],
parser.add_argument('--aspects', default=['recommend'],
                    type=str, nargs="+", help='score aspects')
parser.add_argument('--n-labeled', default=-1, type=int,
                    help='Number of labeled data')
parser.add_argument('--n-unlabeled', default=-1, type=int, 
                    help='Number of unlabeled data')
parser.add_argument('--n-folds', default=5, type=int,
                    help='Number of folds')
parser.add_argument('--out-dir', default='results', type=str,
                    help='output directory')
parser.add_argument('--eval-metrics', default=['f1'], type=str, nargs="+",
                    help='evaluation metrices')
parser.add_argument('--bert', default='bert-base-uncased', type=str,
                    help='bert pretrained model')
parser.add_argument('--freeze-bert', type=str, default='all',
                    help='Freeze BERT weights except the specified layer and above layers')
parser.add_argument('--max-length', default=512, type=int,
                    help='max length')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout')
parser.add_argument('--ckpt-dir', default='results/checkpoints', type=str,
                    help='output directory')
parser.add_argument('--load-checkpoint', type=int,
                    help='load checkpoint at epoch N')
parser.add_argument('--save-checkpoint', type=int, default=0,
                    help='save checkpoint every N epochs')
parser.add_argument('--loss-fn', default='bce', type=str,
                    help='loss function')
parser.add_argument('--batch-size', type=int, default=8,
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.003,
                    help='learning rate')
parser.add_argument('--eval-steps', type=int, default=10,
                    help='evaluate every N steps')
parser.add_argument('--epochs', type=int, default=30,
                    help='Number of epochs')
parser.add_argument('--seed', type=int, default=0,
                    help='Random seed')
args = parser.parse_args()

out_dir = args.out_dir
U.mkdir_p(out_dir)
U.set_logger(out_dir)
U.print_args(args)

if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
dataset, _ = get_binary_balanced_peerread(args)
x, y = dataset.split_xy()
logging.info('Total labeled: {}'.format(len(dataset)))

test_eval = Evaluator(args)
    
skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
sss = StratifiedShuffleSplit(train_size=args.n_labeled, random_state=args.seed)
for kf, (train_idx, test_idx) in enumerate(skf.split(x, y)):    
    
    train_dataset, tokenizer = get_binary_balanced_peerread_by_indexs(args, train_idx)
    if args.n_labeled > 0:
        x, y = train_dataset.split_xy()
        train_idx, _ = list(sss.split(x, y))[0]
        train_dataset.set_by_idxs(train_idx)
    test_dataset, _ = get_binary_balanced_peerread_by_indexs(args, test_idx, tokenizer)
    peerread_unlabeled = get_unlabeled_peerread(args, test_ids=test_dataset.paper_ids, tokenizer=tokenizer)
    scisumm_unlabeled = get_unlabeled_scisumm(args, tokenizer)    
    unlabeled_dataset = ConcatDataset([peerread_unlabeled, scisumm_unlabeled])
    unlabeled_dataset.sample(args.n_unlabeled, args.seed)
    logging.info('Train: {}'.format(len(train_dataset)))
    logging.info('Test: {}'.format(len(test_dataset)))
    logging.info('Total unlabeled: {}'.format(len(unlabeled_dataset)))

    train_loader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=args.batch_size,
        drop_last=True)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size*5)

    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        sampler=RandomSampler(unlabeled_dataset),
        batch_size=args.batch_size,
        drop_last=True)

    checkpoint = args.ckpt_dir + '/checkpoints/fold_{}/ckpt.e{}'.format(kf, args.load_checkpoint) if args.load_checkpoint else None
    model = LadderNetworkFC(
        args, layer_sizes=[768, 1000, 500, 250, 250, 250, len(args.aspects)],
        bert_checkpoint=checkpoint, binary=True)
    model.cuda()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 0.1*args.epochs*len(unlabeled_loader), args.epochs*len(unlabeled_loader))

    writer = SummaryWriter(args.out_dir+'/fold_{}'.format(kf))

    current_step = 0   
    losses = AverageMeter()
    den_losses = AverageMeter()
    train_acc = AverageMeter()
    test_losses = AverageMeter()

    train_loader_iter = iter(train_loader)
    for epoch in range(args.epochs):
        with tqdm(unlabeled_loader) as p_bar:
            
            for batch_idx, (inputs_u, _) in enumerate(unlabeled_loader):
                
                current_step += 1
            
                model.train()
                
                try:
                    inputs_x, targets_x = next(train_loader_iter)
                except:
                    train_loader_iter = iter(train_loader)
                    inputs_x, targets_x = next(train_loader_iter)
                
                inputs_u = tuple(map(lambda x: x.cuda(), inputs_u))
                inputs_x, targets_x = tuple(map(lambda x: x.cuda(), inputs_x)), targets_x.cuda()
                cost, u_cost, logits = model(inputs_x, targets_x, inputs_u, args.epochs, epoch)
            
                losses.update(cost.item())
                den_losses.update(u_cost.item())
                cost.backward()
                
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                      
                p_bar.set_description("Train Epoch: {epoch}/{epochs:3}. Iter: {batch:3}/{iter:3}. LR: {lr:.2e}. Loss: {loss:.4f}. Denoising Loss: {den_loss:.4f}".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=len(unlabeled_loader),
                    lr=scheduler.get_last_lr()[0],
                    loss=losses.avg,
                    den_loss=den_losses.avg))
                p_bar.update()

                writer.add_scalars("loss", {'train': losses.avg}, current_step)
                
                if current_step % args.eval_steps == 0:
                    
                    with torch.no_grad():
                        model.eval() 
                        test_pred, test_truth = [], []
                        for batch_idx, (inputs, targets) in enumerate(test_loader):
                            inputs, targets = tuple(map(lambda x: x.cuda(), inputs)), targets.cuda()
                            cost, logits = model(inputs, targets)
                            test_losses.update(cost)
                            test_pred.extend(logits.tolist())
                            test_truth.extend(targets.tolist())
                            
                    writer.add_scalars("loss", {'test': test_losses.avg}, current_step)
                    
                    test_pred, test_truth = torch.tensor(test_pred).T, torch.tensor(test_truth).T                    
                    test_eval.evaluate(test_pred, test_truth, writer, kf, current_step)

                    test_losses.reset()
                
            if args.save_checkpoint > 0 and (epoch+1) % args.save_checkpoint == 0:
                save_dir = out_dir + "/checkpoints/fold_{}/ckpt.e{}".format(kf, epoch+1)
                model.bert.save_pretrained(save_dir)
                
            losses.reset()
            den_losses.reset()
            train_acc.reset()
            
test_eval.print_final()