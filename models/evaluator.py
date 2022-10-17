import logging
import json
import math
import numpy as np

from models.metrics import METRICS

logger = logging.getLogger(__name__)

class EvaluatorKFold():
    
    def __init__(self, args):
        self.out_dir = args.out_dir
        self.eval_metrics = args.eval_metrics
        self.aspects = args.aspects
        self.ref = {}
        self.preds = {}
        self.best_test = {}
        self.best_eval = {}
        for m in args.eval_metrics:
            self.preds[m] = [None] * args.n_folds
            self.best_test[m] = [-1] * args.n_folds
            self.best_eval[m] = [math.inf]*args.n_folds if 'e' in m else [-1]*args.n_folds      

    def save_best(self, m, fold, eval_avg_aspects, eval_aspects, preds, targets):
        self.best_eval[m][fold] = eval_avg_aspects[m]
        self.best_test[m][fold] = eval_aspects
        self.ref[fold] = targets.cpu().tolist()
        self.preds[m][fold] = preds.cpu().tolist()
            
    def evaluate(self, preds, targets, writer, fold, current_step):
        description = '[TEST] '
        eval_avg_aspects = {}
        is_best = False
        for m in self.eval_metrics:
            eval_aspects = []
            for aid in range(len(self.aspects)):
                eval_output = METRICS[m](preds[aid], targets[aid])
                eval_aspects.append(eval_output)
                writer.add_scalars("metrics/{}".format(m), {'{}'.format(aid): eval_output}, current_step)
            eval_avg_aspects[m] = np.average(eval_aspects)
            description += "{}: {:.4f} ".format(m.upper(), eval_avg_aspects[m])
            if 'e' in m:
                if eval_avg_aspects[m] < self.best_eval[m][fold]:
                    self.save_best(m, fold, eval_avg_aspects, eval_aspects, preds, targets)
            else:
                if eval_avg_aspects[m] > self.best_eval[m][fold]:
                    self.save_best(m, fold, eval_avg_aspects, eval_aspects, preds, targets)
                    is_best = True
                    
        description += '  ( '
        for m in self.eval_metrics:
            description += "{}: {:.4f} ".format(m.upper(), np.average(self.best_test[m][fold]))
        description += ')'
        logger.info(description)
        return is_best

    def dump_predictions(self):
        with open(self.out_dir+'/predictions.json', 'w') as f:
            json.dump({
                'targets': self.ref,
                'predictions': self.preds
            }, f)
        
    def print_final(self):
        log_format = '\t{:13s}'
        for m in self.eval_metrics:
            log_format += '\t{:7.4f}'
        for a in range(len(self.aspects)):
            log_vars = [self.aspects[a].upper()]
            for m in self.eval_metrics:
                log_vars.append(np.average(self.best_test[m], 0)[a])
            logging.info(log_format.format(*log_vars))
            
        log_format = '\t{:13s}'
        log_vars = ['TOTAL']
        for m in self.eval_metrics:
            log_format += '\t{:7.4f}'
            log_vars.append(np.average(self.best_test[m]))
        logging.info(log_format.format(*log_vars))

        self.dump_predictions()
        
class Evaluator():
    
    def __init__(self, args):
        self.out_dir = args.out_dir
        self.eval_metrics = args.eval_metrics
        self.aspects = args.aspects
        self.ref = {}
        self.preds = {}
        self.best_eval = {}
        self.test_eval = {}
        for m in args.eval_metrics:
            self.preds[m] = None
            self.test_eval[m] = None
            self.best_eval[m] = math.inf if 'e' in m else -1

    def save_best(self, m, fold, eval_avg_aspects, eval_aspects, preds, targets):
        self.best_eval[m][fold] = eval_avg_aspects[m]
        self.best_test[m][fold] = eval_aspects
        self.ref[fold] = targets.cpu().tolist()
        self.preds[m][fold] = preds.cpu().tolist()
            
    def evaluate(self, dev_pred, dev_truth, test_pred, test_truth, current_step):
        dev_eval = {}
        description = '[DEV] '
        for m in self.eval_metrics:
            dev_eval_aspects = []
            test_eval_aspects = []
            for aid in range(len(self.aspects)):
                dev_eval_aspects.append(METRICS[m](dev_pred[aid], dev_truth[aid]))
                test_eval_aspects.append(METRICS[m](test_pred[aid], test_truth[aid]))
            dev_eval[m] = np.average(dev_eval_aspects)
            description += "{}: {:.4f} ".format(m.upper(), dev_eval[m])
            if 'e' in m:
                if dev_eval[m] < self.best_eval[m]:
                    self.best_eval[m] = dev_eval[m]
                    self.test_eval[m] = test_eval_aspects
            else:
                if dev_eval[m] > self.best_eval[m]:
                    self.best_eval[m] = dev_eval[m]
                    self.test_eval[m] = test_eval_aspects
        description += '  ( '
        for m in self.eval_metrics:
            description += "{}: {:.4f} ".format(m.upper(), np.average(self.best_eval[m]))
        description += ')'
        logger.info(description)
        
        description = '[TEST] '
        for m in self.eval_metrics:
            description += "{}: {:.4f} ".format(m.upper(), np.average(self.test_eval[m]))
        logger.info(description)
        
    def print_final(self):
        for a in range(len(self.aspects)):
            logging.info('\t%13s\t%.4f\t%.4f\t%7.4f\t%7.4f'%(
                self.aspects[a].upper(),
                self.test_eval['rmse'][a],
                self.test_eval['mae'][a],
                self.test_eval['spr'][a],
                self.test_eval['prs'][a]))
        logging.info('\t%13s\t%.4f\t%.4f\t%7.4f\t%7.4f'%(
                'TOTAL',
                np.average(self.test_eval['rmse']),
                np.average(self.test_eval['mae']),
                np.average(self.test_eval['spr']),
                np.average(self.test_eval['prs'])))