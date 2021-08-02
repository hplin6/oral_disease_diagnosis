# ------------------------------------------------------------------------------
# Copy from https://github.com/HRNet/HRNet-Image-Classification
# Modified by us
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import torch
import numpy as np
import math

from core.evaluate import accuracy,cal_metrics,save_pregt_result,get_confusion_matrix
from sklearn.metrics import classification_report

logger = logging.getLogger(__name__)


def stats(ground_truth, preds):
    labels = range(ground_truth.shape[2])
    g = np.argmax(ground_truth, axis=2).ravel()
    p = np.argmax(preds, axis=2).ravel()
    stat_dict = {}
    for i in labels:
        # compute all the stats for each label
        tp = np.sum(g[g == i] == p[g == i])
        fp = np.sum(g[p == i] != p[p == i])
        fn = np.sum(g == i) - tp
        tn = np.sum(g != i) - fp
        stat_dict[i] = (tp, fp, fn, tn)
    return stat_dict

def to_set(preds):
    idxs = np.argmax(preds, axis=2)
    return [list(set(r)) for r in idxs]

def set_stats(ground_truth, preds):
    labels = range(ground_truth.shape[2])
    ground_truth = to_set(ground_truth)
    preds = to_set(preds)
    stat_dict = {}
    for x in labels:
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        for g, p in zip(ground_truth, preds):
            if x in g and x in p:  # tp
                tp += 1
            if x not in g and x in p:  # fp
                fp += 1
            if x in g and x not in p:
                fn += 1
            if x not in g and x not in p:
                tn += 1
        stat_dict[x] = (tp, fp, fn, tn)
    return stat_dict

def compute_f1(tp, fp, fn, tn):
    precision = tp / float(tp + fp)
    recall = tp / float(tp + fn)
    specificity = tn / float(tn + fp)
    npv = tn / float(tn + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return f1, tp + fn

def print_results(seq_sd, set_sd):
    print("\t\t Seq F1    Set F1")
    seq_tf1 = 0
    seq_tot = 0
    set_tf1 = 0
    set_tot = 0
    for k, v in seq_sd.items():
        set_f1, n = compute_f1(*set_sd[k])
        set_tf1 += n * set_f1
        set_tot += n
        seq_f1, n = compute_f1(*v)
        seq_tf1 += n * seq_f1
        seq_tot += n
        print("{:>10} {:10.3f} {:10.3f}".format(preproc.classes[k], seq_f1, set_f1))
    print("{:>10} {:10.3f} {:10.3f}".format("Average", seq_tf1 / float(seq_tot), set_tf1 / float(set_tot)))

def train(config, train_loader, model, criterion, optimizer, epoch, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        #target = target - 1 # Specific for imagenet

        # compute output
        output = model(input)
        target = target.cuda(non_blocking=True)

        loss = criterion(output, target)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        prec = accuracy(output, target)
        top1.update(prec, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, top1=top1)
            logger.info(msg)

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.val, global_steps)
                writer.add_scalar('train_top1', top1.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1


def validate_speed(config, val_loader, model, criterion, writer_dict=None):

    model.eval()
    run_num = 1000
    count_num = 0
    start_time = time.time()
    iter_num = math.ceil(run_num / len(val_loader))
    with torch.no_grad():
        end = time.time()
        for j in range(iter_num):
            for i, (input, target) in enumerate(val_loader):
                # compute output
                output = model(input)
                if count_num < run_num:
                    count_num += 1
                else:
                    break

        end_time = time.time()
        run_time = (end_time - start_time)
        print("runtime=%.4fMS(%d)" % (run_time, count_num))
    return 11

def validate(config, val_loader, model, criterion, writer_dict=None,b_test=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        outputs=[]
        targets=[]
        preds=[]
        for i, (input, target) in enumerate(val_loader):
            # compute output
            output = model(input)

            target = target.cuda(non_blocking=True)

            loss = criterion(output, target)

            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))
            prec = accuracy(output, target)
            top1.update(prec, input.size(0))
            preds+=output.cuda().data.cpu().numpy().tolist()

            pred_class = output.argmax(dim=1, keepdim=False)  # get the index of the max log-probability
            true_class = target.view_as(pred_class)
            true_class = true_class.cuda().data.cpu().numpy()
            pred_class = pred_class.cuda().data.cpu().numpy()
            outputs+=pred_class.tolist()
            targets+=true_class.tolist()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        _,_,_,F1_score,y_true,y_pred,cmat=cal_metrics(np.array(targets),np.array(preds))
        if b_test:
            save_pregt_result("./output/test_result.csv", y_true, y_pred)
            get_confusion_matrix(cmat)
        print(classification_report(np.array(targets),np.array(outputs)))

        msg = 'Test: Time {batch_time.avg:.3f}\t' \
              'Loss {loss.avg:.4f}\t' \
              'Error@1 {error1:.3f}\t' \
              'Accuracy@1 {top1.avg:.3f}\t'.format(
                  batch_time=batch_time, loss=losses, top1=top1, error1=100-top1.avg)
        logger.info(msg)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar('valid_loss', losses.avg, global_steps)
            writer.add_scalar('valid_top1', top1.avg, global_steps)
            writer_dict['valid_global_steps'] = global_steps + 1

    return F1_score#top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
