import argparse
from datetime import datetime
import yaml

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from tqdm import tqdm

import sys, os

cwd = os.getcwd()
sys.path.append(cwd)

import models
import utils
from datasets import VQADataset


def train(model, loader, optimizer, tracker, epoch, split):
    
    # switch to training mode
    model.train()

    # tracking progress purpose
    tracker_class, tracker_params = tracker.MovingMeanMonitor, {'momentum': 0.99}
    tq = tqdm(loader, desc='{} E{:03d}'.format(split, epoch), ncols=0)
    loss_tracker = tracker.track('{}_loss'.format(split), tracker_class(**tracker_params))
    acc_tracker = tracker.track('{}_acc'.format(split), tracker_class(**tracker_params))
    
    log_softmax = nn.LogSoftmax(dim=1)
    if torch.cuda.is_available():
        # transfer to GPU if available
        log_softmax = log_softmax.cuda()
        
    for item in tq:
        v = item['img']
        q = item['question']
        a = item['answer']
        q_length = item['q_length']

        if torch.cuda.is_available():
                # transfer to GPU if available
                    v = v.cuda()
                    q = q.cuda()
                    a = a.cuda()
                    q_length = q_length.cuda()

        out = model(v, q, q_length)

        # This is the Soft-loss described in https://arxiv.org/pdf/1708.00584.pdf

        nll = -log_softmax(out)
        loss = (nll * a / 10).sum(dim=1).mean()

        acc = utils.vqa_accuracy(out.data, a.data).cpu()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_tracker.append(loss.item())
        acc_tracker.append(acc.mean())
        fmt = '{:.4f}'.format
        tq.set_postfix(loss=fmt(loss_tracker.mean.value), acc=fmt(acc_tracker.mean.value))
        

def evaluate(model, loader, tracker, epoch, split):
    
    # switch to evaluation mode
    model.eval()
    tracker_class, tracker_params = tracker.MeanMonitor, {}

    predictions = []
    samples_ids = []
    accuracies = []

    # track progress
    tq = tqdm(loader, desc='{} E{:03d}'.format(split, epoch), ncols=0)
    loss_tracker = tracker.track('{}_loss'.format(split), tracker_class(**tracker_params))
    acc_tracker = tracker.track('{}_acc'.format(split), tracker_class(**tracker_params))
    
    log_softmax = nn.LogSoftmax(dim=1)
    if torch.cuda.is_available():
        # transfer to GPU if available
        log_softmax = log_softmax.cuda()
    
    with torch.inference_mode():
        for item in tq:
            v = item['img']
            q = item['question']
            a = item['answer']
            sample_id = item['sample_id']
            q_length = item['q_length']

            if torch.cuda.is_available():
                # transfer to GPU if available
                    v = v.cuda()
                    q = q.cuda()
                    a = a.cuda()
                    sample_id = sample_id.cuda()
                    q_length = q_length.cuda()

            out = model(v, q, q_length)

            # This is the Soft-loss described in https://arxiv.org/pdf/1708.00584.pdf

            nll = -log_softmax(out)

            loss = (nll * a / 10).sum(dim=1).mean()
            acc = utils.vqa_accuracy(out.data, a.data).cpu()

            # save predictions of this batch
            _, answer = out.data.cpu().max(dim=1)

            predictions.append(answer.view(-1))
            accuracies.append(acc.view(-1))
            # Sample id is necessary to obtain the mapping sample-prediction
            samples_ids.append(sample_id.view(-1).clone())

            loss_tracker.append(loss.item())
            acc_tracker.append(acc.mean())
            fmt = '{:.4f}'.format
            tq.set_postfix(loss=fmt(loss_tracker.mean.value), acc=fmt(acc_tracker.mean.value))

        predictions = list(torch.cat(predictions, dim=0))
        accuracies = list(torch.cat(accuracies, dim=0))
        samples_ids = list(torch.cat(samples_ids, dim=0))

    eval_results = {
        'answers': predictions,
        'accuracies': accuracies,
        'samples_ids': samples_ids,
        'avg_accuracy': acc_tracker.mean.value,
        'avg_loss': loss_tracker.mean.value
    }

    return eval_results


def main():

    # Load config yaml file
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default='config/default.yaml', type=str,
                        help='path to a yaml config file')
    args = parser.parse_args([])
    if args.config_file is not None:
        with open(args.config_file, 'r') as handle:
            config = yaml.safe_load(handle)
            
    # generate log directory
    # log directory name = the current datetime in string 
    log_dir_name = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    model_log_dir_path = os.path.join(config['logs']['dir_logs'], log_dir_name)
    # if the log directory does not exist, create one
    if not os.path.exists(model_log_dir_path):
        os.mkdir(model_log_dir_path)
    print('Model logs will be saved in {}'.format(model_log_dir_path))

    # Generate VQA datasets then get data loaders
    train_loader = VQADataset.get_loader(config, split='train')
    val_loader = VQADataset.get_loader(config, split='val')

    # create model
    model = models.Model(config, train_loader.dataset.num_tokens)

    if torch.cuda.is_available():
        # transfer to GPU if available
        model = model.cuda()
        model = nn.DataParallel(model).cuda()
        # Benchmark mode is good whenever your input sizes for your network do not vary
        # cudnn will look for the optimal set of algorithms for that particular configuration. 
        # usually leads to faster runtime
        # NOTE: input size is set in config['img_size']
        cudnn.benchmark = True
  
    # Adam optimizer is used
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 config['training']['lr'])
    
    # Load pretrained model weights if exists
    if config['model']['pretrained_model'] is not None:
        print("Loading Pretrained Model")
        log = torch.load(config['model']['pretrained_model'])
        dict_weights = log['weights']
        model.load_state_dict(dict_weights)
    
    tracker = utils.Tracker()

    min_loss = 10
    max_accuracy = 0

    path_best_accuracy = os.path.join(model_log_dir_path, 'best_accuracy_log.pth')
    path_best_loss = os.path.join(model_log_dir_path, 'best_loss_log.pth')
    
    # start training
    for i in range(config['training']['epochs']):

        # train
        train(model, train_loader, optimizer, tracker, epoch=i, split='train')
        # then evaluate
        eval_results = evaluate(model, val_loader, tracker, epoch=i, split='val')

        # update current progress to log file
        log_data = {
             'epoch': i,
             'tracker': tracker.to_dict(),
             'config': config,
             'weights': model.state_dict(),
             'eval_results': eval_results,
             'vocabs': train_loader.dataset.vocabs,
        }

        # save logs for min validation loss
        if eval_results['avg_loss'] < min_loss:
            torch.save(log_data, path_best_loss)  # save model
            min_loss = eval_results['avg_loss']  # update min loss value
        
        # save logs for max validation accuracy
        if eval_results['avg_accuracy'] > max_accuracy:
            torch.save(log_data, path_best_accuracy)  # save model
            max_accuracy = eval_results['avg_accuracy']  # update max accuracy value


    # Save final model
    log_data = {
        'tracker': tracker.to_dict(),
        'config': config,
        'weights': model.state_dict(),
        'vocabs': train_loader.dataset.vocabs,
    }
    torch.save(log_data, os.path.join(model_log_dir_path, 'final_log.pth'))


if __name__ == '__main__':
    main()
