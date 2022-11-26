import argparse
import json
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import yaml
from tqdm import tqdm

import sys, os

cwd = os.getcwd()
sys.path.append(cwd)

import models
from datasets import VQADataset


def predict_answers(model, loader, split):
    model.eval()
    predicted = []
    samples_ids = []

    tq = tqdm(loader)

    print("Evaluating...\n")

    for item in tq:
        v = item['img']
        q = item['question']
        sample_id = item['sample_id']
        q_length = item['q_length']

        if torch.cuda.is_available():
            # transfer to GPU if available
            v = v.cuda()
            q = q.cuda()
            sample_id = sample_id.cuda()
            q_length = q_length.cuda()
        
        out = model(v, q, q_length)

        _, answer = out.data.cpu().max(dim=1)

        predicted.append(answer.view(-1))
        samples_ids.append(sample_id.view(-1).clone())

    predicted = list(torch.cat(predicted, dim=0))
    samples_ids = list(torch.cat(samples_ids, dim=0))

    print("Evaluation completed")

    return predicted, samples_ids


def create_submission(input_annotations, predicted, samples_ids, vocabs):
    answers = torch.FloatTensor(predicted)
    indexes = torch.IntTensor(samples_ids)
    ans_to_id = vocabs['answer']
    # need to translate answers ids into answers
    id_to_ans = {idx: ans for ans, idx in ans_to_id.items()}
    # sort based on index the predictions
    sort_index = np.argsort(indexes)
    sorted_answers = np.array(answers, dtype='int_')[sort_index]
    sorted_answers = np.array(answers, dtype='int_')

    real_answers = []
    for ans_id in sorted_answers:
        ans = id_to_ans[ans_id]
        real_answers.append(ans)

    # Integrity check
    assert len(input_annotations) == len(real_answers)

    submission = []
    for i in range(len(input_annotations)):
        pred = {}
        pred['image'] = input_annotations[i]['image']
        pred['answer'] = real_answers[i]
        submission.append(pred)

    return submission


def main():
    # Load config yaml file
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_config', default='config/default.yaml', type=str,
                        help='path to a yaml config file')
    args = parser.parse_args()

    if args.path_config is not None:
        with open(args.path_config, 'r') as handle:
            config = yaml.safe_load(handle)

    if torch.cuda.is_available():
        cudnn.benchmark = True

    # Generate dataset and loader
    print("Loading samples to predict from %s" % os.path.join(config['annotations']['anno_dir'],
                                                              config['prediction']['split'] + '.json'))

    # Load annotations
    path_annotations = os.path.join(config['annotations']['anno_dir'], config['prediction']['split'] + '.json')
    input_annotations = json.load(open(path_annotations, 'r'))

    # Data loader and dataset
    input_loader = VQADataset.get_loader(config, split=config['prediction']['split'])

    # Load model weights
    print("Loading Model from %s" % config['prediction']['model_path'])
    log = torch.load(config['prediction']['model_path'])
    #log = torch.load(config['prediction']['model_path'],map_location=torch.device('cpu'))

    # Num tokens seen during training
    num_tokens = len(log['vocabs']['question']) + 1
    # Use the same configuration used during training
    train_config = log['config']

    model = models.Model(train_config, num_tokens)
    model = nn.DataParallel(model)

    dict_weights = log['weights']
    model.load_state_dict(dict_weights)

    predicted, samples_ids = predict_answers(model, input_loader, split=config['prediction']['split'])

    submission = create_submission(input_annotations, predicted, samples_ids, input_loader.dataset.vocabs)

    with open(config['prediction']['submission_file'], 'w') as fd:
        json.dump(submission, fd)

    print("Submission file saved in %s" % config['prediction']['submission_file'])


if __name__ == '__main__':
    main()
