import argparse
import json
import itertools
from collections import Counter
import yaml
import sys, os

cwd = os.getcwd()
sys.path.append(cwd)

from PreprocessQA import tokenize_questions, tokenize_answers


def extract_vocab(iterable, idx_start, top_k=None):
    """ Turns an iterable of list of tokens into a vocabulary.
        These tokens could be single answers or word tokens in questions.
    """
    # iterate through all tokens
    all_tokens = itertools.chain.from_iterable(iterable)
    # count token occurence frequency
    counter = Counter(all_tokens)
    
    # creating vocab for answers
    # where only top_k tokens are included
    if top_k: 
        most_common_tokens = counter.most_common(top_k)
        
    # creating vocab for questions
    else: 
        most_common_tokens = counter.most_common()

    # form vocab (token, index)
    # for ques vocab, the index starts with 1 
    # for ans vocab, the index starts with 0
    vocab = {t[0]: i for i, t in enumerate(most_common_tokens, start=idx_start)}
    
    return vocab


def main():
    # Load config from yaml file
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_config', default='config/default.yaml', type=str,
                    help='path to a yaml config file')
    args = parser.parse_args([])

    if args.path_config is not None:
        with open(args.path_config, 'r') as handle:
            config = yaml.safe_load(handle)

    # Load annotations
    dir_path = config['annotations']['anno_dir']

    # vocabs are created based on training set only
    train_path = os.path.join(dir_path, 'train.json')
    with open(train_path, 'r') as fd:
        train_anno = json.load(fd)
    
    # first, tokenize
    questions = tokenize_questions(train_anno)
    answers = tokenize_answers(train_anno)
    
    # second, extract vocab among all tokens
    question_vocab = extract_vocab(questions, idx_start=1)
    answer_vocab = extract_vocab(answers, idx_start=0, top_k=config['annotations']['top_ans'])
    
    # Save vocabs
    vocabs = {
        'question': question_vocab,
        'answer': answer_vocab
    }

    with open(config['annotations']['vocab_path'], 'w') as fd:
        json.dump(vocabs, fd)

    print("vocabs saved in {}".format(config['annotations']['vocab_path']))


if __name__ == '__main__':
    main()
