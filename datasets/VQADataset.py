import json

import h5py
import torch
import torch.utils.data as data

import sys, os

cwd = os.getcwd()
sys.path.append(cwd)

from datasets.ImageFeatureDataset import ImageFeatureDataset
from preprocessing.PreprocessQA import tokenize_questions, tokenize_answers, encode_ques, encode_answers


def get_loader(config, split):
    """ Return the data loader given the specified dataset split """
    
    # get the VQA dataset given the specified dataset split
    dataset = VQADataset(
        config,
        split
    )
    
    # get the dataloader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size = config['training']['batch_size'],
        # only shuffle the data in training phase
        shuffle=True if split == 'train' or split == 'val' else False,
        num_workers = config['training']['data_workers'],
        collate_fn = collate_fn
    )
    
    return loader

# overwrite
# call __getitem__ function to perform custom batching before dataloader returns it
def collate_fn(batch):
    # Sort the batch samples according to question lengths in descending order
    # To pack the pack_padded_sequence when encoding questions using RNN
    batch.sort(key=lambda x: x['q_length'], reverse=True)
    return data.dataloader.default_collate(batch)


class VQADataset(data.Dataset):

    def __init__(self, config, split):
        super(VQADataset, self).__init__()
        
        # split = train / test / val
        self.split = split

        # retrieve the vocabulary built 
        with open(config['annotations']['vocab_path'], 'r') as fd:
            vocabs = json.load(fd)
        # vocab file
        self.vocabs = vocabs
        # question vocab
        self.token_to_index = self.vocabs['question']
        # answer vocab
        self.answer_to_index = self.vocabs['answer']

        # retrieve the annotation directory
        annotations_dir = config['annotations']['anno_dir']

        # retrieve the annotation file (train / test / val)
        path_ann = os.path.join(annotations_dir, self.split + ".json")
        with open(path_ann, 'r') as fd:
            self.annotations = json.load(fd)

        # retrieve the max question length
        self.max_question_length = config['annotations']['max_length']

        # tokenize questions
        self.questions = tokenize_questions(self.annotations)
        # encode questions 
        self.questions = [encode_ques(q, self.token_to_index, 
                        self.max_question_length) for q in self.questions]  
        
        # if it is not in testing phase
        if self.split != 'test':
            # tokenize and encode answers
            self.answers = [encode_answers(a, self.answer_to_index) 
                            for a in tokenize_answers(self.annotations)] 
            self._filter_unanswerable_samples()
        
        # load image names in feature extraction order from h5 file
        with h5py.File(config['images']['feat_path'], 'r') as f:
            img_names = list(f['img_name'])
        self.name_to_id = {name.decode(): i for i, name in enumerate(img_names)}
	# load image names from annotations 
        # to get items from dataset
        self.img_names = [s['image'] for s in self.annotations]
        # image features that reside in h5 file
        self.features = ImageFeatureDataset(config['images']['feat_path'])

    # Filter training samples that do not have at least one ground truth
    def _filter_unanswerable_samples(self):
        
        a = []
        q = []
        annotations = []
        
        # loop through all encoded answers for the single sample
        # it is a vector of answer tokens and its count
        for i in range(len(self.answers)):
            # for each encoded answer, where value 0 is the unknown token
            # loop through all tokens in the single answer
            # if all tokens are unknown tokens, the answer itself is unanswerable
            # skip if it is all zeroes, as it does not represent a ground truth
            #if len(self.answers[i].nonzero()) > 0:
            if len(torch.nonzero(self.answers[i], as_tuple=False)) > 0:
                a.append(self.answers[i])
                q.append(self.questions[i])
                annotations.append(self.annotations[i])
        
        # reassign back
        self.answers = a
        self.questions = q
        self.annotations = annotations

    @property
    def num_tokens(self):
        # additional token is used to represent <unknown> token at index 0
        return len(self.token_to_index) + 1

    def __getitem__(self, i):

        item = {}
        item['question'], item['q_length'] = self.questions[i]
        if self.split != 'test':
            item['answer'] = self.answers[i]
        img_name = self.img_names[i]
        feature_id = self.name_to_id[img_name]
        item['img_name'] = self.img_names[i]
        item['img'] = self.features[feature_id]
        # collate_fn is overwritten to sort the samples in descending order 
        # to pack the question with pack_padded_sequence in the model.
        # sample_id used to restore original order when evaluating predictions
        item['sample_id'] = i

        return item

    def __len__(self):
        return len(self.questions)
