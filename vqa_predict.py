import argparse
import json
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import yaml
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import base64

import sys, os
cwd = os.getcwd()
sys.path.append(cwd)

import models
from datasets import VQADataset
import torchvision.transforms as transforms
from preprocessing.ImageFeatureExtractor import ImageFeatureExtractor
from preprocessing.PreprocessQA import encode_ques


def predict_preprocess(img_base64, question):
    
    ''' Load config yaml file '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_config', default='config/default.yaml', type=str,
                        help='path to a yaml config file')
    args = parser.parse_args()

    if args.path_config is not None:
        with open(args.path_config, 'r') as handle:
            config = yaml.safe_load(handle)

    
    ''' extract image feature '''
            
    # image in base 64 string
    # Use PIL to load the image
    img = Image.open(BytesIO(base64.b64decode(img_base64)))
    img_size = config['images']['img_size']
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        # ImageNet normalization setting is used
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
        ])
    img = transform(img)
    # single sample to the network, thus add batch size dimension
    img = img.unsqueeze(0)
        
    
    # load the image feature extractor class
    net = ImageFeatureExtractor()
    # eval mode
    net.eval()
    
    # start extracting features
    # no calculating gradient
    with torch.no_grad():
        # passing image to network and features are extracted 
        att_feat = net(img)
    
    ''' question feature extraction '''
    
    # retrieve the vocabulary built 
    with open(config['annotations']['vocab_path'], 'r') as fd:
        vocabs = json.load(fd)
    # question vocab
    token_to_index = vocabs['question']
    # retrieve the max question length
    max_question_length = config['annotations']['max_length']
    
    # tokenize questions
    # lower case
    question = question.lower()
    # remove question mark
    question = question[:-1]
     # sentence to list
    question = question.split(' ')
    # encode questions 
    question, q_length = encode_ques(question, token_to_index, max_question_length) 
        
    ''' load the model '''
    
    # Load model weights
    log = torch.load(config['prediction']['model_path'])
    # load model weights with mapping to cpu in case of only cpu is available during prediction but model is trained on gpu
    # log = torch.load(config['prediction']['model_path'], map_location=torch.device('cpu'))
    
    # Num tokens seen during training
    num_tokens = len(log['vocabs']['question']) + 1
    # Use the same configuration used during training
    train_config = log['config']

    model = models.Model(train_config, num_tokens)
    # when model trains using nn.DataParallel
    model = nn.DataParallel(model)
    dict_weights = log['weights']
    model.load_state_dict(dict_weights)
    
    ''' make prediction '''
    
    out = model(att_feat, question.unsqueeze(0), torch.tensor([q_length]))

    _, answer = out.data.cpu().max(dim=1)

    ''' convert prediction into textual answer '''
    # answer vocab
    ans_to_id = vocabs['answer']
    # need to translate answers ids into answers
    id_to_ans = {idx: ans for ans, idx in ans_to_id.items()}
    ans = np.array(answer, dtype='int_')
    ans = id_to_ans[ans[0]]
    return ans



