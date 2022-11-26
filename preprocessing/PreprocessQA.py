import torch

def tokenize_questions(annotations):
    
    questions = [q['question'] for q in annotations]
    prepared = []
    
    # Tokenize each question
    for question in questions:
        # lower case
        question = question.lower()
        # remove question mark
        question = question[:-1]
        # sentence to list
        question = question.split(' ')
        # remove empty strings
        question = list(filter(None, question))
        
        prepared.append(question)
    # a list of vectors of question tokens
    return prepared


def tokenize_answers(annotations):
    answers = [[a['answer'] for a in ans_dict['answers']] for ans_dict in annotations]
    prepared = []
   
    # 10 answers for each question, tokenize all 10
    for answer in answers:
        prepared2 = []
        for word in answer:
            # lower case
            word = word.lower()
            
            prepared2.append(word)

        prepared.append(prepared2)
        
    return prepared

def encode_ques(ques, to_idx, max_len):
    # question is broken down into tokens
    # the indices of tokens in vocab are retrieved 
    # then encoded into a vector of indices 
    # 0 when the token is not found in vocab OR 
    # 0 in empty spaces when tokens are less than maxlength
    vec = torch.zeros(max_len).long()
    length = min(len(ques), max_len)
    for i in range(length):
        token = ques[i]
        index = to_idx.get(token, 0)
        vec[i] = index
    
    # return the vector and its length 
    # empty encoded questions are problematic during pack padded sequence,
    # by setting min length = 1, a 0 token can be fed to RNN
    # where the token 0 does not represent a word
    return vec, max(length, 1)

def encode_answers(answers, to_idx):
    # a vector of answer tokens and its count 
    # for each question containing the occurances of each answer
    # to determine which answers among all contribute to the loss.
    # should be multiplied with 0.1 * negative log-likelihoods and summed up
    # to get the loss that is weighted by how many humans gave that answer
    answer_vec = torch.zeros(len(to_idx))
    for answer in answers:
        index = to_idx.get(answer)
        if index is not None:
            answer_vec[index] += 1
    return answer_vec
