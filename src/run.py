import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn import functional as F
import random
import argparse
random.seed(0)

import dataset
import model
import trainer
import utils

argp = argparse.ArgumentParser()
argp.add_argument('function',
    help="Whether to pretrain, finetune or evaluate a model",
    choices=["pretrain", "finetune", "evaluate"])
argp.add_argument('variant',
    help="Which variant of the model to run ('vanilla' or 'synthesizer')",
    choices=["vanilla", "synthesizer"])
argp.add_argument('pretrain_corpus_path',
    help="Path of the corpus to pretrain on", default=None)
argp.add_argument('--reading_params_path',
    help="If specified, path of the model to load before finetuning/evaluation",
    default=None)
argp.add_argument('--writing_params_path',
    help="Path to save the model after pretraining/finetuning", default=None)
argp.add_argument('--finetune_corpus_path',
    help="Path of the corpus to finetune on", default=None)
argp.add_argument('--eval_corpus_path',
    help="Path of the corpus to evaluate on", default=None)
argp.add_argument('--outputs_path', default=None)
args = argp.parse_args()

device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

block_size = 128
text = open(args.pretrain_corpus_path).read()
pretrain_dataset = dataset.CharCorruptionDataset(text, block_size)

mconf = model.GPTConfig(pretrain_dataset.vocab_size, pretrain_dataset.block_size,
    n_layer=4, n_head=8, n_embd=256)

"""
Don't change above here; write your code below
"""

if args.variant == 'vanilla':
    model = model.GPT(mconf)
    model = model.to(device)
    
    
elif args.variant == 'synthesizer':
    mconf = model.GPTConfig(pretrain_dataset.vocab_size, pretrain_dataset.block_size,
    n_layer=4, n_head=8, n_embd=256,synthesizer = True)
    model = model.GPT(mconf)
    model = model.to(device)

if args.function == 'pretrain':
    assert args.pretrain_corpus_path is not None
    assert args.writing_params_path is not None

    tconf = trainer.TrainerConfig(max_epochs=650, batch_size=128,
                                learning_rate=6e-3, lr_decay=True, warmup_tokens=512*20,
                                final_tokens=200*len(pretrain_dataset)*block_size,num_worker=4) 
            
    ft_trainer = trainer.Trainer(model, pretrain_dataset,None,tconf) #create a trainer object
    ft_trainer.train()
    torch.save(model.state_dict(),args.writing_params_path) # save model state to this path

elif args.function == 'finetune':
    assert args.writing_params_path is not None
    assert args.finetune_corpus_path is not None

    tconf = None
    if args.reading_params_path : # contains pretrained model
        
        model.load_state_dict(torch.load(args.reading_params_path))
        model = model.to(device)

        tconf = trainer.TrainerConfig(max_epochs=10, batch_size=256,
                            learning_rate=6e-4, lr_decay=True, warmup_tokens=512*20,
                            final_tokens=200*len(pretrain_dataset)*block_size,num_workers=4)

    else:
        # fine-tune without pre-training
        tconf = trainer.TrainerConfig(max_epochs=75, batch_size=256,
                            learning_rate=6e-4, lr_decay=True, warmup_tokens=512*20,
                            final_tokens=200*len(pretrain_dataset)*block_size)
            
    train_dataset = dataset.NameDataset(pretrain_dataset, open(args.finetune_corpus_path,'r').read())
    ft_trainer = trainer.Trainer(model, train_dataset,None,tconf)
    ft_trainer.train()
    torch.save(model.state_dict(),args.writing_params_path)


elif args.function == 'evaluate':
        assert args.outputs_path is not None
        assert args.reading_params_path is not None
        assert args.eval_corpus_path is not None
        model.load_state_dict(torch.load(args.reading_params_path))
        model = model.to(device)
        correct = 0
        total = 0
        with open(args.outputs_path, 'w') as fout:
            predictions = []
            for line in tqdm(open(args.eval_corpus_path)):
                x = line.split('\t')[0]
                x = x + '⁇'
                x = torch.tensor([pretrain_dataset.stoi[s] for s in x], dtype=torch.long)[None, ...].to(device)
                pred = utils.sample(model, x, 32, sample=False)[0]
                completion = ''.join([pretrain_dataset.itos[int(i)] for i in pred])
                pred = completion.split('⁇')[1]
                predictions.append(pred)
                fout.write(pred + '\n')
            total, correct = utils.evaluate_places(args.eval_corpus_path, predictions)
        if total > 0:
            print('Correct: {} out of {}: {}%'.format(correct, total, correct / total * 100))
        else:
            print('Predictions written to {}; no targets provided'.format(args.outputs_path))


