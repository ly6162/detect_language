import torch
import os
import sys
import random
from torch.autograd import Variable
from model import *
import lib.utils as utils
from hparam import hparam
from lib.logging import init_logger, logger

init_logger(utils.get_log(hparam.save_train))
sys.path.append(os.path.join(os.path.dirname(__file__), './'))

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_data():
    logger.info("load data")

    path=os.path.join(hparam.save_data, "vocab.pt")
    vocab = torch.load(path)
    logger.info("load vocab: %s size: %s"%(path,vocab.size))

    path=os.path.join(hparam.save_data, "train.pt")
    data=torch.load(path)
    logger.info("load vocab: %s "%path)

    return data,vocab

def categoryFromOutput(vocab,output):
    output=output.cpu()
    _, top_i = output.data.topk(1) # Tensor out of Variable with .data
    label_i = top_i.item()
    return vocab.label["iton"][label_i], label_i

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingPair(data,vocab):
    #ランダムに教師データのラベルを選定
    label_name = randomChoice(list(vocab.label["ntoi"].keys()))
    #教師データを取得
    label_tensor = vocab.label["ntoi"][label_name]

    #ランダムに学習データを選定
    line = randomChoice(data.train[label_name])
    #学習データ
    line_tensor=utils.lineToTensor(vocab.size,line)
    if torch.cuda.is_available():
        label_tensor=label_tensor[2]


    return label_name,label_tensor,line, line_tensor

def train(model,label_tensor, line_tensor):
    hidden = model.rnn.initHidden()

    model.optimizer.zero_grad()
    #TODO
    for i in range(line_tensor.size()[0]):
        output, hidden = model.rnn(line_tensor[i], hidden)
    loss = model.criterion(output, label_tensor)
    loss.backward()
    model.optimizer.step()
    return output, loss.item()

def itos(line,vocab):
    new_line = []
    for id in line:
        if id in vocab.itos:
            char = vocab.itos[id]
        else:
            char = utils.UNK
        new_line.append(char)
    return "".join(new_line)

import json
def do_epoch(model,data,vocab):
    before_loss = 0
    all_losses = []
    current_loss = 0
    for epoch in range(1, hparam.steps + 1):
        label_name,label_tensor,line,line_tensor = randomTrainingPair(data,vocab)
        output, loss = train(model,label_tensor, line_tensor)
        current_loss += loss
        # Print epoch number, loss, name and guess
        if epoch % hparam.every == 0:
            ll=itos(line,vocab)
            guess, guess_i = categoryFromOutput(vocab,output)
            correct = '✓' if guess == label_name else '✗ (%s)' % label_name
            logger.info('step:%d %.2f%% loss: %.4f %s / %s %s' % (epoch, epoch / hparam.steps * 100,  loss, ll, guess, correct))
            if loss<before_loss or before_loss==0:
                ckpt="model_%s_%.4f.pt"%(epoch,loss)
                utils.save(logger, model.rnn, hparam.save_train, ckpt)
                before_loss=loss

            all_losses.append(current_loss / hparam.every)
            current_loss = 0
            f = open(os.path.join(hparam.save_train,'%s_loss.json'%hparam.loss), 'w',encoding='utf-8')
            json.dump(all_losses,f,indent=0)


def main():

    data, vocab = load_data()
    model=Model(vocab.size,vocab.label["size"])
    logger.info("model: [rnn_size: %s learning_rate: %s loss: %s ]"%(hparam.n_hidden,hparam.learning_rate,hparam.loss))
    do_epoch(model,data,vocab)

if __name__ == '__main__':
    main()