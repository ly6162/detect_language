from model import *
import os
from tqdm import tqdm
from hparam import hparam
import lib.utils as utils
from lib.logging import init_logger, logger

init_logger(utils.get_log(hparam.eval))

rnn = utils.load_data(hparam.model,logger)
vocab = utils.load_data(hparam.vocab,logger)
n_hidden=1024
# Just return an output given a line
def evaluate(line_tensor):
    hidden = rnn.initHidden(n_hidden)
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    return output

def predict(line,label):
    li = utils.stoi(line, vocab)
    char_tensor=utils.lineToTensor(vocab.size,li)
    output = evaluate(char_tensor)
    output=output.cpu()
    # Get top N categories
    topv, topi = output.data.topk(1, 1, True)
    out=vocab.label["iton"][topi.item()]
    return out

def eavl():
    for id,filename in enumerate(utils.findFiles(os.path.join(hparam.eval, hparam.files))):
        label = filename.split('/')[-1].split('.')[0]
        lines=utils.readLines(filename)
        logger.info("eval %s"%filename)
        err=[]
        for l in tqdm(lines):
            out=predict(l,label)
            if out!=label:
                err.append([l,out])
                #print(out)
        acc=(1-(len(err)/len(lines)))*100
        logger.info("accuracy: %s%%"%acc)

if __name__ == '__main__':
    input="Liu"
    #predict(input)
    eavl()