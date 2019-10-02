from hparam import hparam
import lib.utils as utils
import os
from tqdm import tqdm
import codecs
import torch
from lib.logging import init_logger, logger

init_logger(utils.get_log(hparam.save_train))

def make_vocab():
    chars=set()
    ntoi, iton = {}, {}
    for id, filename in enumerate(utils.findFiles(os.path.join(hparam.train, hparam.files))):
        category = filename.split('/')[-1].split('.')[0]
        ntoi[category]=utils.id2tensor(id)
        iton[id]=category
        logger.info("proess: %s"%filename)
        lines = utils.readLines(filename)
        label = {"size": len(iton), "ntoi": ntoi, "iton": iton}
        for line in tqdm(range(len(lines))):
            char=set(list(lines[line]))
            chars=chars.union(char)
    logger.info("label size: [%s]" % len(iton))
    return chars,label

def load_vocab():

    if os.path.exists(hparam.vocab):
        vocab=torch.load(hparam.vocab)
    else:
        logger.info("make vocab")
        vocab,label=make_vocab()
        if not os.path.exists(hparam.save_data):
            os.mkdir(hparam.save_data)
        vocab_path=os.path.join(hparam.save_data,"vocab.txt")
        with codecs.open(vocab_path,'w',encoding="utf-8") as f:
            f.write(utils.UNK + "\n")
            for char in vocab:
                f.write(char+"\n")
        vocab= utils.readLines(vocab_path)
        logger.info("load vocab...size: [%s]" % len(vocab))
        vocab= utils.vocabulary(label,vocab)
        utils.save(logger, vocab, hparam.save_data, "vocab.pt")

    return vocab

def make_data(vocab,path,flag):
    label_lines = {}
    logger.info("process %s data..."%flag)
    logger.info("load vocab...size: [%s]" % vocab.size)
    for id,filename in enumerate(utils.findFiles(os.path.join(path, hparam.files))):
        logger.info("proess file: %s"%filename)
        category = filename.split('/')[-1].split('.')[0]
        lines = utils.readLines(filename)
        new_lines=[]
        for id in tqdm(range(len(lines))):
            line=lines[id]
            if flag=="train" :
                if len(line)<hparam.max_length:
                    chars_id=utils.stoi(line,vocab)
                    new_lines.append(chars_id)
                    label_lines[category] = new_lines
                else:
                    logger.info("not proess: %s"%len(line))
            else:
                chars_id=utils.stoi(line,vocab)
                new_lines.append(chars_id)
                label_lines[category] = new_lines

    logger.info("process: %s size: [%s]"%(category,len(new_lines)))
    data=utils.data(label_lines)
    utils.save(logger,data, hparam.save_data, "%s.pt"%flag)


if __name__ == '__main__':

    logger.info("make data start")
    vocab=load_vocab()
    make_data(vocab,hparam.train,"train")
    make_data(vocab,hparam.eval,"test")
    logger.info("make data end")

