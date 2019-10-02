import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

group = parser.add_argument_group('Data')
group.add_argument('-train', default="/home/liu/work/guess_lang/test_100w/tmp/",help="train data")
group.add_argument('-files', default="*.txt",help="train data")
group.add_argument('-save_data',default="/home/liu/work/guess_lang/test_100w/data", help="save a model to path")
group.add_argument('-save_train',default="/home/liu/work/guess_lang/test_100w/train_adam", help="save a model to path")
group.add_argument('-eval',default="/gs1/liu/work/guess_lang/eval", help="test data of dir")
group.add_argument('-max_length',default=1000, help="")

group = parser.add_argument_group('predict')
group.add_argument('-vocab',default="/gs1/liu/work/guess_lang/test_100w/data/vocab.pt", help="vocab data")
group.add_argument('-model',default="/gs1/liu/work/guess_lang/test_100w/sgd_train_rnn1024/model_39000_0.0002.pt", help="vocab data")

group = parser.add_argument_group('train')
group.add_argument('-steps', default=100000,help="train all steps")
group.add_argument('-every', default=1000,help="every step")
group.add_argument('-gpu', type=str, default="1",help="use gpu id")

group = parser.add_argument_group('model')
group.add_argument('-n_hidden', default=1024,help="hidden size")
group.add_argument('-learning_rate', type=float, default=0.001,help="Starting learning rate")
group.add_argument('-loss', type=str, default="Adam",help="[sgd; Adam]")


hparam = parser.parse_args()
