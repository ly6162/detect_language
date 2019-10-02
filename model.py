import torch
import torch.nn as nn
from torch.autograd import Variable
from hparam import hparam

class Model():
    def __init__(self, vocab_size, label_size):
        
        self.rnn = RNN(vocab_size, hparam.n_hidden, label_size)
        #self.rnn = SimpleLSTM(vocab_size, hparam.n_hidden, label_size)
        if hparam.loss=="sgd":
            self.optimizer = torch.optim.SGD(self.rnn.parameters(), lr=hparam.learning_rate)
        elif hparam.loss=="Adam":
            self.optimizer = torch.optim.Adam(self.rnn.parameters(), lr=hparam.learning_rate,betas=(0.9, 0.998))
        
        self.criterion = nn.NLLLoss()
        self.vocab_size=vocab_size
        self.label_size=label_size
        if torch.cuda.is_available():
            self.rnn.cuda()
            self.criterion.cuda()

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self,hidden_size):
        if torch.cuda.is_available():
            tensor=Variable(torch.zeros(1, hidden_size,device='cuda:0'))
        else:
            tensor = Variable(torch.zeros(1, hidden_size))
        return tensor

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)  # outは3Dtensorになるのでdim=2

    def forward(self, input, h):
        # nn.RNNは系列をまとめて処理できる
        # outputは系列の各要素を入れたときの出力
        # hiddenは最後の隠れ状態（=最後の出力） output[-1] == hidden[0]
        output, (h, c) = self.lstm(input, h)

        # RNNの出力がoutput_sizeになるようにLinearに通す
        output = self.out(output)

        # 活性化関数
        output = self.softmax(output)

        return output, (h, c)

    def initHidden(self):
        # 最初に入力する隠れ状態を初期化
        # LSTMの場合は (h, c) と2つある
        # (num_layers, batch, hidden_size)
        h = Variable(torch.zeros(1, 1, self.hidden_size))
        c = Variable(torch.zeros(1, 1, self.hidden_size))
        return (h, c)

