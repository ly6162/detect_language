from hparam import hparam

import os
import json

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
if __name__ == '__main__':
    print("dd")
    f = open(os.path.join(hparam.save_train,'Adam_loss.json'), 'r',encoding='utf-8')
    all_losses=json.load(f)
    plt.figure()
    plt.plot(all_losses)
    plt.show()




