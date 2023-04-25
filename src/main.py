import configparser

import train

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('../config.ini')

    model, data = train.train_baseline(config,
                                       "../tmp/fake-news/train.csv",
                                       "../tmp/fake-news/test.csv",
                                       "../tmp/auto.pt")

