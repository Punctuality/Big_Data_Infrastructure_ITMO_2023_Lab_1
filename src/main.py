import train

if __name__ == "__main__":
    model, data = train.train_baseline("../tmp/fake-news/train.csv", "../tmp/fake-news/test.csv", "../tmp/auto.pt")

