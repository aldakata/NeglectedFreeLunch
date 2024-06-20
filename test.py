def test(path, loss):
    def loader(path):
        return path, loss[path]
    return loader

if __name__=="__main__":
    path = 10
    loss = [i for i in range(20)]
    load = test(path, loss)
    print(load)
    print(load(path))