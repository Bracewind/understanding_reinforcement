from Interface import *
if __name__ == '__main__':
    inter = Interface()
    inter.make()
    inter.step()
    print("the stock is composed of", inter.company.stock.stock)
    print("capital: ", inter.company.capital)