from Interface import *

if __name__ == '__main__':
    
    inter = Interface()
    inter.make()
    output = inter.step(vector)

    print("the stock is composed of", inter.company.stock.stock)
    print("capital: ", inter.company.capital)
    print("return :", output)
    print("time", inter.market.time)
    print("\n")

    inter.reset()
    print("after reset")
    print("the stock is composed of", inter.company.stock.stock)
    print("capital: ", inter.company.capital)
    print("time", inter.market.time)
    
