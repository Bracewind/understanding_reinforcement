from Interface import *

if __name__ == '__main__':
    
    inter = Interface()
    inter.make()
    output = inter.step(vector)
    '''inter.company.produceProducts(inter.company.listOfProductConcepts[0], 10)
    inter.company.produceProducts(inter.company.listOfProductConcepts[1], 10)
    print("the stock is composed of", inter.company.stock.stock)
    print("capital: ", inter.company.capital)
    inter.company.fixPrice(inter.company.listOfProductConcepts[0].name, 10)
    print("fixed prices :", inter.company.productPrices)
    inter.market.sellProduct(inter.company, inter.company.listOfProductConcepts[0])
    inter.market.sellProduct(inter.company, inter.company.listOfProductConcepts[1])
    print("capital: ", inter.company.capital)
    print("the stock is composed of", inter.company.stock.stock)'''
    
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
    
