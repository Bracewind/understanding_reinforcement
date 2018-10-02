from Company import *
from Market import *
import numpy as np
from MatrixDecision import *

def f1(price):
    return 0

def f2(price):
    return 1

def f3(price):
    return 1

def f4(price):
    return 1

class Interface(object):
    def __init__(self):
        self.market = Market()
        self.company = Company()

    def make(self):
        product1 = ProductConcept("product1", 10)
        product2 = ProductConcept("product2", 30)
        product3 = ProductConcept("product3", 50)
        product4 = ProductConcept("product4", 100)
        stock = Stock()
        stock.addProductConcept("product1")
        stock.addProductConcept("product2")
        stock.addProductConcept("product3")
        stock.addProductConcept("product4")
        self.company.stock = stock
        self.company.capital = 1000
        self.company.fixPrice("product1",0)
        self.company.fixPrice("product2",0)
        self.company.fixPrice("product3",0)
        self.company.fixPrice("product4",0)
        self.company.addProductConcept(product1)
        self.company.addProductConcept(product2)
        self.company.addProductConcept(product3)
        self.company.addProductConcept(product4)
        self.market.addProductMarket("product1", f1)
        self.market.addProductMarket("product2", f2)
        self.market.addProductMarket("product3", f3)
        self.market.addProductMarket("product4", f4)

    def step(self):
        matDec = MatrixDecision(4)
        for product in range(matDec.matrix_np.shape[0]):
            productConcept = self.company.listOfProductConcepts[product]
            self.company.produceProducts( productConcept,matDec.matrix_np[product][0])
            self.company.fixPrice(productConcept.name, matDec.matrix_np[product][1])
            print("the stock is composed of", self.company.stock.stock)
            print("capital: ", self.company.capital)
        self.market.sellProducts(self.company)



        
        
    
