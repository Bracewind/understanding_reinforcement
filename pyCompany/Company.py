
from Stock import *
from productConcept import *

class Company(object):
    def __init__(self):
        self.stock = Stock()
        self.capital = 0
        self.productPrices = {}
        self.listOfProductConcepts = []

    def addProductConcept(self, productConcept):
        self.listOfProductConcepts.append(productConcept)

    def produceProducts(self, productConcept, quantity):
        cost = productConcept.productionCost
        if cost*quantity > self.capital :
            try:
                raise CompanyError
            except CompanyError:
                print("Not possible to spend more money than you have")
                raise
        else :
            self.stock.addProducts(productConcept.name, quantity)
            self.capital -= cost*quantity

    def fixPrice(self, productName, price):
        if price < 0 :
            try:
                raise CompanyError
            except CompanyError:
                print("not possible to fix a negative price")
                raise
        else:
            self.productPrices[productName] = price
    
    def soldProduct(self, ProductConcept, quantity):
        realQuantity = min(quantity, self.stock.stock[ProductConcept.name])
        self.stock.stock[ProductConcept.name] -= realQuantity
        self.capital += realQuantity * self.productPrices[ProductConcept.name]
        
        
