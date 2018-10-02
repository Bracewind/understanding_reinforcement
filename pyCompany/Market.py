class Market(object):
    def __init__(self):
        self.productMarkets = {}
        
    def addProductMarket(self, productConceptName, funct):
        self.productMarkets[productConceptName] = funct
        
    def sellProduct(self, company, productConcept):
        func = self.productMarkets[productConcept.name]
        quantity = round(func(company.productPrices[productConcept.name]))
        company.soldProduct(productConcept, quantity)
        print("the stock is composed of", company.stock.stock)
        print("capital: ", company.capital)

    def sellProducts(self, company):
        for product in company.listOfProductConcepts:
            self.sellProduct(company, product)



    
