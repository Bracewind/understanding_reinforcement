from companyException import StockError

class Stock(object):
    def __init__(self):
        self.stock = {}
        
    def addProductConcept(self, nameOfProductConcept):
        self.stock[nameOfProductConcept] = 0
        
    def addProducts(self, nameOfProductConcept, numberOfProducts):
        try:
            self.stock[nameOfProductConcept] += numberOfProducts
        except KeyError:
            print("this name of product concept not exists yet !")
            raise

    def removeProducts(self, nameOfProductConcept, numberOfProducts):
        try:
            newValue = self.stock[nameOfProductConcept] - numberOfProducts
        except KeyError:
            print("this name of product concept not exists yet !")
            raise
        if newValue < 0:
            try:
                raise StockError
            except StockError:
                print("transaction not allowed : you can't remove more products than you get")
                raise
        else :
            self.stock[nameOfProductConcept] = newValue
            
        
