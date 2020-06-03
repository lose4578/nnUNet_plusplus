class Config:
    def __init__(self):
        self.name = 'unet++_14_6_DC_CE_GA_big'
        self.path = '../tb/' + self.name
        self.comment = 'unet++ patch:172 batch:4 filters:6'
        self.debug = None
        self.val_num = 30
        self.GA_size = 10
        self.val_name = ''
