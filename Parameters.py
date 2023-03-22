import os

dim_window = 36  # exemplele pozitive (fete de oameni cropate) au 36x36 pixeli
dim_hog_cell = 6  # dimensiunea celulei
dim_descriptor_cell = 36  # dimensiunea descriptorului unei celule
overlap = 0.3

class Parameters:
    def __init__(self):
        self.dir_test_examples = "C:\\Users\\carja\\Desktop\\testare\\testare"  
        # set the parameters
        self.dim_window = 36  # exemplele pozitive (fete de oameni cropate) au 36x36 pixeli
        self.dim_hog_cell = 6  # dimensiunea celulei
        self.dim_descriptor_cell = 36  # dimensiunea descriptorului unei celule
        self.overlap = 0.3
        self.overlap = 0.3
        self.has_annotations = False
        self.threshold = 0
