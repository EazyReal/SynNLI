import config
import model
import utils

# step 1 preprocess
preprocess.utils.process_data(data_file, p_data_file)

# step 2 load to mem
reader = Reader()
dataset = reader.read(file_path)
loader = DataLoader(dataset, batch_size) # this part is tricky
# followbatch?
# i think should be dense Graph Encoder
# or change size in graph encoder

# step 3 build flexible model

class Model():
    def forward(batch):
        batch = self.embedding(batch)
        cls = self.gmn(batch)
        
        
        
        

model = Model()

# step 4 train by config and do statistics
allennlp train -s ...
# make sure to use tensorflow

# step 5 error analysis and redo 4

# paper writinh