import json
import random
random.seed(58)
json_path ='commonsense_170k.json'

with open(json_path) as file:
    json_data = json.load(file)
len_input = set([len(j['input']) for j in json_data])
val_set_size = 120
random.shuffle(json_data)

train_data = json_data[:-val_set_size]
val_data = json_data[-val_set_size:]

with open('train.json', 'w') as file:
    json.dump(train_data,file, indent=4)

with open('validation.json', 'w') as file:
    json.dump(val_data,file, indent=4)

with open('test.json', 'w') as file:
    json.dump(val_data,file, indent=4)