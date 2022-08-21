import glob
import os
import json

root_dir = os.path.join(os.getcwd(), "data/plan_waypoints")

one_hot_set = set([])
for filename in glob.iglob(root_dir + '**/**', recursive=True):
    # print(filename)
    if os.path.isfile(filename):
        with open(filename,'r') as file:
            # print(file.read())
            sentence = file.read()
            words = sentence.split(" ") # temporary
            for w in words:
                one_hot_set.add(w.strip().lower())

one_hot_set.remove('')

one_hot_dict = {}
i = 0
for w in one_hot_set:
    one_hot_dict[w] = i
    i += 1

with open(os.path.join(os.getcwd(), "data/one_hot_dict/dict.txt"), 'w') as dict_file:
     dict_file.write(json.dumps(one_hot_dict))
print((one_hot_dict))
    
# with open(os.path.join(os.getcwd(), "data/one_hot_dict/dict.txt")) as json_file:
#     data = json.load(json_file)
#     # print(type(data[0]), type(data[1]))
#     print(type(data['reach']))
