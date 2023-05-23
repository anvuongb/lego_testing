import numpy as np
import pandas as pd
import ast
import json

with open("color_codes", "r") as f:
    lines = f.readlines()
lines = [l.rsplit() for l in lines]
color_dict = {}

for l in lines:
    color_dict[ast.literal_eval(l[4])]=l[6]
    
with open("color_codes.json", "w") as f:
    json.dump(color_dict, f, sort_keys=True, indent=4)
    
print(color_dict)