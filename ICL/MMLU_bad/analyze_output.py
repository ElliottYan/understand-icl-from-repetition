import sys

inf = sys.argv[1]

with open(inf, 'r', encoding='utf8') as f:
    lines = f.readlines()

i = 0
while i < len(lines):
    fn = lines[i]
    # parse fn
    splits = fn.split('/')
    ratio = float(splits[1])
    
    
    accs = lines[i+1]
    i += 3