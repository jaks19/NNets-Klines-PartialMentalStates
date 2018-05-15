labels = [
'apple',
'pear',
'bread',
'cheese',
'tomato',
'broccoli',
'pumpkin',
'grapes',
'cucumber',
'lettuce',
'durian',
'squash',
'rice',
'banana',
'carrot',
'mango',
'black beans',
'jackfruit',
'fish',
'avocado',
'century eggs',
'tofu',
'custard',
'yoghurt',
'seaweed',
'wood block',
'beads',
'mud',
'old socks',
'Play Dough',
'sponge',
'balloon',
'rose (flower)',
'paper',
'glass',
'bottle',
'slime',
'glue stick',
'chalk',
'football',
'tennis ball',
'table',
'clown nose',
'ipad',
'rope',
]

names = {}
for i in range(len(labels)):
    names[i] = labels[i]

numbers = {}
for k in names.keys():
    item = names[k]
    numbers[item] = k