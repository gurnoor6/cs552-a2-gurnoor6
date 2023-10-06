"""
Script to combine the annotated dataset of me and my partner
I choose my own labels
"""
import jsonlines
from tqdm import tqdm

samples, labels_1, labels_2 = [], [], []
with jsonlines.open("03-366788.jsonl") as reader:
    for sample in tqdm(reader.iter()):
        labels_1.append(sample['label_student1'])

with jsonlines.open("03-337560.jsonl") as reader:
    for sample in tqdm(reader.iter()):
        labels_2.append(sample['label_student2'])
        samples.append(sample)


i = 0
for a, b, c in zip(labels_1, labels_2, samples):
    if a != b:
        i += 1
        print(c)
        print(a, b)
print(i)