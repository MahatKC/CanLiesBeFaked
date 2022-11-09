import pandas as pd

df = pd.read_csv("Both Datasets/videos_by_individuals.csv")
files_list = df['FILE'].values.tolist()
individuals_list = df['INDIVIDUAL'].values.tolist()

folds = []

for i in range(5):
    fold = []
    with open("Both Datasets/fold_"+str(i)+".txt", "r") as f:
        for line in f:
            x = line.split(' ')
            video = x[0][:-4]
            video_idx = files_list.index(video)
            individual = individuals_list[video_idx]
            if individual not in fold:
                fold.append(individual)
    folds.append(fold)

for fold in folds:
    fold.sort()
    print(fold, len(fold))