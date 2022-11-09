from cgi import test
from random import triangular
import pandas as pd
import numpy as np

def write_set_to_file(set, set_name):
    with open("Real-life_Deception_Detection_2016/"+set_name+".txt", "w") as f:
        for element in set:
            if element == set[-1]:
                skip_line = ""
            else:
                skip_line = "\n"
            if element[6]=='t':
                f.write(element+" 500 1"+skip_line)
            else:
                f.write(element+" 500 0"+skip_line)
        f.close()

if __name__=="__main__":
    #reads CSV with the list of videos x individuals and randomizes it so proper train, val and test txt files are created
    df = pd.read_csv("Real-life_Deception_Detection_2016/videos_by_individuals.csv")
    chosen_individuals = []
    val_count = 0
    test_count = 0

    while(val_count<16 or test_count<16):
        if len(chosen_individuals)==51 or len(chosen_individuals)==0:
            chosen_individuals = []
            val_count = 0
            val_lies = 0
            val_truths = 0
            test_count = 0
            test_lies = 0
            test_truths = 0
            val_vids = []
            test_vids = []

        random_index = np.random.randint(110)
        individual_index = int(df["INDIVIDUAL"][random_index])

        while(individual_index in chosen_individuals):
            random_index = np.random.randint(110)
            individual_index = df["INDIVIDUAL"][random_index]
        chosen_individuals.append(individual_index)

        lies = int(df["LIES"][random_index])
        truths = int(df["TRUTHS"][random_index])

        query_df = df.loc[df['INDIVIDUAL'] == individual_index]
        individual_videos = query_df['FILE'].values.tolist()

        if (val_lies+lies)<=8 and (val_truths+truths)<=8:
            val_lies += lies
            val_truths += truths
            val_count += lies + truths
            for individual_video in individual_videos:
                val_vids.append(individual_video+".mp4")
        elif (test_lies+lies)<=8 and (test_truths+truths)<=8:
            test_lies += lies
            test_truths += truths
            test_count += lies + truths
            for individual_video in individual_videos:
                test_vids.append(individual_video+".mp4")

    train_vids = []

    all_files = df['FILE'].values.tolist()
    for file in all_files:
        name = file+".mp4"
        if name not in val_vids and name not in test_vids:
            train_vids.append(name)

    val_vids.sort()
    test_vids.sort()
    train_vids.sort()
            
    write_set_to_file(val_vids, "val")
    write_set_to_file(test_vids, "test")
    write_set_to_file(train_vids, "train")