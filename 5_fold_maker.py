from cv2 import add
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

def add_video(fold_index, number_lies, number_truths, individual_videos, fold_counts, lie_counts, truth_counts, fold_vids):
    lie_counts[fold_index] += number_lies
    truth_counts[fold_index] += number_truths
    fold_counts[fold_index] += number_lies + number_truths
    for individual_video in individual_videos:
        fold_vids[fold_index].append(individual_video+".mp4")
    
    return fold_counts, lie_counts, truth_counts, fold_vids

def reset(df, chosen_individuals, fold_counts, lie_counts, truth_counts, fold_vids):
    chosen_individuals = []
    fold_counts = [0]*5
    lie_counts = [0]*5
    truth_counts = [0]*5
    fold_vids = [[],[],[],[],[]]

    query_df = df.loc[df['INDIVIDUAL'] == 3]
    individual_videos = query_df['FILE'].values.tolist()
    chosen_individuals.append(3)

    fold_counts, lie_counts, truth_counts, fold_vids = add_video(0, 18, 3, individual_videos, fold_counts, lie_counts, truth_counts, fold_vids)
    
    return chosen_individuals, fold_counts, lie_counts, truth_counts, fold_vids

if __name__=="__main__":
    #reads CSV with the list of videos x individuals and randomizes it so proper train, val and test txt files are created
    df = pd.read_csv("Real-life_Deception_Detection_2016/videos_by_individuals.csv")
    files_list = df['FILE'].values.tolist()
    files = []
    for file in files_list:
        files.append(file+".mp4")
    
    chosen_individuals = []
    fold_vids = [[],[],[],[],[]]
    fold_counts = [0]*5
    lie_counts = [0]*5
    truth_counts = [0]*5

    while(sum(fold_counts)<110):
        if len(chosen_individuals)==51 or len(chosen_individuals)==0:
            chosen_individuals, fold_counts, lie_counts, truth_counts, fold_vids = reset(df, chosen_individuals, fold_counts, lie_counts, truth_counts, fold_vids)

        random_index = np.random.randint(110)
        individual_index = int(df["INDIVIDUAL"][random_index])

        while(individual_index in chosen_individuals):
            random_index = np.random.randint(110)
            individual_index = df["INDIVIDUAL"][random_index]
        chosen_individuals.append(individual_index)

        number_lies = int(df["LIES"][random_index])
        number_truths = int(df["TRUTHS"][random_index])

        query_df = df.loc[df['INDIVIDUAL'] == individual_index]
        individual_videos = query_df['FILE'].values.tolist()

        if (number_truths == 1 and number_lies == 0) and (fold_counts[0]<22):
            
            fold_counts, lie_counts, truth_counts, fold_vids = add_video(0, number_lies, number_truths, individual_videos, fold_counts, lie_counts, truth_counts, fold_vids)
        else:
            for fold_index in range(1,5):
                if (lie_counts[fold_index]+number_lies)<=11 and (truth_counts[fold_index]+number_truths)<=11:
                    fold_counts, lie_counts, truth_counts, fold_vids = add_video(fold_index, number_lies, number_truths, individual_videos, fold_counts, lie_counts, truth_counts, fold_vids)
                    break
                elif (fold_index == 4) and (lie_counts[fold_index]+number_lies)<=2 and (truth_counts[fold_index]+number_truths)<=20:
                    fold_counts, lie_counts, truth_counts, fold_vids = add_video(fold_index, number_lies, number_truths, individual_videos, fold_counts, lie_counts, truth_counts, fold_vids)
                    break
                elif (fold_index == 4):
                    print(individual_index)
                    print(len(chosen_individuals), fold_counts, lie_counts, truth_counts)
                    chosen_individuals = []
                    break

    for fold_vid in fold_vids:
        fold_vid.sort()
            
    for i in range(5):
        write_set_to_file(fold_vids[i], "fold_"+str(i))