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

def reset(df, chosen_individuals, fold_counts, lie_counts, truth_counts, fold_vids):
    chosen_individuals = []
    fold_counts = [0]*5
    lie_counts = [0]*5
    truth_counts = [0]*5
    fold_vids = [[],[],[],[],[]]
    
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

    indiv_order = [3, 2, 1, 22, 7, 12, 25, 30, 36, 37]
    indiv_list_index = [42,18,0,138,90,112,158,170,184,188]
    indiv_index = 0

    while(sum(fold_counts)<220):
        if len(chosen_individuals)==51 or len(chosen_individuals)==0:
            chosen_individuals, fold_counts, lie_counts, truth_counts, fold_vids = reset(df, chosen_individuals, fold_counts, lie_counts, truth_counts, fold_vids)

        if indiv_index==10:
            random_index = np.random.randint(220)
            individual_index = int(df["INDIVIDUAL"][random_index])

            while individual_index in chosen_individuals:
                random_index = np.random.randint(220)
                individual_index = int(df["INDIVIDUAL"][random_index])
        else:
            individual_index = indiv_order[indiv_index]
            random_index = indiv_list_index[indiv_index]
            indiv_index += 1

        chosen_individuals.append(individual_index)

        number_lies = int(df["LIES"][random_index])
        number_truths = int(df["TRUTHS"][random_index])

        query_df = df.loc[df['INDIVIDUAL'] == individual_index]
        individual_videos = query_df['FILE'].values.tolist()

        if number_lies<number_truths:
            individual_videos.sort(reverse=True)

        if individual_index == 3:
            current_fold = np.random.randint(5)
        else:
            current_fold = np.argmin(fold_counts)

        for i in range(len(individual_videos)):
            #9 lies 13 truths
            if individual_videos[i][6] == 'l':
                while (lie_counts[current_fold]==22 and current_fold<4) or (lie_counts[current_fold]==18 and current_fold==4):
                    current_fold = (current_fold+1)%5
                lie_counts[current_fold] += 1

            else:
                while (truth_counts[current_fold]==22 and current_fold<4) or (truth_counts[current_fold]==26 and current_fold==4):
                    current_fold = (current_fold+1)%5
                truth_counts[current_fold] += 1

            fold_vids[current_fold].append(individual_videos[i]+".mp4")
            fold_counts[current_fold]+=1
            current_fold = (current_fold+1)%5
        
    for fold_vid in fold_vids:
        fold_vid.sort()
            
    for i in range(5):
        write_set_to_file(fold_vids[i], "foldBF_"+str(i))