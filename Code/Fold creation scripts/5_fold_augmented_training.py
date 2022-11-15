import os
from pathlib import Path

upper_dir = Path(os.getcwd()).parents[0]

def create_augmented_folds(dataset_name, dataset_dir, other_dataset_whole_path):
    

    for i in range(5):
        with open(str(upper_dir)+"/Augmented "+dataset_name+"/train_with_fold_"+str(i)+".txt", 'w') as outfile:
            with open(str(upper_dir)+"/"+dataset_dir+"/train_with_fold_"+str(i)+"_as_test.txt") as infile:
                for line in infile:
                    outfile.write(line)
            with open(str(upper_dir)+other_dataset_whole_path) as infile:
                for line in infile:
                    outfile.write(line)
        with open(str(upper_dir)+"/Augmented "+dataset_name+"/fold_"+str(i)+".txt", 'w') as outfile:
            with open(str(upper_dir)+"/"+dataset_dir+"/fold_"+str(i)+".txt") as infile:
                for line in infile:
                    outfile.write(line)

create_augmented_folds('RLT', "Real-life_Deception_Detection_2016", "/Box of Lies Vids/whole_bol.txt")
create_augmented_folds('BoL', "Box of Lies Vids", "/Real-life_Deception_Detection_2016/whole_rlt.txt")