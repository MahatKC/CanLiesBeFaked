The code runs on main.py, which was created considering manual runs of hyperparameter combinations for the network. Be sure to adjust the hyperparameter values at the end of main.py prior to running it.

GluonCV, Pandas, NumPy and Tensorboard must be installed for the code to run.

Also, Real-life Trial clips and Box of Lies clips should be save in a Clips folder within their respective folders (Real-life_Deception_Detection_2016 and 'Box of Lies Vids'), following the name convention in each folder's CSV.

Prior to running main.py, the TXT files used by GluonCV to import the videos should be created by running the desired script in the 'Fold creation scripts' folder. Mind you, each time the script runs, a new randomized 5-fold separation is created.

As it stands, results from main.py will be stored in this folder, but the results we got while running it are stored in a separate Results folder for safekeeping.