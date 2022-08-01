filenames = ['foldBF_0.txt', 'foldBF_1.txt', 'foldBF_2.txt', 'foldBF_3.txt', 'foldBF_4.txt']

for i in range(5):
    with open('Real-life_Deception_Detection_2016/train_with_foldBF_'+str(i)+'_as_test.txt', 'w') as outfile:
        for fname in filenames:
            if fname == 'foldBF_'+str(i)+'.txt':
                continue
            else:
                with open('Real-life_Deception_Detection_2016/'+fname) as infile:
                    for line in infile:
                        outfile.write(line)
                    outfile.write('\n')