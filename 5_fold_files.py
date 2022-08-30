filenames = ['fold_0.txt', 'fold_1.txt', 'fold_2.txt', 'fold_3.txt', 'fold_4.txt']

for i in range(5):
    with open('Both Datasets/train_with_fold_'+str(i)+'_as_test.txt', 'w') as outfile:
        for fname in filenames:
            if fname == 'fold_'+str(i)+'.txt':
                continue
            else:
                with open('Both Datasets/'+fname) as infile:
                    for line in infile:
                        outfile.write(line)
                    outfile.write('\n')