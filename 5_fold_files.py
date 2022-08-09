filenames = ['foldBoL_0.txt', 'foldBoL_1.txt', 'foldBoL_2.txt', 'foldBoL_3.txt', 'foldBoL_4.txt']

for i in range(5):
    with open('Box of Lies Vids/train_with_foldBoL_'+str(i)+'_as_test.txt', 'w') as outfile:
        for fname in filenames:
            if fname == 'foldBoL_'+str(i)+'.txt':
                continue
            else:
                with open('Box of Lies Vids/'+fname) as infile:
                    for line in infile:
                        outfile.write(line)
                    outfile.write('\n')