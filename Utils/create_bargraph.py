import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse


def main(args):
    """
    Plots the average accuracy and the max and min accuracy for each method
    over different benchmark datasets.
    """
    parser = argparse.ArgumentParser(description='Plots the average accuracy and max and min accuracy for each method over different benchmark datasets. The input is the raw data put out by run.sh. and the output is a png file in the "figures" directory')
    parser.add_argument('-results', help='results file (txt) which you would like to turn into a latex table',
                        default='results/results.txt')
    parser.add_argument('-outfile', help='file where you would like to save the latex code',
                        default='figures/acc_on_datasets.png')

    args = vars(parser.parse_args())
    results_file = args['results']
    outfile = args['outfile']

    f = open(results_file).readlines()
    mean_accs = []
    std_accs = [[],[]]

    for i, line in enumerate(f):
        if line.startswith('+++'): # Each experiment in file starts with +++

            # Get relevant lines (10 datasets)
            res = f[i+3:i+10]
            if len(res[-1].split()) == 5:
                ave_acc, ave_prec, ave_rec, ave_f1 = np.array(res[-1].split()[1:], dtype='float32')
                # create a numpy array of the results for each model
                # r[:,0] = accuracy r[:,1] = precision
                # r[:,2] = recall   r[:,3] = f1
                r = np.array([np.array(l.split()[1:], dtype='float32') for l in res[:-1]])
            else:
                ave_acc, ave_prec, ave_rec, ave_f1 = np.array(res[-1].split()[1::2], dtype='float32')

                # create a numpy array of the results for each model
                # r[:,0] = accuracy r[:,1] = precision
                # r[:,2] = recall   r[:,3] = f1
                r = np.array([np.array(l.split()[1::2], dtype='float32') for l in res[:-1]])

            # get maximum and minimum error for each model
            # low error = |mean accuracy - lowest accuracy|
            # high error = |mean accuracy - highest accuracy|
            std_accs[0].append(abs(ave_acc - r[:, 0].min()))
            std_accs[1].append(abs(ave_acc - r[:, 0].max()))
            mean_accs.append(ave_acc)

    # Plot the bar chart
    ind = np.arange(len(mean_accs))
    width = .6
    colors = ['red', 'blue', 'green']
    plt.figure()
    plt.title('Accuracy on datasets')
    plt.bar(ind, mean_accs, width, color=colors, yerr=std_accs, ecolor='black')
    plt.xticks(ind,('BOW','BOV-wiki', 'BOV-retro',
                    'BOV-amazon', 'Joint', 'CNN', 'LSTM', 'BiLSTM'))
    plt.savefig(outfile)

if __name__ == '__main__':

    args = sys.argv
    main(args)
