import numpy as np
import argparse
import sys

"""
This script automatically creates the latex tables for
our paper on comparing sentiment models.
"""

head = """
\\begin{table*}[!htbp]
\centering
 \\begin{tabular}{llllllll} 
 \hline
		& 	RT		& SST 	 & OpeNER & Sentube-a & Sentube-t & Semeval& Overall \\\\ 
 \hline\hline

"""

pattern1 = '{0} & {1:.1f} & {2:.1f} & {3:.1f}& {4:.1f}& {5:.1f}& {6:.1f}& {7:.1f} \\\\ \n'
pattern2 = '{0} & {1:.1f}({2:.1f}) & {3:.1f}({4:.1f}) & {5:.1f}({6:.1f}) & {7:.1f}({8:.1f}) & {9:.1f}({10:.1f}) & {11:.1f}({12:.1f}) & {13:.1f}({14:.1f}) \\\\ \n' 

tail = """

 \hline
 \end{tabular}
 \caption{Accuracy of different approaches on 6 datasets. For all neural networks we report the average and standard deviation (in parenthesis) over 5 runs.}
 \label{}
\end{table*}
"""

def main(args):
    parser = argparse.ArgumentParser(description='creates a latex table from the raw data put out by run.sh')
    parser.add_argument('-results', help='results file (txt) which you would like to turn into a latex table',
                        default='/home/jeremy/Escritorio/comparison_of_sentiment_methods/results/results.txt')
    parser.add_argument('-outfile', help='file where you would like to save the latex code',
                        default='/home/jeremy/Escritorio/comparison_of_sentiment_methods/results/table.txt')

    args = vars(parser.parse_args())
    results_file = args['results']
    outfile = args['outfile']

    f = open(results_file).readlines()

    table = ''

    table += head

    names = ['BOW', 'Wiki', 'Retrofit', 'Amazon', 'Joint', 'LSTM', 'BiLSTM', 'CNN']
    name_idx = 0

    for i, line in enumerate(f):
        if line.startswith('+++'): # Each experiment in file starts with +++
            name = names[name_idx]
            name_idx += 1
            # Get relevant lines (6 datasets)
            results = f[i+3:i+10]
            if len(results[-1].split()) == 5:
                data = np.array([l.split()[1:] for l in results], dtype=float) * 100
                accs = data[:,0].reshape(7)
                out = pattern1.format(*[name] + list(accs))
            else:
                results = [l.replace('±','') for l in results]
                data = np.array([l.split()[1:] for l in results], dtype=float) * 100
                accs = data[:,:2].reshape(14)
                out = pattern2.format(*[name] + list(accs))
            table += out
            
    table += tail
    with open(outfile, 'w') as out:
        out.write(table)


if __name__ == '__main__':

    args = sys.argv
    main(args)
