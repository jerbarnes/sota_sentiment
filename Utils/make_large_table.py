import numpy as np
import argparse
import sys, os


header = """
\\newcommand{\\rt}[1]{\\rotatebox{90}{#1}}
\\newcommand{\\rrt}[1]{\\rotatebox{45}{#1}}
\\newcommand{\\F}{$\\text{F}_1$\\xspace}
\\newcommand{\\sd}[1]{\\par \\tiny #1}
\\newcommand{\\sep}{\\cmidrule(r){1-2}\\cmidrule(lr){3-3}\\cmidrule(lr){4-4}\\cmidrule(lr){5-5}\\cmidrule(lr){6-6}\\cmidrule(lr){7-7}\\cmidrule(lr){8-8}\\cmidrule(lr){9-9}}
\\begin{table*}[t]
  \\centering
  \\renewcommand*{\\arraystretch}{0.9}
  \\setlength\\tabcolsep{1.8mm}
  \\newcolumntype{P}{>{\\centering\\arraybackslash}p{5mm}}

  \\begin{tabular}{llllllllll}
    \\toprule
    \\rrt{Model} & \\rrt{Dim.}&\\rrt{SST-fine} & \\rrt{SST-binary} & \\rrt{OpeNER} & \\rrt{SenTube-A} & \\rrt{SenTube-T} & \\rrt{SemEval} & \\rrt{Overall} \\\\
    \\sep"""

footer ="""
  \\bottomrule
  \\end{tabular}
  \\caption{Accuracy on the test sets. For all neural models we perform 5 runs and show the
  mean and standard deviation.}
  \\label{results}
\\end{table*}
"""

BOW_pattern = """\\multirow{{1}}{{*}}{{\\rt{{BOW}}}} \\\\ & & {0:.1f} & {1:.1f} & {2:.1f}& {3:.1f}& {4:.1f}& {5:.1f}& {6:.1f} \\\\\\ \\\\
   \\sep"""

Linear_pattern = """\\multirow{{4}}{{*}}{{\\rt{{{0}}}}} & 50 & {1:.1f} & {2:.1f} & {3:.1f}& {4:.1f}& {5:.1f}& {6:.1f}& {7:.1f} \\\\
                            & 100 & {8:.1f} & {9:.1f} & {10:.1f}& {11:.1f}& {12:.1f}& {13:.1f}& {14:.1f} \\\\
                            & 200 & {15:.1f} & {16:.1f} & {17:.1f}& {18:.1f}& {19:.1f}& {20:.1f}& {21:.1f} \\\\
                            & 300 & {22:.1f} & {23:.1f} & {24:.1f}& {25:.1f}& {26:.1f}& {27:.1f}& {28:.1f} \\\\
   \\sep
   """

Neural_pattern = """   \\multirow{{4}}{{*}}{{\\rt{{{0}}}}} & 50 & {1:.1f} \\sd{{({2:.1f})}} & {3:.1f} \\sd{{({4:.1f})}}& {5:.1f} \\sd{{({6:.1f})}}& {7:.1f} \\sd{{({8:.1f})}}& {9:.1f} \\sd{{({10:.1f})}}& {11:.1f} \\sd{{({12:.1f})}}& {13:.1f} \\sd{{({14:.1f})}}  \\\\ 
                            & 100 & {15:.1f} \\sd{{({16:.1f})}} & {17:.1f} \\sd{{({18:.1f})}}& {19:.1f} \\sd{{({20:.1f})}}& {21:.1f} \\sd{{({22:.1f})}}& {23:.1f} \\sd{{({24:.1f})}}& {25:.1f} \\sd{{({26:.1f})}}& {27:.1f} \\sd{{({28:.1f})}}  \\\\
                            & 200 & {29:.1f} \\sd{{({30:.1f})}} & {31:.1f} \\sd{{({32:.1f})}}& {33:.1f} \\sd{{({34:.1f})}}& {35:.1f} \\sd{{({36:.1f})}}& {37:.1f} \\sd{{({38:.1f})}}& {39:.1f} \\sd{{({40:.1f})}}& {41:.1f} \\sd{{({42:.1f})}}  \\\\
                            & 300 & {43:.1f} \\sd{{({44:.1f})}} & {45:.1f} \\sd{{({46:.1f})}}& {47:.1f} \\sd{{({48:.1f})}}& {49:.1f} \\sd{{({50:.1f})}}& {51:.1f} \\sd{{({52:.1f})}}& {53:.1f} \\sd{{({54:.1f})}}& {55:.1f} \\sd{{({56:.1f})}}  \\\\
    \\sep
    """

def main(args):
    parser = argparse.ArgumentParser(description="creates large table from 4 results files")
    parser.add_argument('-results_dir', default='results')
    parser.add_argument('-output', default='figs/large_table.txt')

    args = vars(parser.parse_args())
    results_dir = args['results_dir']
    output_file = args['output']

    results_files = ['50-results.txt', '100-results.txt', '200-results.txt', '300-results.txt']


    #Keep all of the results for each model across
    model_results = [[],[],[],[],[],[],[]]
    names = ['BOW', 'Average', 'Retrofit', 'Joint', 'LSTM', 'BiLSTM', 'CNN']
    name_idx = 0


    # get the results and keep them in model_results, as np.arrays
    for file in results_files:
        f = open(os.path.join(results_dir, file)).readlines()
        for i, line in enumerate(f):
            # each model's results start with +++
            if line.startswith('+++'):
                # set the name to the next in the list 'names'
                name = names[name_idx]
                # get the results we are interested in
                results = f[i+3:i+10] 
                # the linear methods don't have std deviations and are shorter
                if len(results[-1].split())  == 5:
                    data = np.array([l.split()[1:] for l in results], dtype=float) * 100
                    accs = data[:,0].reshape(7) # first column == accuracy
                    model_results[name_idx].append(accs)
                # the neural models have std dev, so they are longer
                else:
                    results = [l.replace('±','') for l in results]
                    data = np.array([l.split()[1:] for l in results], dtype=float) * 100
                    accs = data[:,:2].reshape(14)
                    model_results[name_idx].append(accs)
                name_idx += 1
        name_idx = 0


    # Create the final table
    table = ''
    table += header

    for i, model in enumerate(model_results):
        # BOW model 
        if i == 0:
            vec = model_results[i][0]
            table += BOW_pattern.format(*vec)
        elif i < 4:
            vec = np.array(model_results[i]).reshape(28)
            name = [names[i]]
            data = name + list(vec)
            table += Linear_pattern.format(*data)
        else:
            vec = np.array(model_results[i]).reshape(56)
            name = [names[i]]
            data = name + list(vec)
            table += Neural_pattern.format(*data)

    table += footer

    with open(output_file, 'w') as out:
        out.write(table)
    

if __name__ == "__main__":

    args = sys.argv
    main(args)