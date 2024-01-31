import argparse, json ,math, os, pprint, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')
from IsotonicLogisticRegression import IsotonicLogisticRegression

import matplotlib
import matplotlib.backends.backend_pdf
import scipy

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

parser = argparse.ArgumentParser('Plot an example data transformation made by the isotonic feature normalizer to show the key idea behind this normalizer. ')
parser.add_argument('-o', '--output', help=F'Output PDF file', default='/tmp/key-idea-for-iso-feature-normalizer.pdf')
parser.add_argument('-s', '--seed',   help=F'NumPy random-number generator seed', type=int, default=0)
parser.add_argument('--libname',      help=F'Libname', default='IsotonicLogisticRegression')

args = parser.parse_args()

modulename = __import__(args.libname)
ilr = modulename.IsotonicLogisticRegression()

np.random.seed(args.seed)

plt.rcParams.update({
#    "text.usetex": True, # requires sudo apt-get install texlive-full
    #"font.family": "Helvetica"
})

params = {#'legend.fontsize': 18,
          #'axes.labelsize': 18,
          #'axes.titlesize': 18,
          #'xtick.labelsize' :12,
          #'ytick.labelsize': 12,
          #'grid.color': 'k',
          #'grid.linestyle': ':',
          #'grid.linewidth': 0.5,
          'mathtext.fontset' : 'cm',
          'mathtext.rm'      : 'serif',
          #'font.family'      : 'serif',
          #'font.serif'       : 'Times' # "Times New Roman", # or "Times"          
         }
matplotlib.rcParams.update(params)


#ilr = IsotonicLogisticRegression()
# x0 = np.exp(scipy.stats.norm.rvs(size = 10000))

def mylog(x): return x # return -np.log(x) / np.log(10)

N_NEGATIVES = 10**5 # 10**5*1
MAX_VAL = 9

#x0 = mylog(scipy.stats.uniform.rvs(size=N_NEGATIVES)**1)

x0a = mylog(  scipy.stats.uniform.rvs(size=math.ceil(N_NEGATIVES/10**( 0.0) /  1)) / 3 + 0/3)**2 * MAX_VAL
x0b = mylog(  scipy.stats.uniform.rvs(size=math.ceil(N_NEGATIVES/10**( 1.0) /  2)) / 3 + 1/3)**2 * MAX_VAL
x0c = mylog(  scipy.stats.uniform.rvs(size=math.ceil(N_NEGATIVES/10**( 2.0) /  4)) / 3 + 2/3)**2 * MAX_VAL
x0d = mylog(1-scipy.stats.uniform.rvs(size=math.ceil(N_NEGATIVES/10**( 2.0) /  8))**1.00    )**3 * MAX_VAL

x0e = np.array([MAX_VAL+0.05]*6)
x0f = np.array([MAX_VAL+0.35]*8)
x0g = np.array([MAX_VAL+0.45]*3)
x0h = np.array([MAX_VAL+0.75]*4)
x0i = np.array([MAX_VAL+0.99]*1)
x0 = np.concatenate([x0a, x0b, x0c, x0e, x0f, x0g, x0h, x0i])

x1a = mylog(  scipy.stats.uniform.rvs(size=math.ceil(N_NEGATIVES/10**( 2.0) /  1)) / 3 + 0/3)**2 * MAX_VAL
x1b = mylog(  scipy.stats.uniform.rvs(size=math.ceil(N_NEGATIVES/10**( 2.0) /  2)) / 3 + 1/3)**2 * MAX_VAL
x1c = mylog(  scipy.stats.uniform.rvs(size=math.ceil(N_NEGATIVES/10**( 2.0) /  4)) / 3 + 2/3)**2 * MAX_VAL
x1d = mylog(1-scipy.stats.uniform.rvs(size=math.ceil(N_NEGATIVES/10**( 2.0) /  8))**1.00    )**(2/3) * MAX_VAL
#x1e = mylog(scipy.stats.uniform.rvs(size=math.ceil(N_NEGATIVES/10**( 2.0) /  8))**4 /20 +  19/20)**1 * MAX_VAL
#x1f = mylog(scipy.stats.uniform.rvs(size=math.ceil(N_NEGATIVES/10**( 2.0) / 16))**4 /20 +  18/20)**1 * MAX_VAL
#x1af = np.concatenate([x1a, x1b, x1c, x1d, x1e, x1f])
#x1g = np.array([(max((max(x0), max(x1af))) + 1) / 2] * 10) * 9 # mylog(scipy.stats.uniform.rvs(size=math.ceil(N_NEGATIVES/10**( 2.0) /  4)) /16 + 14/16)**0.5 * 9

x1e = np.array([MAX_VAL+0.20]*40)
x1f = np.array([MAX_VAL+0.40]* 5)
x1g = np.array([MAX_VAL+0.60]*40)
x1h = np.array([MAX_VAL+0.80]* 5)
x1i = np.array([MAX_VAL+1.00]*35)

x1 = np.concatenate([x1a, x1b, x1c, x1d, x1e, x1f, x1g, x1h, x1i])
x = np.concatenate([x0, x1])
y = np.array([0] * len(x0) + [1] * len(x1))

ilr.fit(np.array([[x_] for x_ in x]), y)
raw_log_odds = np.array(ilr.get_density_estimated_log_odds()).flatten()
iso_X = np.array(ilr.get_isotonic_X()).flatten()
iso_log_odds = np.array(ilr.get_isotonic_log_odds()).flatten()
cir_X = np.array(ilr.get_centered_isotonic_X()).flatten()
cir_log_odds = np.array(ilr.get_centered_isotonic_log_odds()).flatten()

offset = 0 # np.log(len(x1)/ len(x0))

#print(raw_log_odds)
#print(iso_log_odds)
#print(cir_log_odds)

fig, axes = plt.subplots(2, 1, height_ratios=[1, 2], layout='constrained')
fig.set_figheight(3.000*1.1*1.05)
fig.set_figwidth(3.000*3.2)
xrange1 = (
        min((min(x0), min(x1))),
        max((max(x0), max(x1))))
xrange2 = (int(math.floor(xrange1[0])), int(math.ceil(xrange1[1])))
bin_per_unit = 2
axes[0].hist([x1, x0], range=[0,MAX_VAL+1], # range=((0,9),(0,9)), 
    label=['Tested positive ($A_f$)', 'Tested negative ($B_f$)'], color = [(0.75, 0.00, 0.00), (0.25, 0.25, 0.25)], 
    bins=(MAX_VAL+1)*bin_per_unit,
    log=True)
# .scatter(x, y, label='1. Data-points', alpha=1.0/16)
axes[0].grid(linestyle='dotted')
axes[0].legend(ncols = 2)
axes[0].set_ylabel('Frequency')
#axes[1].set_xlim(xrange2[0] - 0*1.090/bin_per_unit, xrange2[1] + 0*1.090/bin_per_unit)
axes[1].set_xlim(0-(0.40-0.005)/(9-1)*(MAX_VAL+0), MAX_VAL + 1 + (0.40-0.005)/(9-1)*(MAX_VAL+0))
#axes[1].plot   (iso_X, raw_log_odds + offset, label='After (Step 1: adaptive kernel density estimation\n          of $\\ln(A_f \\div B_f)$)', alpha = 0.200, marker = '^', linewidth=0.5, markersize=(16*3)**0.5) # (, linestyle='dotted')
axes[1].plot   (iso_X, raw_log_odds + offset, label='After (Step 1: adaptive kernel density estimation of $\\ln(A_f \\div B_f)$)', alpha = 0.200, marker = '^', linewidth=0.5, markersize=(16*3)**0.5) # (, linestyle='dotted')

axes[1].plot   (iso_X, iso_log_odds + offset, label='After (Step 2: isotonic regression (IR))',            alpha = 0.300, marker = '<', linewidth=0.5, markersize=(16*2)**0.5) # (, linestyle='dashed')
axes[1].plot   (cir_X, cir_log_odds + offset, label='After (Step 3: centered IR (CIR))',                   alpha = 0.600, marker = '>', linewidth=0.5, markersize=(16*1)**0.5)
axes[1].legend()
axes[1].set_ylabel('Log odds')
axes[1].grid(linestyle='dotted')
fig.supxlabel('Raw feature $f$') # ('(e.g., peptide-MHC binding stability)')

plt.savefig(args.output, format="pdf", bbox_inches="tight")
with open(args.output + '.txt', 'w') as txtout:
    # pp = pprint.PrettyPrinter(indent=2, stream=(txtout))
    json.dump({
        'KDE_log_odds': list(raw_log_odds),
        'ISO_log_odds': list(iso_log_odds),
        'CIR_log_odds': list(cir_log_odds),
    }, txtout, indent=2)

