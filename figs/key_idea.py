import math, os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')
from IsotonicLogisticRegression import IsotonicLogisticRegression

import matplotlib
import matplotlib.backends.backend_pdf
import scipy

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

np.random.seed(2)

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

ilr = IsotonicLogisticRegression()
# x0 = np.exp(scipy.stats.norm.rvs(size = 10000))

def mylog(x): return -np.log(x) / np.log(10)

x0 = mylog(scipy.stats.uniform.rvs(size=10**5*1)**1 * 1.0)

x1a = mylog(((scipy.stats.uniform.rvs(size=math.ceil(10**3.0*1)) + 1e-2)**1 * (0.1**1)   )**2.0)
x1b = mylog(((scipy.stats.uniform.rvs(size=math.ceil(10**2.0*1)) + 1e-3)**1 * (0.1**0.5) )**2.0)
x1c = mylog(((scipy.stats.uniform.rvs(size=math.ceil(10**1.0*1)) + 1e-4)**1 * (0.1**0)   )**2.0)
x1d = np.array(()) # scipy.stats.uniform.rvs(size=500)**1 * 1.0

x1 = x = np.concatenate([x1a, x1b, x1c, x1d])
x = np.concatenate([x0, x1])
y = np.array([0] * len(x0) + [1] * (len(x1a) +len(x1b) + len(x1c) + len(x1d)))

ilr.fit(np.array([[x_] for x_ in x]), y)
raw_log_odds = np.array(ilr.get_density_estimated_log_odds()).flatten()
iso_X = np.array(ilr.get_isotonic_X()).flatten()
iso_log_odds = np.array(ilr.get_isotonic_log_odds()).flatten()
cir_X = np.array(ilr.get_centered_isotonic_X()).flatten()
cir_log_odds = np.array(ilr.get_centered_isotonic_log_odds()).flatten()

offset = np.log(len(x1)/ len(x0))

#print(raw_log_odds)
#print(iso_log_odds)
#print(cir_log_odds)

fig, axes = plt.subplots(2, 1, height_ratios=[1, 1.5], layout='constrained')
fig.set_figheight(3)
fig.set_figwidth(8)
xrange1 = (
        min((min(x0), min(x1))),
        max((max(x0), max(x1))))
xrange2 = (int(math.floor(xrange1[0])), int(math.ceil(xrange1[1])))
axes[0].hist([x1, x0], label=['Tested positive ($A_f$)', 'Tested negative ($B_f$)'], color = [(0.75, 0.00, 0.00), (0.25, 0.25, 0.25)], bins=xrange2[1]*4, range=xrange2, log=True)
# .scatter(x, y, label='1. Data-points', alpha=1.0/16)
axes[0].legend()
axes[0].set_ylabel('Frequency')
axes[1].set_xlim(xrange2[0] - 0.4, xrange2[1] + 0.4)
axes[1].plot   (iso_X, raw_log_odds + offset, label='After (Step 1: estimation of $\\ln(A_f \\div B_f)$)', alpha = 0.25, marker = 'o', linewidth=0.5, markersize=6) # (, linestyle='dotted')
axes[1].plot   (iso_X, iso_log_odds + offset, label='After (Step 2: isotonic regression (IR))',            alpha = 0.50, marker = '<', linewidth=0.5, markersize=5) # (, linestyle='dashed')
axes[1].plot   (cir_X, cir_log_odds + offset, label='After (Step 3: centered IR (CIR))',                   alpha = 0.75, marker = '>', linewidth=0.5, markersize=4)
axes[1].legend()
axes[1].set_ylabel('Log odds')

fig.supxlabel('Raw feature $f$') # ('(e.g., peptide-MHC binding stability)')

plt.savefig("key_idea.pdf", format="pdf", bbox_inches="tight")

