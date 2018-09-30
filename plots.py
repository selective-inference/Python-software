import statsmodels.api as sm
import matplotlib.pyplot as plt
import pickle
import numpy as np

U = np.linspace(0, 1, 101)
file_labels = ['ss.pickle', 'ss_logit.pickle']

for label in file_labels:
    print(label)
    coverage, P, L, naive_coverage, naive_P, naive_L = pickle.load( open(label, "rb" ) )
    print("selective:", np.mean(P), np.std(P), np.mean(L), np.mean(coverage))
    print("naive:", np.mean(naive_P), np.std(naive_P), np.mean(naive_L), np.mean(naive_coverage))
    print("len ratio selective divided by naive:", np.mean(np.array(L) / np.array(naive_L)))


_, probit_P, _, _, naive_P, _ = pickle.load( open(file_labels[0], "rb" ) )
_, logit_P, _, _, _, _ = pickle.load( open(file_labels[1], "rb" ) )

plt.clf()
plt.plot(U, sm.distributions.ECDF(probit_P)(U), 'c', linewidth=2, label = "fit probit")
plt.plot(U, sm.distributions.ECDF(logit_P)(U), 'b', linewidth=2, label="fit logit")
plt.plot(U, sm.distributions.ECDF(naive_P)(U), 'y', linewidth=2, label="naive")
plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
plt.xlabel("Observed pivot", fontsize=18)
plt.ylabel("Proportion (empirical CDF)", fontsize=18)
plt.title("Pivots", fontsize=18)
plt.legend(fontsize=18, loc="lower right")
plt.savefig('ss_pivots.pdf')