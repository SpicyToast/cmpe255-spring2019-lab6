import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

age = [17.5,22,29.5,44.5,64.5,80]
deaths = [38,36,24,20,18,28]

%matplotlib inline
np.random.seed(42)

plt.plot(age, deaths, "ko")
plt.xlabel("Age (Years)")
plt.ylabel("Number of Deaths (in 100,000s)")
plt.axis([0, 100, 0, 40])


age_conc = np.c_[np.ones((6, 1)), age]  
theta_best = np.linalg.inv(age_conc.T.dot(age_conc)).dot(age_conc.T).dot(deaths)

                                          
age_b = np.array([[0], [100]])
age_b_conc = np.c_[np.ones((2, 1)), age_b]
death_pred = age_b_conc.dot(theta_best)

plt.plot(age, deaths, "ko")
plt.plot(age_b, death_pred, "g-")
plt.xlabel("Age (Years)")
plt.ylabel("Number of Deaths (in 100,000s)")
plt.axis([0, 100, 0, 40])

from scipy.stats.stats import pearsonr
pearsonr(age, deaths)