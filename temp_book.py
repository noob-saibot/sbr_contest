import pickle
import pandas as pd
import matplotlib.pylab as plt
import seaborn
seaborn.set_style()
rs = pickle.load(open('feature_statistic', 'rb'))
pd.DataFrame(rs).sort_values(by=1).plot.barh(x=0, y=1, xlim=(0.45, 0.55))
plt.show()