'''
Make this a Python package for iPython.

Notes:

# Time Series:
scipy.signal.find_peaks

# ANOVA
anova = smf.ols('y ~ x_cat', data=df)
anova_fit = anova.fit()
table = sm.stats.anova_lm(anova_fit, typ=3)
table

# ANOVA for Model Reduction
anovaResults = sm.stats.anova_lm(lm2_fit, lm1_fit)
anovaResults

# QQ-plot
sm.qqplot(lm.resid);

# Contingency Table Tests
data_ = df[['exog', 'endog']]   # Order is important here
data_crosstab = pd.crosstab(data_['exog'], data_['endog'])
sqtable = sm.stats.SquareTable(data_crosstab.values)
table = sm.stats.Table.from_data(data_)

# Testing Ordinal/Nominal Dependence
results = sqtable.test_ordinal_association()
results = sqtable.test_nominal_association()   # This one is a Chi-Squared Test
results.pvalue

# Rotate axis labels
for tick in axis.get_xticklabels():
    tick.set_rotation(90)

# Grid Search
params = {
    "n_neighbors": list(range(1, 21)),
    "weights": ['uniform', 'distance']
}

grid_est = GridSearchCV(KNeighborsClassifier(), param_grid=params, cv=ksplits,
                        return_train_score=True, scoring='accuracy')
grid_est.fit(X_train, y_train)
grid_est.score(X_test, y_test)
grid_df = pd.DataFrame(grid_est.cv_results_)
grid_df['alpha'] = grid_df.params.apply(lambda val: val['alpha'])
plt.plot(np.log10(grid_df.alpha), grid_df.mean_test_score);
grid_df[['alpha', 'mean_test_score']].sort_values('mean_test_score', ascending=False)
'''
