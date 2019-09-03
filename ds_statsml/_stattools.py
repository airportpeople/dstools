import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.preprocessing import Imputer
from statsmodels.stats.outliers_influence import variance_inflation_factor


class StatsmodelSKLearn(BaseEstimator, RegressorMixin):
    '''
    Usage:
    # create a model
    >> clf = StatsmodelSKLearn(sm.OLS)

    # Print cross val score on this model
    >> print('crossval', cross_val_score(clf, sm.add_constant(ccard.data[['AGE', 'INCOME', 'INCOMESQ']]), ccard.data['AVGEXP'], cv=5))

    '''
    def __init__(self, sm_model, regularize=False):
        self.sm_model = sm_model
        self.model = None
        self.result = None
        self.regularize = regularize

    def fit(self, X, y):

        self.model = self.sm_model(y, X)

        if not self.regularize:
            self.result = self.model.fit()
        else:
            self.result = self.model.fit_regularized()

    def predict(self, X):
        return self.result.predict(X)


def inv_boxcox(a, lam):
    return np.exp(lam ** -1 * np.log(a * lam + 1)) - 1


class ReduceVIF(BaseEstimator, TransformerMixin):
    '''
    Use:
    transformer = ReduceVIF()
    # Only use 10 columns for speed in this example
    X_vif = transformer.fit_transform(X)
    X_vif.head()

    CREDIT:
    Original Author: Roberto Ruiz
    Reference: https://www.kaggle.com/robertoruiz/sberbank-russian-housing-market/dealing-with-multicollinearity

    I made only slight changes in the calculate_vif function to print what I wanted (what makes sense), changed some
    variable names, and added a few comments. Also, there was reference to a 'y' sort of target variable, which isn't
    needed.
    LTJDS, 18 July 2018.
    '''

    def __init__(self, thresh=5.0, impute=True, impute_strategy='median'):
        # From looking at documentation, values between 5 and 10 are "okay".
        # Above 10 is too high and so should be removed.
        self.thresh = thresh

        # The statsmodel function will fail with NaN values, as such we have to impute them.
        # By default we impute using the median value.
        # This imputation could be taken out and added as part of an sklearn Pipeline.
        if impute:
            self.imputer = Imputer(strategy=impute_strategy)

    def fit(self, X):
        print('ReduceVIF fit')
        if hasattr(self, 'imputer'):
            self.imputer.fit(X)
        return self

    def transform(self, X):
        print('ReduceVIF transform')
        columns = X.columns.tolist()
        if hasattr(self, 'imputer'):
            X = pd.DataFrame(self.imputer.transform(X), columns=columns)
        return ReduceVIF.calculate_vif(X, self.thresh)

    @staticmethod
    def calculate_vif(X, thresh=5.0):
        '''
        These are my personal edits. We just want this to print the max VIF values as it takes out features.
        :param X: pd.DataFrame, The X DataFrame
        :param thresh: Lowest accepted VIF to go through.
        :return:
        '''

        assert isinstance(X, pd.DataFrame)
        # Taken from https://stats.stackexchange.com/a/253620/53565 and modified
        # Each time we loop through, we drop a variable, and we need to check the next iteration
        checknext = True

        while checknext:
            variables = X.columns
            checknext = False

            # Run the variance_inflation_factor function on all the factors left over, and find the max
            # First parameter is the whole matrix (left), second is the variable in question
            vif = [variance_inflation_factor(X[variables].values, X.columns.get_loc(var)) for var in X.columns]
            max_vif = max(vif)

            # If the maximum vif is larger than the threshold, take that factor out of the dataframe, and try again
            if max_vif > thresh:
                maxloc = vif.index(max_vif)
                print(f'Current maximum VIF is {X.columns[maxloc]} .... VIF={max_vif}')
                print(f'Dropping {X.columns[maxloc]}.')
                X = X.drop([X.columns.tolist()[maxloc]], axis=1)
                checknext = True

        return X


def stepwise_selection(X, y,
                       initial_list=(),
                       threshold_in=0.05,
                       threshold_out=0.05,
                       verbose=True):
    """ Perform a forward-backward feature selection
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features
    Always set threshold_in < threshold_out to avoid infinite looping.
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    """
    included = list(initial_list)

    while True:
        changed = False

        # forward step
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded)

        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]

        best_pval = new_pval.min()

        if best_pval < threshold_in:
            best_feature = new_pval.argmin()
            included.append(best_feature)
            changed = True

            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()

        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()  # null if pvalues is empty

        if worst_pval > threshold_out:
            changed = True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)

            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))

        if not changed:
            break

    return included


def get_outliers(df_all, groupcols, group_abbrev, statcols, outlier_factors=1.5, keep_agg=False, inplace=True):
    '''
    Determine if values in `statcols` are outliers, based on some grouping scheme. I.e., if the row is in <this> group, is it drastically different
    from other items in that group?

    :param df_all:
    :param groupcols: (list) List of columns representing the groupings
    :param group_abbrev: (str) This is the abbreviation you'd like to use in naming columns
    :param statcols: (list) the column names of the values that you'd like to determine outliers
    :param outlier_factors: (str, float) The value to multiply the IQR range to determine outleirs. I.e., we say an outlier is an outlier if

        value > 75th percentile + `outlier_factor` * IQR
                             OR
        value < 25th percentile - `outlier_factor` * IQR

        and we do this for each of the statcols. outlier_factors is either the same length as statcols, or it's a float; the same value for all
        statcols
    :param keep_agg: (bool) Do you want to keep the aggregate values like mean, median, max, min, 75thperc, 25thperc?
    :param inplace: Inplace

    :return:
        df. This is done inplace right now.
    '''

    if inplace:
        df = df_all
    else:
        df = df_all.copy()

    if not hasattr(outlier_factors, 'len'):
        outlier_factors = [outlier_factors] * len(statcols)

    inflated_values = []

    if np.percentile(df[statcols[0]], 25) == np.percentile(df[statcols[0]], 75):
        inflated_value = np.percentile(df[statcols[0]], 25)
        print(f"{statcols[0]} is inflated at the value {inflated_value} (comprises >= 50% of the data).")
        print("Only checking outliers for non-inflated values.")
        inflated_values.append(inflated_value)
    else:
        inflated_values.append(np.nan)

    df_aggdata = df.groupby(groupcols)[statcols[0]] \
        .agg([(group_abbrev + "_mean_" + statcols[0], 'mean'),
              (group_abbrev + "_median_" + statcols[0], 'median'),
              (group_abbrev + "_max_" + statcols[0], 'max'),
              (group_abbrev + "_min_" + statcols[0], 'min'),
              (group_abbrev + "_25perc_" + statcols[0], lambda a: np.percentile(a, 25)),
              (group_abbrev + "_75perc_" + statcols[0], lambda a: np.percentile(a, 75))]) \
        .reset_index()

    for statcol in statcols[1:]:
        if np.percentile(df[statcols], 25) == np.percentile(df[statcols], 75):
            inflated_value = np.percentile(df[statcols], 25)
            print(f"{statcols} is inflated at the value {inflated_value} (comprises >= 50% of the data).")
            print("Only checking outliers for non-inflated values.")
            inflated_values.append(inflated_value)
        else:
            inflated_values.append(np.nan)

        df_aggdata_ = df.groupby(groupcols)[statcol] \
            .agg([(group_abbrev + "_mean_" + statcol, 'mean'),
                  (group_abbrev + "_median_" + statcol, 'median'),
                  (group_abbrev + "_max_" + statcol, 'max'),
                  (group_abbrev + "_min_" + statcol, 'min'),
                  (group_abbrev + "_25perc_" + statcol, lambda a: np.percentile(a, 25)),
                  (group_abbrev + "_75perc_" + statcol, lambda a: np.percentile(a, 75))]) \
            .reset_index()

        df_aggdata = pd.merge(df_aggdata, df_aggdata_, on=groupcols)

    df_outliers = pd.merge(df[statcols + groupcols], df_aggdata, on=groupcols, how='left')

    df_outliers.index = df.index

    for statcol, outlier_factor, inflated_value in zip(statcols, outlier_factors, inflated_values):
        upper = df_outliers[f'{group_abbrev}_75perc_{statcol}']
        lower = df_outliers[f'{group_abbrev}_25perc_{statcol}']
        iqr = upper - lower

        df[f'{statcol}_outlier'] = (df_outliers[statcol] != inflated_value) & \
                                   ((df_outliers[statcol] > upper + outlier_factor * iqr) |
                                    (df_outliers[statcol] < lower - outlier_factor * iqr))

    if keep_agg:
        df = pd.concat((df, df_outliers[[col for col in df_outliers.columns if col not in statcols + groupcols]]), axis=1)

    if not inplace:
        return df


def _N_bma(alpha, beta, l, x):
    '''
    Get sample size for a dataset (x), alpha, beta, and effect size (l). This is one iteration.
    '''
    z_alpha2 = stats.norm.ppf(alpha / 2)
    z_beta = stats.norm.ppf(beta)
    sigma = x.std()

    nh = ((z_alpha2 + z_beta) ** 2 * sigma ** 2) / l ** 2
    return nh


def sample_size_bma(conv_rates, alpha=0.1, beta=0.15, l=0.05, iters=1000):
    '''
    Bootstrap Sample Size calculation as described in "Sample Size Calculations in Clinical Research 2nd ed", by
    S. Chow, et al., (Chapman and Hall, 2008) WW. on page 350 (section n13.3)

    Parameters
    ----------
    conv_rates
    alpha
    beta
    l
    iters

    Returns
    -------

    '''
    nh_all = []
    conv_rates = np.array(conv_rates)

    for h in range(iters):
        x = np.random.choice(conv_rates, conv_rates.shape)
        nh_all.append(_N_bma(alpha, beta, l, x))

    N_bma = np.ceil(np.median(nh_all))  # Bootstrap-Median Approach
    return int(np.ceil(N_bma))