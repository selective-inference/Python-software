import numpy as np, pandas as pd

from ..drop_losers import drop_losers
from ..screening import topK
from ..randomization import randomization

def test_drop_losers(p=50,
                     K=5,
                     n=300,
                     use_MLE=True):

    arm = []
    data = []
    stage = []
    for a in range(p):
        N = int(np.random.poisson(n, size=(1,)))
        arm.extend([a]*N)
        stage.extend([1]*N)
        data.extend(list(np.random.standard_normal(N)))

    df = pd.DataFrame({'arm':arm,
                       'stage':stage,
                       'data':data})

    grouped = df.groupby('arm')
    stage1_means = df.groupby('arm').mean().sort_values('data', ascending=False)
    winners = list(stage1_means.index[:K])

    for winner in winners:
        N = int(np.random.poisson(30, size=(1,)))
        arm.extend([winner]*N)
        stage.extend([2]*N)
        data.extend(list(np.random.standard_normal(N)))

    df = pd.DataFrame({'arm':arm,
                       'stage':stage,
                       'data':data})

    dtl = drop_losers(df,
                      K=K)

    dtl.MLE_inference()
    if not use_MLE:
        result = dtl.summary(ndraw=20000, burnin=5000)
    else:
        result = dtl.MLE_inference()[0]
    pvalue = np.asarray(result['pvalue'])
    lower = np.asarray(result['lower_confidence'])
    upper = np.asarray(result['upper_confidence'])
    cover = (lower < 0) * (upper > 0)

    return pvalue, cover

def test_compare_topK(p=20,
                      K=5,
                      n=100):

    arm = []
    data = []
    stage = []
    for a in range(p):
        N = int(np.random.poisson(n, size=(1,)))
        arm.extend([a]*N)
        stage.extend([1]*N)
        data.extend(list(np.random.standard_normal(N)))

    df1 = pd.DataFrame({'arm':arm,
                       'stage':stage,
                       'data':data})

    grouped = df1.groupby('arm')
    stage1_means = df1.groupby('arm').mean().sort_values('data', ascending=False)
    winners = list(stage1_means.index[:K])

    for winner in winners:
        N = int(np.random.poisson(30, size=(1,))) + 5
        arm.extend([winner]*N)
        stage.extend([2]*N)
        data.extend(list(np.random.standard_normal(N)))
        
    df2 = pd.DataFrame({'arm':arm,
                        'stage':stage,
                        'data':data})

    dtl = drop_losers(df2,
                      K=K)

    # need additional data for randomized api with non-degenerate covariance

    for a in range(p):
        if a not in winners:
            N = int(np.random.poisson(30, size=(1,))) + 5
            arm.extend([a]*N)
            stage.extend([2]*N)
            data.extend(list(np.random.standard_normal(N)))

    df_full = pd.DataFrame({'arm':arm,
                            'stage':stage,
                            'data':data})
    full_means = df_full.groupby('arm').mean()['data'].iloc[range(p)]
    full_std = df_full.groupby('arm').std()['data'].iloc[range(p)]
    n_1 = df1.groupby('arm').count()['data'].iloc[range(p)]
    n_full = df_full.groupby('arm').count()['data'].iloc[range(p)]
    print(n_1, n_full)
    stage1_means = df1.groupby('arm').mean()['data'].iloc[range(p)]
    perturb = np.array(stage1_means) - np.array(full_means)

    covariance = np.diag(np.array(full_std)**2 / np.array(n_full))
    randomizer = randomization.gaussian(np.diag(np.array(full_std)**2 / np.array(n_1)) - 
                                        covariance)

    randomized_topK = topK(full_means,
                           covariance,
                           randomizer,
                           K,
                           perturb=perturb)

    randomized_topK.fit(perturb=perturb)

    (observed_target,
     target_cov,
     target_score_cov,
     _) = randomized_topK.marginal_targets(randomized_topK.selection_variable['variables'])

    # try with a degenerate covariance now

    means2 = df2.groupby('arm').mean()['data'].iloc[range(p)]
    std2 = df2.groupby('arm').std()['data'].iloc[range(p)]
    n_2 = df2.groupby('arm').count()['data'].iloc[range(p)]
    stage1_means = df1.groupby('arm').mean()['data'].iloc[range(p)]
    perturb2 = np.array(stage1_means) - np.array(means2)
    covariance2 = np.diag(np.array(std2)**2 / np.array(n_2))
    degenerate_randomizer = randomization.degenerate_gaussian(
                               np.diag(np.array(std2)**2 / 
                                       np.array(n_1)) - 
                               covariance2)

    degenerate_topK = topK(means2,
                           covariance2,
                           degenerate_randomizer,
                           K,
                           perturb=perturb2)

    np.random.seed(0)
    summary1 = randomized_topK.summary(observed_target,
                                       target_cov,
                                       target_score_cov,
                                       alternatives=['twosided']*K,
                                       ndraw=10000,
                                       burnin=2000,
                                       compute_intervals=True)
    np.random.seed(0)
    summary2 = dtl.summary(ndraw=10000,
                           burnin=2000)

    np.testing.assert_allclose(summary1['pvalue'], summary2['pvalue'], rtol=1.e-3)
    np.testing.assert_allclose(summary1['target'], summary2['target'], rtol=1.e-3)
    np.testing.assert_allclose(summary1['lower_confidence'], summary2['lower_confidence'], rtol=1.e-3)
    np.testing.assert_allclose(summary1['upper_confidence'], summary2['upper_confidence'], rtol=1.e-3)

    np.random.seed(0)
    degenerate_topK.fit(perturb=perturb2)
    summary3 = degenerate_topK.summary(observed_target,
                                       target_cov,
                                       target_score_cov,
                                       alternatives=['twosided']*K,
                                       ndraw=10000,
                                       burnin=2000,
                                       compute_intervals=True)
    
    np.testing.assert_allclose(summary1['pvalue'], summary3['pvalue'], rtol=1.e-3)
    np.testing.assert_allclose(summary1['target'], summary3['target'], rtol=1.e-3)
    np.testing.assert_allclose(summary1['lower_confidence'], summary3['lower_confidence'], rtol=1.e-3)
    np.testing.assert_allclose(summary1['upper_confidence'], summary3['upper_confidence'], rtol=1.e-3)


def main(nsim=100, use_MLE=True):

    P0, cover = [], []
    
    for i in range(nsim):
        p0, cover_ = test_drop_losers(use_MLE=use_MLE)

        cover.extend(cover_)
        P0.extend(p0)
        print('coverage', np.mean(cover))
