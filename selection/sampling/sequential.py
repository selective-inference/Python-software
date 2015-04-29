"""
Sequential Monte Carlo for approximately constrained Gaussians.

http://arxiv.org/abs/1410.8209

"""

import numpy as np

def sample(white_constraint,
           nsample,
           proposal_sigma=0.2,
           temps=np.linspace(0, 50, 51.)):
    """
    Build up an approximately constrained Gaussian
    based on relaxations of the constraint.

    Parameters
    ----------

    white_constraint : `selection.constraints.affine`
        Affine constraint with identity covariance

    nsample : int
        How many samples to draw?

    proposal_sigma : float
        
    """

    n = white_constraint.dim
    sample_z = np.random.standard_normal((n, nsample))

    def constraint_function(z, con):
        value = (np.dot(con.linear_part, z) - con.offset[:,None])
        return value.max(0)

    def constraint_logit(temp, z, con):
        tmp_z = constraint_function(z, con)
        tmp_v = np.exp(-temp * tmp_z)
        return tmp_v / (1 + tmp_v)

    def MH_sample(temp, z_cur, con):
        step = np.random.standard_normal(z_cur.shape) * proposal_sigma
        z_new = z_cur + step

        W_new = constraint_logit(temp, z_new, con)
        W_cur = constraint_logit(temp, z_cur, con)
        W_new *= np.exp(-(z_new**2).sum(0)/2)
        W_cur *= np.exp(-(z_cur**2).sum(0)/2)

        coin_flip = np.less_equal(np.random.sample(z_cur.shape[1]), W_new / W_cur)
        final_sample = coin_flip * z_new + (1 - coin_flip) * z_cur
        return final_sample

    weights = np.ones(nsample, np.float) / nsample

    num = np.ones(nsample) / 2
    for i in range(temps.shape[0]-1):

        num, den = constraint_logit(temps[i+1], sample_z, white_constraint), num

        weights *= np.exp(np.log(num) - np.log(den))
        weights /= weights.sum()

        ESS = 1. / (weights**2).sum()
        if ESS < nsample / 2.:
            idx_z = np.random.choice(np.arange(nsample), size=(nsample,), replace=True, p=weights)
            sample_z = sample_z[:, idx_z]
            weights = np.ones(nsample, np.float) / nsample
        sample_z = MH_sample(temps[i+1], sample_z, white_constraint)

    return sample_z

          
