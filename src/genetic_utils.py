import numpy as np

def epistasis_scale(var_main, var_epi, epi_fraction):

    assert 0 < epi_fraction < 1

    scale = np.sqrt(
        (epi_fraction / (1 - epi_fraction))
        * (var_main / var_epi)
    )

    return scale


def compute_epistasis_effect(
        genotypes,
        pairs,
        gamma,
        PRS,
        epi_variance_fraction,
        apply_scaling=True):

    N = genotypes.shape[0]

    epi_effect = np.zeros(N)

    for idx, (i, j) in enumerate(pairs):

        if gamma[idx] != 0:

            xi = genotypes[:, i] - np.mean(genotypes[:, i])

            xj = genotypes[:, j] - np.mean(genotypes[:, j])

            epi_term = xi * xj

            epi_term = (
                epi_term
                - np.cov(xi, xj)[0, 1] / np.var(xi) * xi
            )

            epi_term = (
                epi_term
                - np.cov(xi, xj)[0, 1] / np.var(xj) * xj
            )

            epi_effect += gamma[idx] * epi_term

    var_main = np.var(PRS)

    var_epi = np.var(epi_effect)

    if apply_scaling and var_epi > 0:

        scale = epistasis_scale(
            var_main,
            var_epi,
            epi_variance_fraction
        )

        epi_effect *= scale
        gamma *= scale

    return epi_effect, gamma