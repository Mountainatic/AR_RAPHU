import numpy as np
import pytest

from ar_raphu.spectral.amplitude_domain import (
    AmplitudeDomain,
    AmplitudeOutOfDomainError,
)
from ar_raphu.spectral.spline_basis import CenteredSplineBasis


def test_train_fit_domain_has_complete_coverage_without_clipping():
    values = np.linspace(-2.0, 3.0, 101)
    domain = AmplitudeDomain.fit(values, padding_fraction=0.10)
    basis = CenteredSplineBasis.fit(
        values, n_basis=16, degree=3, domain=domain
    )
    transformed = basis.transform(values)
    assert np.all(domain.in_domain_mask(values))
    assert np.isfinite(transformed).all()
    assert domain.fit_lower == pytest.approx(-2.5)
    assert domain.fit_upper == pytest.approx(3.5)


def test_strict_transform_rejects_out_of_domain_values():
    values = np.linspace(-1.0, 1.0, 51)
    domain = AmplitudeDomain.fit(values)
    basis = CenteredSplineBasis.fit(values, n_basis=8, domain=domain)
    outside = np.array([domain.fit_lower - 1e-3, 0.0])
    with pytest.raises(AmplitudeOutOfDomainError):
        basis.transform(outside)
    transformed, mask = basis.transform_with_mask(outside)
    assert mask.tolist() == [False, True]
    assert np.isnan(transformed[0]).all()
