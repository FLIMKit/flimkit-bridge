import numpy as np
import pytest

from flimkit_bridge import irf

needs_machine_irf = pytest.mark.skipif(
    not irf.available(),
    reason='this FLIMKit install ships no machine IRF; they are gitignored, so '
           'an install from source has none')


def _decay(n_bins=256):
    t = np.arange(n_bins)
    return np.exp(-(t - 30) ** 2 / 50.0) * 1000 + 1.0


@needs_machine_irf
def test_machine_irfs_are_discovered_when_present():
    found = irf.available()

    assert found
    assert all(entry['path'].endswith('.npy') for entry in found)
    assert irf.default_path() is not None


@needs_machine_irf
def test_a_machine_irf_can_be_built():
    prompt, description = irf.build('machine_irf', 256, 5e-11, _decay())

    assert prompt.shape == (256,)
    assert np.isfinite(prompt).all()
    assert prompt.sum() > 0
    assert description


def test_a_gaussian_irf_can_be_built():
    prompt, description = irf.build('gaussian', 256, 5e-11, _decay(), fwhm_ns=0.3)

    assert prompt.shape == (256,)
    assert '0.3' in description


def test_the_session_strategy_needs_a_session_irf():
    with pytest.raises(ValueError, match='no instrument response'):
        irf.build('session', 256, 5e-11, _decay(), session_irf=None)


def test_the_session_strategy_uses_what_it_is_given():
    supplied = np.ones(256)

    prompt, description = irf.build('session', 256, 5e-11, _decay(),
                                    session_irf=supplied)

    np.testing.assert_array_equal(prompt, supplied)
    assert description == 'flimkit session'


def test_an_unknown_strategy_is_refused():
    with pytest.raises(ValueError, match='unknown IRF strategy'):
        irf.build('telepathy', 256, 5e-11, _decay())


def test_a_missing_scatter_file_is_refused():
    with pytest.raises(ValueError, match='needs an existing file'):
        irf.build('scatter_file', 256, 5e-11, _decay(), path='/no/such/file.ptu')


@needs_machine_irf
def test_the_default_is_flimkits_configured_one_not_alphabetical():
    """FLIMKit ships several machine IRFs and configures which is the default,
    including a user override. Picking the alphabetically first one would give
    a site-specific IRF instead."""
    from flimkit.configs import MACHINE_IRF_DEFAULT_PATH
    import os

    chosen = irf.default_path()

    assert os.path.realpath(chosen) == os.path.realpath(str(MACHINE_IRF_DEFAULT_PATH))
    assert 'default' in os.path.basename(chosen)


@needs_machine_irf
def test_exactly_one_machine_irf_is_flagged_as_default():
    flagged = [entry for entry in irf.available() if entry['default']]

    assert len(flagged) == 1


@needs_machine_irf
def test_site_specific_irfs_are_still_offered():
    ids = {entry['id'] for entry in irf.available()}

    assert len(ids) > 1, 'only one machine IRF was offered'


def test_a_missing_machine_irf_gives_a_clear_error(monkeypatch):
    """A machine IRF is specific to one microscope, so none ship with FLIMKit
    and a fresh install has none. The failure has to say how to make one."""
    monkeypatch.setattr(irf, 'available', lambda: [])
    monkeypatch.setattr(irf, 'default_path', lambda: None)

    with pytest.raises(ValueError, match='no machine IRF is installed') as caught:
        irf.build('machine_irf', 256, 5e-11, _decay())

    message = str(caught.value)
    assert 'Machine IRF Builder' in message, 'the error must say how to fix it'
    assert 'wrong' in message, (
        'the error must warn against substituting a gaussian, which returns '
        'plausible but incorrect lifetimes')
