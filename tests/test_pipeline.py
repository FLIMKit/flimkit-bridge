import os
from pathlib import Path

import pytest

from flimkit_bridge import pipeline

MOSAIC = os.environ.get('FLIMKIT_TEST_MOSAIC', '')

needs_mosaic = pytest.mark.skipif(
    not MOSAIC or not Path(MOSAIC).exists(),
    reason='set FLIMKIT_TEST_MOSAIC to a .xlif or .lif mosaic')


def test_defaults_describe_every_parameter():
    described = pipeline.defaults()
    assert set(described['values']) == {entry['key'] for entry in pipeline.SCHEMA}
    assert all('label' in entry and 'type' in entry for entry in described['schema'])


def test_unknown_parameters_are_refused():
    with pytest.raises(ValueError) as raised:
        pipeline.merge_params({'nexp': 2})
    assert 'unknown tile fit parameter' in str(raised.value)


def test_tau_bounds_are_checked():
    with pytest.raises(ValueError):
        pipeline.merge_params({'tau_min_ns': 6.0, 'tau_max_ns': 1.0})


def test_a_missing_container_is_refused():
    with pytest.raises(ValueError):
        pipeline.resolve('/nowhere/scan.xlif')


@needs_mosaic
def test_a_container_resolves_to_its_tiles():
    container, tile_dir, output_dir, basename, n_tiles = pipeline.resolve(MOSAIC)
    assert n_tiles > 1
    assert Path(tile_dir).is_dir()
    assert basename == Path(MOSAIC).stem
    assert str(output_dir).startswith(str(tile_dir))


@needs_mosaic
def test_the_pipeline_gets_the_arguments_it_needs():
    container, tile_dir, output_dir, basename, _ = pipeline.resolve(MOSAIC)
    params = pipeline.merge_params({'n_exp': 1, 'binning': 2})
    args = pipeline.build_args(container, tile_dir, output_dir, basename, params)
    for name in ('xlif', 'ptu_dir', 'output_dir', 'ptu_basename', 'rotate_tiles',
                 'nexp'):
        assert hasattr(args, name), f'_run_tile_fit reads args.{name}'
    assert args.nexp == 1
    assert args.binning == 2


@pytest.mark.parametrize('described', [
    pytest.param('pipeline', id='pipeline'),
    pytest.param('fitting', id='roi_fit'),
])
def test_every_field_the_dialog_builds_has_a_value(described):
    import importlib

    module = importlib.import_module(f'flimkit_bridge.{described}')
    defaults = module.defaults()
    shown = [entry['key'] for entry in defaults['schema'] if not entry['advanced']]
    missing = [key for key in shown
               if key not in defaults['values'] or defaults['values'][key] is None]
    assert not missing, (
        f'{missing} would reach the QuPath dialog with no value, and Gson raises '
        'on a null default')


def test_choices_survive_the_round_trip_through_the_dialog():
    assert pipeline.merge_params({'binning': '4'})['binning'] == 4
    assert pipeline.merge_params({'pipeline': 'tile_fit'})['pipeline'] == 'tile_fit'
    with pytest.raises(ValueError):
        pipeline.merge_params({'binning': '3'})


def positions(count=3):
    return [{'file': f'R 2_s{i + 1}.ptu'} for i in range(count)]


def lay_out(root, container_dir, tile_dir):
    made = root / container_dir if container_dir else root
    made.mkdir(parents=True, exist_ok=True)
    container = made / 'R 2.xlif'
    container.write_text('')
    tiles = root / tile_dir
    tiles.mkdir(parents=True, exist_ok=True)
    for entry in positions():
        (tiles / entry['file']).write_bytes(b'')
    return container, tiles


def test_tiles_are_found_in_a_folder_beside_the_container(tmp_path):
    container, tiles = lay_out(tmp_path, '', 'PTU.sptw')
    assert pipeline.locate_tiles(container, positions()) == tiles


def test_tiles_are_found_when_the_container_sits_in_its_own_folder(tmp_path):
    container, tiles = lay_out(tmp_path, 'Metadata', 'PTU.sptw')
    assert pipeline.locate_tiles(container, positions()) == tiles


def test_tiles_beside_the_container_win(tmp_path):
    container, _ = lay_out(tmp_path, '', 'PTU.sptw')
    for entry in positions():
        (container.parent / entry['file']).write_bytes(b'')
    assert pipeline.locate_tiles(container, positions()) == container.parent


def test_an_explicit_tile_folder_is_used_first(tmp_path):
    container, _ = lay_out(tmp_path, 'Metadata', 'PTU.sptw')
    elsewhere = tmp_path / 'somewhere else'
    elsewhere.mkdir()
    for entry in positions():
        (elsewhere / entry['file']).write_bytes(b'')
    assert pipeline.locate_tiles(container, positions(), elsewhere) == elsewhere


def test_missing_tiles_name_everywhere_that_was_searched(tmp_path):
    made = tmp_path / 'Metadata'
    made.mkdir()
    container = made / 'R 2.xlif'
    container.write_text('')
    with pytest.raises(FileNotFoundError) as raised:
        pipeline.locate_tiles(container, positions())
    message = str(raised.value)
    assert str(made) in message
    assert str(tmp_path) in message


def test_tile_fit_is_the_default_pipeline():
    assert pipeline.defaults()['values']['pipeline'] == 'tile_fit'


def test_products_carry_real_units(tmp_path):
    import numpy as np
    import tifffile

    counts = np.array([[0.0, 3.0], [7.0, 12.0]], dtype=np.float32)
    np.save(str(tmp_path / 'R_2_intensity.npy'), counts)
    tifffile.imwrite(str(tmp_path / 'R_2_tau_intensity_weighted_fullrange.tif'),
                     np.array([[1.2, 2.4], [3.6, 4.8]], dtype=np.float32))
    products = pipeline.summarise(None, tmp_path)['products']
    by_unit = {entry['unit']: entry for entry in products}
    assert set(by_unit) == {'photons', 'ns'}
    written = tifffile.imread(by_unit['photons']['file'])
    assert written.dtype == np.uint16
    assert np.array_equal(written.astype(np.float32), counts)


def test_products_stay_float_when_counts_are_not_whole(tmp_path):
    import numpy as np
    import tifffile

    counts = np.array([[0.0, 3.5]], dtype=np.float32)
    np.save(str(tmp_path / 'R_2_intensity.npy'), counts)
    products = pipeline.summarise(None, tmp_path)['products']
    written = tifffile.imread(products[0]['file'])
    assert written.dtype == np.float32
    assert np.array_equal(written, counts)
