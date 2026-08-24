from importlib.metadata import entry_points

import flimkit_bridge
from flimkit import plugins


def test_api_version_matches():
    assert flimkit_bridge.FLIMKIT_PLUGIN_API == plugins.API_VERSION


def test_tool_is_registered():
    found = plugins.get_tool('flimkit_bridge_open')
    assert found is not None
    assert found.menu_path == ('Tools',)
    assert callable(found.callback)


def test_entry_point_is_declared():
    names = [e.name for e in entry_points(group='flimkit.plugins')]
    assert 'flimkit_bridge' in names


def test_the_tool_is_not_named_after_one_client():
    assert plugins.get_tool('qupath_bridge_open') is None
    assert plugins.get_tool('fiji_bridge_open') is None
