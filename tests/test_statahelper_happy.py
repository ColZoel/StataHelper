import pytest

from StataHelper import StataHelper


def test_stata_init():
    stata = StataHelper()
    assert stata is not None


def test_stata_init_path():
    stata = StataHelper('C:/Program Files/Stata17/StataMP-64.exe')
    assert stata.stata_path == 'C:/Program Files/Stata17/StataMP-64.exe'


def test_stata_init_path_error():
    with pytest.raises(ValueError) as exp:
        stata = StataHelper('C:/Program Files/Stata17/StataMP-64.exe')
    assert str(exp.value) == 'StataHelper executable not found at C:/Program Files/Stata17/StataMP-64.exe'


def test_stata_init_config():
    stata = StataHelper(config='tests/stata_config.yaml')
    assert stata.config == {
        'stata_path': "C:\\Program Files\\Stata18",
        'edition': 'mp',
        'display_splash': False,
        'maxcores': None,
        'buffer': 1
    }
    assert stata.stata_path == "C:\\Program Files\\Stata18"
    assert stata.edition == 'mp'
    assert stata.splash is False
    assert stata.maxcores is None
    assert stata.set_graph_format is None
    assert stata.set_graph_size is None
    assert stata.set_graph_show is None
    assert stata.set_command_show is None
    assert stata.set_autocompletion is None
    assert stata.set_streaming_output is None
    assert stata.set_output_file is None


def test_stata_init_config_and_args():
    stata = StataHelper(config='tests/stata_config.yaml', set_graph_show=True, set_command_show=True)
    assert stata.config == {
        'stata_path': "C:\\Program Files\\Stata18",
        'edition': 'mp',
        'display_splash': False,
        'maxcores': 4,
        'buffer': 1
    }
    assert stata.stata_path == "C:\\Program Files\\Stata18"
    assert stata.edition == 'mp'
    assert stata.splash is False
    assert stata.maxcores is None
    assert stata.set_graph_format is None
    assert stata.set_graph_size is None
    assert stata.set_graph_show is True
    assert stata.set_command_show is True
    assert stata.set_autocompletion is None
    assert stata.set_streaming_output is None
    assert stata.set_output_file is None


def test_stata_init_config_overlap():
    stata = StataHelper(config='tests/stata_config.yaml', splash=False)
    assert stata.config == {
        'stata_path': "C:\\Program Files\\Stata18",
        'edition': 'mp',
        'splash': True,
        'maxcores': 4,
        'buffer': 1
    }
    assert stata.stata_path == "C:\\Program Files\\Stata18"
    assert stata.edition == 'mp'
    assert stata.splash is True
    assert stata.maxcores is None
    assert stata.set_graph_format is None
    assert stata.set_graph_size is None
    assert stata.set_graph_show is True
    assert stata.set_command_show is True
    assert stata.set_autocompletion is None
    assert stata.set_streaming_output is None
    assert stata.set_output_file is None
    