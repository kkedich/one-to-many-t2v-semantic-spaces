"""
Adapted from https://github.com/open-mmlab/mmcv/blob/master/mmcv/utils/config.py
"""
import sys
from addict import Dict
from importlib import import_module
from pathlib import Path
from yapf.yapflib.yapf_api import FormatCode


class ConfigDict(Dict):

    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            value = super(ConfigDict, self).__getattr__(name)
        except KeyError:
            ex = AttributeError("'{}' object has no attribute '{}'".format(
                self.__class__.__name__, name))
        except Exception as exp:
            ex = exp
        else:
            return value
        raise ex


class Config:
    """
    A facility for config files.
    The interface is the same as a dict object and also allows
    access config values as attributes.

    Example:
        >>> cfg = Config('/home/configs/test.py')
        >>> cfg.only_filename
        "/home/configs/test.py"
        >>> cfg.item4
        'test'
        >>> cfg
        "Config [path: /home/configs/test.py]: "
        "{'item1': [1, 2], 'item2': {'a': 0}, 'item3': True, 'item4': 'test'}"
    """

    def __init__(self, filename: str = None):
        path_obj = Path(filename)
        if not path_obj.exists():
            raise FileNotFoundError(f"Config file <{filename}> not found.")

        if filename.endswith('.py'):
            module_name = path_obj.stem
            if '.' in module_name:
                raise ValueError('Dots are not allowed in config file path.')

            sys.path.insert(0, str(path_obj.resolve().parent))
            mod = import_module(module_name)
            sys.path.pop(0)

            cfg_dict = {
                name: value
                for name, value in mod.__dict__.items()
                if not name.startswith('__')
            }

            super(Config, self).__setattr__('_cfg_dict', ConfigDict(cfg_dict))
            super(Config, self).__setattr__('_filename', filename)
            if filename:
                with open(filename, 'r') as file:
                    super(Config, self).__setattr__('_text', file.read())
            else:
                super(Config, self).__setattr__('_text', '')
        else:
            raise IOError('Only .py config files are supported!')

    @property
    def filename(self):
        return self._filename

    @property
    def text(self):
        return self._text

    def __repr__(self):
        return f"Config (path: {self.filename}): {self._cfg_dict.__repr__()}"

    def __len__(self):
        return len(self._cfg_dict)

    def __getattr__(self, name):
        return getattr(self._cfg_dict, name)

    def __getitem__(self, name):
        return self._cfg_dict.__getitem__(name)

    def __setattr__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setattr__(name, value)

    def __setitem__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setitem__(name, value)

    def __iter__(self):
        return iter(self._cfg_dict)

    def dump(self, output_file):
        if output_file.endswith('.py'):
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(self.pretty_text)
        else:
            raise ValueError(f"File type is not supported: <{Path(output_file).suffix}>")

    @property
    def pretty_text(self):
        "https://github.com/open-mmlab/mmcv/blob/bfdd1f9c4281316f4f654672f0e9dbe1b0624ff0/mmcv/utils/config.py#L418C5-L511C1"

        indent = 4

        def _indent(s_, num_spaces):
            s = s_.split('\n')
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * ' ') + line for line in s]
            s = '\n'.join(s)
            s = first + '\n' + s
            return s

        def _format_basic_types(k, v, use_mapping=False):
            if isinstance(v, str):
                v_str = f"'{v}'"
            else:
                v_str = str(v)

            if use_mapping:
                k_str = f"'{k}'" if isinstance(k, str) else str(k)
                attr_str = f'{k_str}: {v_str}'
            else:
                attr_str = f'{str(k)}={v_str}'
            attr_str = _indent(attr_str, indent)

            return attr_str

        def _format_list(k, v, use_mapping=False):
            # check if all items in the list are dict
            if all(isinstance(_, dict) for _ in v):
                v_str = '[\n'
                v_str += '\n'.join(
                    f'dict({_indent(_format_dict(v_), indent)}),'
                    for v_ in v).rstrip(',')
                if use_mapping:
                    k_str = f"'{k}'" if isinstance(k, str) else str(k)
                    attr_str = f'{k_str}: {v_str}'
                else:
                    attr_str = f'{str(k)}={v_str}'
                attr_str = _indent(attr_str, indent) + ']'
            else:
                attr_str = _format_basic_types(k, v, use_mapping)
            return attr_str

        def _contain_invalid_identifier(dict_str):
            contain_invalid_identifier = False
            for key_name in dict_str:
                contain_invalid_identifier |= \
                    (not str(key_name).isidentifier())
            return contain_invalid_identifier

        def _format_dict(input_dict, outest_level=False):
            r = ''
            s = []

            use_mapping = _contain_invalid_identifier(input_dict)
            if use_mapping:
                r += '{'
            for idx, (k, v) in enumerate(input_dict.items()):
                is_last = idx >= len(input_dict) - 1
                end = '' if outest_level or is_last else ','
                if isinstance(v, dict):
                    v_str = '\n' + _format_dict(v)
                    if use_mapping:
                        k_str = f"'{k}'" if isinstance(k, str) else str(k)
                        attr_str = f'{k_str}: dict({v_str}'
                    else:
                        attr_str = f'{str(k)}=dict({v_str}'
                    attr_str = _indent(attr_str, indent) + ')' + end
                elif isinstance(v, list):
                    attr_str = _format_list(k, v, use_mapping) + end
                else:
                    attr_str = _format_basic_types(k, v, use_mapping) + end

                s.append(attr_str)
            r += '\n'.join(s)
            if use_mapping:
                r += '}'
            return r

        cfg_dict = self._cfg_dict.to_dict()
        text = _format_dict(cfg_dict, outest_level=True)
        # copied from setup.cfg
        yapf_style = dict(
            based_on_style='pep8',
            blank_line_before_nested_class_or_def=True,
            split_before_expression_after_opening_paren=True)
        text, _ = FormatCode(text, style_config=yapf_style, verify=True)

        return text
