__all__ = ['parser_config']


def parser_config(config):
    if 'MODEL_CONFIG' in config.keys():
        if 'NAME' in config['MODEL_CONFIG'].keys():
            if config['MODEL_CONFIG']['NAME'] == 'Yolo':
                if 'INPUT' in config['MODEL_CONFIG']:
                    if 'grid_size' in config['MODEL_CONFIG']['INPUT'].keys():
                        _v = config['MODEL_CONFIG']['INPUT']['grid_size']
                        if isinstance(_v, int):
                            _v = (_v, _v)
                        elif isinstance(_v, str):
                            _v = tuple(map(lambda x: int(x.strip()), _v.split(',')))[:2]
                        else:
                            raise ValueError
                        config['MODEL_CONFIG']['INPUT']['grid_size'] = _v
                    if 'LOSS_CONFIG' in config.keys():
                        if 'INPUT' in config['LOSS_CONFIG'].keys():
                            config['LOSS_CONFIG']['INPUT'].update(config['MODEL_CONFIG']['INPUT'])

    return config
