import os
import shutil
from collections import OrderedDict
from configparser import ConfigParser

class Config:
    def __init__(self, main_conf_path, model_conf_path):
        self.main_conf_path = main_conf_path        
        self.main_config = self.read_config(os.path.join(main_conf_path, 'main_config.cfg'))

        model_name = self.main_config['Experiment']['model_name']
        self.model_conf_path = model_conf_path
        self.model_config = self.read_config(os.path.join(model_conf_path, '%s.cfg' % model_name))

    def read_config(self, conf_path):
        conf_dict = OrderedDict() 

        config = ConfigParser()
        config.read(conf_path)
        for section in config.sections():
            section_config = OrderedDict(config[section].items())
            conf_dict[section] = self.type_ensurance(section_config)

        return conf_dict


    def ensure_value_type(self, v):
        BOOLEAN = {'false': False, 'False': False,
                   'true': True, 'True': True}
        if isinstance(v, str):
            try:
                value = eval(v)
                if not isinstance(value, (str, int, float, list, tuple)):
                    value = v
            except:
                if v in BOOLEAN:
                    v = BOOLEAN[v]
                value = v
        else:
            value = v
        return value

    def type_ensurance(self, config):
        BOOLEAN = {'false': False, 'False': False,
                   'true': True, 'True': True}

        for k, v in config.items():
            try:
                value = eval(v)
                if not isinstance(value, (str, int, float, list, tuple)):
                    value = v
            except:
                if v in BOOLEAN:
                    v = BOOLEAN[v]
                value = v
            config[k] = value
        return config

    def get_param(self, section, param):
        if section in self.main_config:
            section = self.main_config[section]
        elif section in self.model_config:
            section = self.model_config[section]
        else:
            raise NameError("There are not the parameter named '%s'" % section)

        if param in section:
            value = section[param]
        else:
            raise NameError("There are not the parameter named '%s'" % param)

        return value

    def update_params(self, params):
        # for now, assume 'params' is dictionary
        new_params = OrderedDict()
        if params.get('model_name'):
            new_params['model_name'] = params.get('model_name')
            del params['model_name']
        new_params.update(params)
        params = new_params
        for k, v in params.items():
            updated=False
            for section in self.main_config:
                if k in self.main_config[section]:
                    self.main_config[section][k] = self.ensure_value_type(v)
                    updated = True
                    break
            
            if not updated:
                for section in self.model_config:
                    if k in self.model_config[section]:
                        self.model_config[section][k] = self.ensure_value_type(v)
                        updated = True
                        break

            if not updated:
                # raise ValueError
                print('Parameter not updated. \'%s\' not exists.' % k)
            elif updated and k == 'model_name':
                model_name = self.main_config['Experiment']['model_name']
                self.model_config = self.read_config(os.path.join(self.model_conf_path, '%s.cfg' % model_name))


    def save(self, base_dir):
        def helper(section_k, section_v):
            sec_str = '[%s]\n' % section_k
            for k, v in section_v.items():
                sec_str += '%s=%s\n' % (str(k), str(v))
            sec_str += '\n'
            return sec_str
        
        # save main config
        main_conf_str =''
        for section in self.main_config:
            main_conf_str += helper(section, self.main_config[section])
        with open(os.path.join(base_dir, 'main_config.cfg'), 'wt') as f:
            f.write(main_conf_str)

        # save model config
        model_conf_str =''
        for section in self.model_config:
            model_conf_str += helper(section, self.model_config[section])
        with open(os.path.join(base_dir, '%s.cfg' % self.main_config['Experiment']['model_name']), 'wt') as f:
            f.write(model_conf_str)

        # copy model.py
        shutil.copy(f"./model/{self.main_config['Experiment']['model_name']}.py", os.path.join(base_dir, f"{self.main_config['Experiment']['model_name']}.py"))
        
        print('main / model config saved in %s' % base_dir)

    def __getitem__(self, item):
        if not isinstance(item, str):
            raise TypeError("index must be a str")

        if item in self.main_config:
            section = self.main_config[item]
        elif item in self.model_config:
            section = self.model_config[item]
        else:
            raise NameError("There are not the parameter named '%s'" % item)
        return section

    def __str__(self):
        config_str = '\n'

        config_str += '>>>>> Main Config\n'
        for section in self.main_config:
            config_str += '[%s]\n' % section
            config_str += '\n'.join(['{}: {}'.format(k, self.main_config[section][k]) for k in self.main_config[section]])
            config_str += '\n\n'

        config_str += '>>>>> model Config\n'
        for section in self.model_config:
            config_str += '[%s]\n' % section
            config_str += '\n'.join(['{}: {}'.format(k, self.model_config[section][k]) for k in self.model_config[section]])
            config_str += '\n\n'

        return config_str

if __name__ == '__main__':
    param = Config('../main_config.cfg')

    print(param)
