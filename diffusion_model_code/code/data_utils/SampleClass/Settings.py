'''
User selections that can help dictate behavior across wide areas of functionality. 

My initial attempt failed. This isn't high-priority, so just using a simple placeholder for now
'''

__attributes = {
        # bool: If False, run computations on the GPU regardless of object device unless
        'avoid_gpu':bool,
        # bool: If True, use double precision for methods regardless of object dtype
        'high_precision_computations':bool,
        # Default dtype, device -- force all objects to take on a specific dtype/device, if not None
        'dtype':[type(None),torch_dtype],
        'device':[type(None),torch_device]
    }

class Settings(dict):

    def __init__(self):
        # Where possible, run computations on the GPU regardless of the
        # object's primary device unless avoid_gpu == True
        self['avoid_gpu'] = False

        # Where possible, run computations with double precision regardless
        # of the object's primary dtype unless high_precision_computations == False
        self['high_precision_computations'] = True

        # If not None, force all objects to use the following dtype & device
        self['dtype'] = None
        self['device'] = None


'''
from torch.cuda import is_available as cuda_is_available
from torch import device as torch_device
from torch import dtype as torch_dtype
from torch import empty as torch_empty
from torch import float as torch_float
from torch import double as torch_double
from torch import float16 as torch_float16
from torch import bfloat16 as torch_bfloat16
'''

# These are kind of doubled for now... should come up with a better solution
'''
__dtype_conversion = {
    'float':torch_double,
    'torch.float':torch_float,
    'torch.float32':torch_float,
    'torch.float64':torch_double,
    'torch.double':torch_double,
    'torch.float16':torch_float16,
    'torch.half':torch_float16,
    'torch.bfloat16':torch_bfloat16
}
__bool_conversion = {
    'true':True,
    'false':False
}

def __format_values(value):
    if type(value) != str:
        # If not a string, it must not have been passed from file/shouldn't need formatting
        return value
    
    value = value.lower()
    if value == 'none':
        return None
    if value in __dtype_conversion:
        return __dtype_conversion[value]
    if value in __bool_conversion:
        return __bool_conversion[bool]

    # Long-term, certain settings will be able to take string values, not related to the above...
'''
    
'''
class Settings(dict):

    __defaults_fp = './__default_settings.txt'

    __attributes = {
        # bool: If False, run computations on the GPU regardless of object device unless
        'avoid_gpu':bool,
        # bool: If True, use double precision for methods regardless of object dtype
        'high_precision_computations':bool,
        # Default dtype, device -- force all objects to take on a specific dtype/device, if not None
        'dtype':[type(None),torch_dtype],
        'device':[type(None),torch_device]
    }
    __attribute_values = {}
    
    __invalid_line_exception_msg = "Lines in a settings file may contain any number of whitespaces and tabs.\n"
    __invalid_line_exception_msg+= "Otherwise, a line may:\n"
    __invalid_line_exception_msg+= "\t1. Be blank;\n"
    __invalid_line_exception_msg+= "\t2. Begin with '#', indicating that the line contains a comment; or\n"
    __invalid_line_exception_msg+= "\t3. Contain <Setting to be chosen>=<value, written in alpha-numeric form except device, which may contain ':', and dtype, which may contain '.'>"

    __dtype_conversion = {
        'none':None,
        'float':torch_double,
        'torch.float':torch_float,
        'torch.float32':torch_float,
        'torch.float64':torch_double,
        'torch.double':torch_double,
        'torch.float16':torch_float16,
        'torch.half':torch_float16,
        'torch.bfloat16':torch_bfloat16
    }
    
    def __init__(
        self,
        *,
        settings=None
    ):
        
        if type(settings) == Settings:
            self = settings.clone()
            for attr,item in settings:
                setattr(self,attr,item)
        else:
            # Initialize necesary attributes
            for attr in self.__attributes:
                self.__attribute_values[attr] = None
                #setattr(self, '__attribute_values'+attr, None)

            # Load the default values
            self.load_settings(self.__defaults_fp)
            
            # Override values with those provided by the user
            if settings is not None:
                # Should be a filepath to settings customizations, which override those provided herein
                self.load_settings(settings)
                
        # Ensure all settings are valid
        self.__validate()
    
    ######################################################
    # Setting/fetching attributes
    ######################################################
        
    def __format_bool(self,argname,value):
        if type(value) == bool:
            return value
        if value in ['True','true']:
            return True
        if value in ['False','false']:
            return False
        raise Exception(f'Setting parameter {argname} should be True, true, False, or false, but received {value}.')

    def __format_dtype(self,argname,value):
        if type(value) == torch_dtype or value is None:
            return value

        if value not in self.__dtype_conversion:
            raise Exception(f'The dtype argument {value} is not recognized! Must be one of: {list(self.__dtype_conversion.keys())}')
        self.__dtype = self.__dtype_conversion[value]
        
    def __format_device(self,argname,value):
        if type(value) == torch_device or value is None:
            return value
        assert type(value) == str, f"Device value must be NoneType, torch.device, or a string indicating one of these!"
        if value == 'none':
            return None
        
        # In case someone uses "torch.device('cuda:0')" or something in a save file. 
        # In case multiple parentheses are used, we'll take the innermost (if nested) or
        # rightmost (if not nested) parenthetical value. 
        v = value
        if '(' in v:
            v = v[v.rindex('('):]
        if ')' in v:
            v = v[:v.lindex(')')]
        try:
            return torch_empty(0,dtype=value).dtype
        except:
            raise Exception(f'Setting parameter {argname}={value} cannot be converted to torch.dtype instance!')

    def __getitem__(self,key):
        if key not in self.keys():
            raise Exception(f'{key} is not a valid settings attribute!')
        return self.__attribute_values[key]
        #getattr(self, '__'+str(key))
    
    def __setitem__(self, key, value):
        if key not in self.__attributes:
            raise f'{key} is not a recongized settings attribute!'

        if self.__attributes[key] == bool:
            #self[key] = self.__format_bool(key,value)
            #setattr(self,'__'+key,self.__format_bool(key,value))
            v = self.__format_bool(key,value)
            
        elif key == 'dtype':
            #self[key] = self.__format_dtype(key,value)
            #setattr(self,'__'+key,self.__format_dtype(key,value))
            v = self.__format_dtype(key,value)

        elif key == 'device':
            #self[key] = self.__format_device(key,value)
            setattr(self,'__'+key,self.__format_device(key,value))
            v = self.__format_device(key,value)
            
        else:
            raise Exception('It appears there is a bug in the Settings class!')
        self.__attribute_values[key] = v
        
    def keys(self):
        return self.__attributes.keys()
        
    ######################################################
    # Load settings from file
    ######################################################
    
    def __interpret_file_line(self,line_text,line_number,filepath):
        # Remove unneeded breaks
        original_line = line_text.rstrip('\n')
        line_text = original_line.replace(' ','').replace('\t','')
        if '#' in line_text:
            line_text = line_text[:line_text.index('#')]
            
        while True:
            # Ignore blank lines & commented lines
            if line_text == '':
                return

            # Remove invalid leading characters
            if not line_text[0].isalnum():
                line_text = line_text.lstrip(line_text[0])
                continue

            # Remove invalid trailing characters
            if not line_text[-1].isalnum() and line_text[-1] != "'":
                line_text = line_text.rstrip(line_text[-1])
                continue
            break
        
        err_msg = self.__invalid_line_exception_msg + '\n'
        err_msg+= f'Line {line_number}, "{original_line}", breaks these rules!'
        if '=' not in line_text:
            raise Exception(err_msg)

        vals = line_text.split('=')
        if len(vals) != 2:
            raise Exception(err_msg)
        if vals[0] == '' or vals[1] == '':
            raise Exception(err_msg)

        self[vals[0]] = vals[1].lower()
        try:
            self[vals[0]] = vals[1].lower()
        except Exception as e:
            raise type(e)(
                str(e) + '\n' + \
                f'This error occurs at line {line_number} of {filepath}.'
            )
    
    def load_settings(self,filepath):
        with open(filepath,'r') as f:
            for i,l in enumerate(f.readlines()):
                self.__interpret_file_line(l,i+1,filepath)

    def __validate(self):
        all_attrs = list(self.__attributes.keys())
        self_attrs = list(self.keys())

        valid_attrs = [
            attr for attr in self_attrs if attr in all_attrs
        ]
        unknown_attrs = [
            attr for attr in self_attrs if attr not in all_attrs
        ]
        missing_attrs = [
            attr for attr in all_attrs if attr not in self_attrs
        ]
        err_msg = ''
        if len(missing_attrs)>0:
            err_msg+= f'Settings object is missing the following attributes: {missing_attrs}.'
        if len(unknown_attrs)>0:
            if err_msg != '':
                err_msg+= '\n'
            err_msg+= f'Settings object has invalid attributes {unknown_attrs}.'
        for key in self_attrs:
            true_type = self.__attributes[key]
            if type(true_type) == list:
                if type(self[key]) not in true_type:
                    if err_msg != '':
                        err_msg+= '\n'
                    err_msg+= f'Settings parameter {key} type should be one of {true_type}, but received {type(self[key])}'
            elif type(self[key]) != true_type:
                if err_msg != '':
                    err_msg+= '\n'
                err_msg+= f'Setting parameter {key} should be of type {true_type}, but received {type(self[key])}'

        if err_msg != '':
            raise Exception(err_msg)
'''