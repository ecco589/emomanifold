import yaml
import threading
from enum import Enum

class CONFIG_MODE(Enum):
    DEVELOPMENT = 1
    TESTING = 2
    PRODUCTION = 3
    
def clamp(value, min_value, max_value): 
    return max(min_value, min(max_value, value))

def thread_it(func, *args):
    """ 将函数打包进线程 """
    myThread = threading.Thread(target=func, args=args, daemon=True)
    myThread.start()
def read_yaml(_path):
    with open(_path, "r") as f:
        yaml_info = yaml.safe_load(f)
    return yaml_info

def write_yaml(_dict, _path):
    with open(_path, "w", encoding="utf-8") as f:
        # os.chmod(_path, 0o755)
        data = yaml.dump(_dict,f)
        print(data)
        print("write yaml completed")

def validate_servo_config_yaml(file_path, n):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
        
        # 检查数据是否为空
        if not data:
            raise ValueError("YAML 文件内容为空")
        
        # 检查数据数量是否符合要求
        if len(data) != n:
            raise ValueError(f"YAML 文件中的条目数量应为 {n}，实际为 {len(data)}")
        
        # 检查每个条目的格式
        for key, value in data.items():
            # 检查键名格式
            if not key.startswith('A') or not key[1:].isdigit():
                raise ValueError(f"键名 {key} 格式不正确，应为 A 后跟数字")
            
            index = int(key[1:])
            
            # 检查 channel_idx 是否与键名匹配
            if value.get('channel_idx') != index:
                raise ValueError(f"键名 {key} 对应的 channel_idx 应为 {index}，实际为 {value.get('channel_idx')}")
            
            # 检查 start_deg 和 end_deg 是否不同
            if value.get('start_deg') == value.get('end_deg'):
                raise ValueError(f"键名 {key} 的 start_deg 和 end_deg 不能相同")
            
            # 检查角度值是否在 0 到 180 之间
            for deg_key in ['start_deg', 'end_deg', 'temp_deg']:
                deg_value = value.get(deg_key)
                if deg_value is None or not (0 <= deg_value <= 180):
                    raise ValueError(f"键名 {key} 的 {deg_key} 应在 0 到 180 之间，实际为 {deg_value}")
            
            # 检查 temp_deg 是否在 start_deg 和 end_deg 之间
            start_deg = value.get('start_deg')
            end_deg = value.get('end_deg')
            temp_deg = value.get('temp_deg')
            if not (min(start_deg, end_deg) <= temp_deg <= max(start_deg, end_deg)):
                raise ValueError(f"键名 {key} 的 temp_deg 应在 start_deg 和 end_deg 之间，实际为 {temp_deg}")
        
        print("YAML 文件验证通过")
    
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 不存在")
    except yaml.YAMLError as e:
        print(f"错误：YAML 文件格式错误 - {e}")
    except ValueError as e:
        print(f"错误：{e}")

def angle2factor(_deg:int, _start_deg:int, _end_deg:int):
    diff = _end_deg - _start_deg
    factor = (_deg - _start_deg) / diff 
    return factor

def factor2angle(_factor:float, _start_deg:int, _end_deg:int):
    diff = _end_deg - _start_deg
    angle = _factor * diff + _start_deg
    return round(angle)



### 十进制转换为十六进制，共八位
def int2hex_str1(_val):
    hex_num = format(int(_val), '02x')
    return hex_num

### 十进制转换为十六进制，共十六位，低八位在前
def int2hex_str2(_val):
    hex_num = format(int(_val), '04x')
    high_byte = hex_num[:2]
    low_byte = hex_num[2:]
    return low_byte + ' ' + high_byte

def angle2pulse(_angle):
    delta_pulse_per_angle = 11.111    # (2500 - 500) / 180 = 11.111...
    pulse = 500 + delta_pulse_per_angle * float(_angle)
    return pulse