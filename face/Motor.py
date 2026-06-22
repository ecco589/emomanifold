from utility import *

class FaceController():
    def __init__(self, _interface, _servo_num:int, _yaml_file:str):
        self.interface = _interface
        self.servo_num = _servo_num
        self.yaml_file = _yaml_file

        self.list_index = [99] * self.servo_num
        self.list_start_deg = [0] * self.servo_num
        self.list_end_deg = [0] * self.servo_num
        self.list_factor = [99] * self.servo_num

        self.init_data()

    def open(self):
        try:
            self.interface.open()
        except Exception as e:
            print(e)
            print('Error: cannot connect to serial port')
    
    def close(self):
        try:
            self.interface.close()
        except Exception as e:
            print(e)
            print('Error: cannot close serial port')

    def init_data(self):
        try:
            validate_servo_config_yaml(self.yaml_file, self.servo_num)
            self.servo_name2servo_info = read_yaml(self.yaml_file)

            for servo_name, servo_info in self.servo_name2servo_info.items():
                # print(servo_name)
                chan_idx = servo_info["channel_idx"]
                start_deg = servo_info['start_deg']
                end_deg = servo_info['end_deg']
                temp_deg = servo_info['temp_deg']
                factor = angle2factor(temp_deg, start_deg, end_deg)

                self.list_index[chan_idx] = chan_idx
                self.list_start_deg[chan_idx] = start_deg
                self.list_end_deg[chan_idx] = end_deg
                self.list_factor[chan_idx] = factor

        except Exception as e:
            print(e)
        print("init data completed...")

    def set_servo_reset(self):
        self.set_servo_angle_time_32(self.list_factor, self.list_index, 400)


    def set_servo_angle_time_32(self, _angle, _index, _time):
        cmd = self.data2cmd_32(_angle, _index, _time)
        send_cmd = bytes.fromhex(cmd)
        self.interface.send_command(send_cmd)

    def data2cmd_32(self, _data, _index, _time):
        cmd = ''
        ### type(_index) == int, _data is single value(degree)
        if type(_index) == int:
            servo_num = 1
            frame_header = '55 55 ' + int2hex_str1(servo_num * 3 + 5) + ' 03 '
            frame_param = int2hex_str1(servo_num) + ' ' + int2hex_str2(_time) + ' '
            cmd += int2hex_str1(_index) + ' ' + int2hex_str2(angle2pulse(_data)) + ' '
        ### type(_index) == list, _data is list(factor)
        elif type(_index) == list:
            servo_num = len(_index) 
            frame_header = '55 55 ' + int2hex_str1(servo_num * 3 + 5) + ' 03 '
            frame_param = int2hex_str1(servo_num) + ' ' + int2hex_str2(_time) + ' '
            for idx in _index:
                # angle = factor2angle(_angle[idx], idx)
                angle = factor2angle(_data[idx], self.list_start_deg[idx], self.list_end_deg[idx])
                cmd += int2hex_str1(idx) + ' ' + int2hex_str2(angle2pulse(angle)) + ' '
        else:
            print("Error: param index type not int or list")

        return frame_header + frame_param + cmd
    
    def angle2factor_from_index(self, _angle, _index):
        factor = angle2factor(int(_angle), self.list_start_deg[_index], self.list_end_deg[_index])
        return factor
    
    def factor2angle_from_index(self, _factor, _index):
        angle = factor2angle(float(_factor), self.list_start_deg[_index], self.list_end_deg[_index])
        return angle