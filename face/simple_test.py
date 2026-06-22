from Motor import FaceController
from Communication import UARTDevice

import os
import time

if __name__ == '__main__' :
    # 确认配置文件
    CONFIG_YAML = '29_servo_config.yaml'
    # 确认端口号
    SERIAL_PORT = '/dev/usb_face'
    os.system('sudo chmod 777 ' + SERIAL_PORT)
    # 确认总舵机数量，一般不修改
    SERVO_NUM = 29
    # 确认当前文件所在路径
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 创建控制器
    face_controller = FaceController(UARTDevice(SERIAL_PORT, 9600), SERVO_NUM
                                              , current_dir + '/' + CONFIG_YAML)
    
    # 控制单个舵机，则传入的第一个参数为int类型的绝对角度值，第二个参数为int类型的舵机索引值，范围0-28，第三个参数为int类型的转动时间，单位ms
    # 例如设置0号舵机的角度为90度，并转动时间为200ms
    # face_controller.set_servo_angle_time_32(90, 0, 200)

    time.sleep(1)

    # 控制多个舵机，则传入的第一个参数为list类型的相对系数值，list的长度为舵机数量，第二个参数为list类型的舵机索引值，长度为总舵机数量29
    # 例如设置第0，1，9号舵机，并转动时间为400ms
    # 基于yaml配置文件得到的舵机初始的相对系数值如下
    list_factor_run = face_controller.list_factor
    print("list_factor_run: ", list_factor_run)
    # 修改对应的系数值
    list_factor_run[10] = 0.5
    list_factor_run[11] = 0.5
    list_factor_run[9] = 0.3
    # face_controller.set_servo_angle_time_32(list_factor_run, [7, 8, 9], 400)

    # 1、time >= 100ms; 2、check factor 0~1
    face_controller.set_servo_angle_time_32(list_factor_run, face_controller.list_index, 400)
    time.sleep(2)
    print("now_factor_run: ", face_controller.list_index)
