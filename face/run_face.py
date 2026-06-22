from utility import CONFIG_MODE, clamp, thread_it, read_yaml
from Motor import FaceController
from Communication import UARTDevice

import sys
import os
import tkinter as tk
import tkinter.ttk as ttk
from tkinter.messagebox import askyesno
from functools import partial
import yaml

# 确认配置文件
CONFIG_YAML = '29_servo_config.yaml'
# 确认配置模式
CURRENT_CONFIG_MODE = CONFIG_MODE.PRODUCTION



SERIAL_PORT = ''
# 开发或测试模式，需自行修改对应串口号
if CURRENT_CONFIG_MODE == CONFIG_MODE.DEVELOPMENT or CURRENT_CONFIG_MODE == CONFIG_MODE.TESTING:
    SERIAL_PORT = '/dev/ttyUSB1'
    os.system('sudo chmod 777 ' + SERIAL_PORT)
# 生产模式，串口号已绑定
elif CURRENT_CONFIG_MODE == CONFIG_MODE.PRODUCTION:
    SERIAL_PORT = '/dev/usb_face'
else:
    print('Error: Invalid config mode')




class GUI:
    def __init__(self, _root):
        self.root = _root

        self.SERVO_NUM = 29
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.face_controller = FaceController(UARTDevice(SERIAL_PORT, 9600), self.SERVO_NUM
                                              , self.current_dir + '/' + CONFIG_YAML)
        
        self.structure()

    def close_window(self):
        ans = askyesno(title='', message='是否确定退出程序？')
        if ans:
            self.root.destroy()
            sys.exit()
        else:
            return None
        
    def structure(self):
        self.root.title("Face GUI")
        self.root.geometry('1300x800')
        self.root.protocol('WM_DELETE_WINDOW', lambda: thread_it(self.close_window))

        # 上侧框架，用于声明基本信息
        frame_upper = ttk.Frame(self.root)
        frame_upper.pack(side='top',fill='x',ipadx=10, ipady=10, expand=0)
        # 中间框架，用于单舵机拖动条
        frame_scale_single = ttk.Frame(self.root)
        frame_scale_single.pack(side='top',fill='x',ipadx=10, ipady=10, expand=0)
        # 下侧框架，用于按键控制和多舵机拖动条
        frame_lower = ttk.Frame(self.root)
        frame_lower.pack(side='top',fill='x',ipadx=10, ipady=10, expand=0)
        # 按键控制部分，下侧框架左侧
        frame_click = ttk.Frame(frame_lower)
        frame_click.pack(side='left',fill='x',ipadx=10, ipady=10, expand=0)
        # 多舵机拖动条部分，下侧框架右侧
        frame_scale_multi = ttk.Frame(frame_lower)
        frame_scale_multi.pack(side='left',fill='x',ipadx=10, ipady=10, expand=0)

        # 基本信息部分
        self.label = tk.Label(frame_upper, text="Face GUI for hobbs", font=('SimHei', 25))
        self.label.grid(padx=20, pady=0, row=0, column=0)

        # 单舵机拖动条部分
        self.tab_scale_single = ttk.Notebook(frame_scale_single)
        self.tab_scale_single.grid(padx=20, pady=0, row=1, column=0)

        tab_scale_single_upper = ttk.Frame(self.tab_scale_single)
        tab_scale_single_upper.grid()
        self.tab_scale_single.add(tab_scale_single_upper, text='Upper face')

        tab_scale_single_lower = ttk.Frame(self.tab_scale_single)
        tab_scale_single_lower.grid()
        self.tab_scale_single.add(tab_scale_single_lower, text='Lower face')
        
        separate_index = 15     # index < separate_index 的为上半脸， >= separate_index 的为下半脸

        self.scale_angle_dict = dict()
        self.scale_factor_dict = dict()
        count = 0; g_row = 0; g_column = 0

        # 遍历yaml文件，为每个舵机创建单舵机拖动条
        for servo_name, servo_info in self.face_controller.servo_name2servo_info.items():
            label = servo_name + ": " + f"{servo_info['info']}"
            index = servo_info["channel_idx"]
            if index == separate_index:
                count = 0

            if index < separate_index:
                window = tab_scale_single_upper
            elif index < self.SERVO_NUM:
                window = tab_scale_single_lower
            else:
                print(f"Error: index {index} is out of range")

            func_scale_set_angle  = partial(self.scale_set_angle, _name = servo_name, _index = index)
            self.scale_angle_dict[servo_name] = tk.Scale(window, from_=servo_info['start_deg'], to=servo_info['end_deg'], 
                                            orient=tk.HORIZONTAL, length=300, font=('SimHei', 10), 
                                            label=label, command=func_scale_set_angle)
            self.scale_angle_dict[servo_name].set(servo_info["temp_deg"])

            func_scale_set_factor = partial(self.scale_set_factor, _name = servo_name, _index = index)
            self.scale_factor_dict[servo_name] = tk.Scale(window, from_=0, to=1,
                                      orient=tk.HORIZONTAL, length=200, resolution = 0.01, digits= 3,
                                      label='', command=func_scale_set_factor)
            
            self.scale_factor_dict[servo_name].set(self.face_controller.angle2factor_from_index(servo_info["temp_deg"], index))

            g_row = int(count / 4)
            g_column = (count % 4 )
            scale_column = g_column * 2
            
            scale1_row = g_row * 2
            scale2_row = scale1_row + 1

            self.scale_angle_dict[servo_name].grid(row = scale1_row, column=scale_column)
            self.scale_factor_dict[servo_name].grid(row = scale2_row, column=scale_column)

            count += 1
        
        # 按键控制部分
        self.save_path = self.current_dir + "/expression29/"
        self.yaml_file_list = []
        self.listbox = tk.Listbox(frame_click, width=15, height=12)
        self.listbox.grid(row=0, column=2)
        self.get_yaml_list()

        self.save_file_label = tk.Label(frame_click, text="文件名：")
        self.save_file_label.grid(row=0, column=0)

        self.save_file_entry = tk.Entry(frame_click, width=10)
        self.save_file_entry.grid(row=0, column=1)

        self.save_button = tk.Button(frame_click, text="保存当前", command=self.save_value)
        self.save_button.grid(row=1, column=0)
                    
        self.load_button = tk.Button(frame_click, text="载入选中", command=self.load_button_click)
        self.load_button.grid(row=1, column=1)
                
        self.delete_button = tk.Button(frame_click, text="删除选中", command=self.delete_button_click)
        self.delete_button.grid(row=1, column=2)
        
        self.reset_button = tk.Button(frame_click, text="恢复默认", command=self.reset_button_click)
        self.reset_button.grid(row=1, column=3)                    

        # 多舵机拖动条部分
        self.tab_scale_multi = ttk.Notebook(frame_scale_multi)
        self.tab_scale_multi.grid(padx=20, pady=0, row=0, column=0)

        self.tab_scale_multi.bind('<<NotebookTabChanged>>', self.tab_scale_multi_select)
        
        tab_scale_multi_upper = ttk.Frame(self.tab_scale_multi)
        tab_scale_multi_upper.grid()
        self.tab_scale_multi.add(tab_scale_multi_upper, text='Upper face')
        
        tab_scale_multi_lower = ttk.Frame(self.tab_scale_multi)
        tab_scale_multi_lower.grid()
        self.tab_scale_multi.add(tab_scale_multi_lower, text='Lower face')

        self.scale_eyebrow_head_vertical = tk.Scale(tab_scale_multi_upper, from_=0, to=1,
                                      orient=tk.HORIZONTAL, length=200, resolution = 0.01, digits= 3,
                                      label='眉头同步 下——上', command=self.func_eyebrow_head_vertical)
        self.scale_eyebrow_head_vertical.set(self.face_controller.list_factor[0])
        self.scale_eyebrow_head_vertical.grid(row=0, column=0)
        
        self.scale_eyebrow_head_horizontal = tk.Scale(tab_scale_multi_upper, from_=0, to=1,
                                      orient=tk.HORIZONTAL, length=200, resolution = 0.01, digits= 3,
                                      label='眉头同步 内——外', command=self.func_eyebrow_head_horizontal)
        self.scale_eyebrow_head_horizontal.set(self.face_controller.list_factor[2])
        self.scale_eyebrow_head_horizontal.grid(row=0, column=1)
        
        self.scale_eyebrow_tail = tk.Scale(tab_scale_multi_upper, from_=0, to=1,
                                      orient=tk.HORIZONTAL, length=200, resolution = 0.01, digits= 3,
                                      label='眉尾同步 上——下', command=self.func_eyebrow_tail)
        self.scale_eyebrow_tail.set(self.face_controller.list_factor[4])
        self.scale_eyebrow_tail.grid(row=0, column=2)

        self.scale_upper_eyelid = tk.Scale(tab_scale_multi_upper, from_=0, to=1,
                                      orient=tk.HORIZONTAL, length=200, resolution = 0.01, digits= 3,
                                      label='上眼皮同步 开——合', command=self.func_upper_eyelid)
        self.scale_upper_eyelid.set(self.face_controller.list_factor[6])
        self.scale_upper_eyelid.grid(row=1, column=0)
        
        self.scale_eyeball_vertical = tk.Scale(tab_scale_multi_upper, from_=0, to=1,
                                      orient=tk.HORIZONTAL, length=200, resolution = 0.01, digits= 3,
                                      label='眼球同步 下——上', command=self.func_eyeball_vertical)
        self.scale_eyeball_vertical.set(self.face_controller.list_factor[8])
        self.scale_eyeball_vertical.grid(row=1, column=1)
        
        self.scale_eyeball_horizontal = tk.Scale(tab_scale_multi_upper, from_=0, to=1,
                                      orient=tk.HORIZONTAL, length=200, resolution = 0.01, digits= 3,
                                      label='眼球同步 外——内', command=self.func_eyeball_horizontal)
        self.scale_eyeball_horizontal.set(self.face_controller.list_factor[10])
        self.scale_eyeball_horizontal.grid(row=1, column=2)
        
        self.scale_cheek = tk.Scale(tab_scale_multi_upper, from_=0, to=1,
                                      orient=tk.HORIZONTAL, length=200, resolution = 0.01, digits= 3,
                                      label='脸颊同步 下——上', command=self.func_cheek)
        self.scale_cheek.set(self.face_controller.list_factor[12])
        self.scale_cheek.grid(row=2, column=0)
        
        self.scale_upper_lip = tk.Scale(tab_scale_multi_lower, from_=0, to=1,
                                      orient=tk.HORIZONTAL, length=200, resolution = 0.01, digits= 3,
                                      label='上嘴唇同步 上——下', command=self.func_upper_lip)
        self.scale_upper_lip.set(self.face_controller.list_factor[16])
        self.scale_upper_lip.grid(row=0, column=0)
        
        self.scale_lower_lip = tk.Scale(tab_scale_multi_lower, from_=0, to=1,
                                      orient=tk.HORIZONTAL, length=200, resolution = 0.01, digits= 3,
                                      label='下嘴唇同步 上——下', command=self.func_lower_lip)
        self.scale_lower_lip.set(self.face_controller.list_factor[18])
        self.scale_lower_lip.grid(row=0, column=1)
        
        self.scale_upper_mouth = tk.Scale(tab_scale_multi_lower, from_=0, to=1,
                                      orient=tk.HORIZONTAL, length=200, resolution = 0.01, digits= 3,
                                      label='嘴角上侧 前——后', command=self.func_upper_mouth)
        self.scale_upper_mouth.set(self.face_controller.list_factor[23])
        self.scale_upper_mouth.grid(row=1, column=0)
        
        self.scale_lower_mouth = tk.Scale(tab_scale_multi_lower, from_=0, to=1,
                                      orient=tk.HORIZONTAL, length=200, resolution = 0.01, digits= 3,
                                      label='嘴角下侧 上——下', command=self.func_lower_mouth)
        self.scale_lower_mouth.set(self.face_controller.list_factor[24])
        self.scale_lower_mouth.grid(row=1, column=1)


    def tab_scale_multi_select(self, *args):
        tab_index = self.tab_scale_multi.index('current')
        self.tab_scale_single.select(tab_index)
        
    def func_eyebrow_head_vertical(self, _factor):
        factor = float(_factor)
        list_run_factor = self.face_controller.list_factor.copy()
        list_run_factor[0] = factor
        list_run_factor[1] = factor
        self.scale_factor_dict['A0'].set(factor)
        self.scale_factor_dict['A1'].set(factor)
    
    def func_eyebrow_head_horizontal(self, _factor):
        factor = float(_factor)
        list_run_factor = self.face_controller.list_factor.copy()
        list_run_factor[2] = factor
        list_run_factor[3] = factor
        self.scale_factor_dict['A2'].set(factor)
        self.scale_factor_dict['A3'].set(factor)
    
    def func_eyebrow_tail(self, _factor):
        factor = float(_factor)
        list_run_factor = self.face_controller.list_factor.copy()
        list_run_factor[4] = factor
        list_run_factor[5] = factor
        self.scale_factor_dict['A4'].set(factor)
        self.scale_factor_dict['A5'].set(factor)
    
    def func_upper_eyelid(self, _factor):
        factor = float(_factor)
        list_run_factor = self.face_controller.list_factor.copy()
        list_run_factor[6] = factor
        list_run_factor[7] = factor
        self.scale_factor_dict['A6'].set(factor)
        self.scale_factor_dict['A7'].set(factor)
    
    def func_eyeball_vertical(self, _factor):
        factor = float(_factor)
        list_run_factor = self.face_controller.list_factor.copy()
        list_run_factor[8] = factor
        list_run_factor[9] = factor
        self.scale_factor_dict['A8'].set(factor)
        self.scale_factor_dict['A9'].set(factor)
    
    def func_eyeball_horizontal(self, _factor):
        factor = float(_factor)
        list_run_factor = self.face_controller.list_factor.copy()
        list_run_factor[10] = factor
        list_run_factor[11] = factor
        self.scale_factor_dict['A10'].set(factor)
        self.scale_factor_dict['A11'].set(factor)
    
    def func_cheek(self, _factor):
        factor = float(_factor)
        list_run_factor = self.face_controller.list_factor.copy()
        list_run_factor[12] = factor
        list_run_factor[13] = factor
        self.scale_factor_dict['A12'].set(factor)
        self.scale_factor_dict['A13'].set(factor)
    
    def func_upper_lip(self, _factor):
        factor = float(_factor)
        list_run_factor = self.face_controller.list_factor.copy()
        list_run_factor[16] = factor
        list_run_factor[19] = factor
        list_run_factor[20] = factor
        self.scale_factor_dict['A16'].set(factor)
        self.scale_factor_dict['A19'].set(factor)
        self.scale_factor_dict['A20'].set(factor)
    
    def func_lower_lip(self, _factor):
        factor = float(_factor)
        list_run_factor = self.face_controller.list_factor.copy()
        list_run_factor[18] = factor
        list_run_factor[21] = factor
        list_run_factor[22] = factor
        self.scale_factor_dict['A18'].set(factor)
        self.scale_factor_dict['A21'].set(factor)
        self.scale_factor_dict['A22'].set(factor)

    def func_upper_mouth(self, _factor):
        factor = float(_factor)
        list_run_factor = self.face_controller.list_factor.copy()
        list_run_factor[23] = factor
        list_run_factor[25] = factor
        self.scale_factor_dict['A23'].set(factor)
        self.scale_factor_dict['A25'].set(factor)
    
    def func_lower_mouth(self, _factor):
        factor = float(_factor)
        list_run_factor = self.face_controller.list_factor.copy()
        list_run_factor[24] = factor
        list_run_factor[26] = factor
        self.scale_factor_dict['A24'].set(factor)
        self.scale_factor_dict['A26'].set(factor)


    def scale_set_angle(self, _angle, _name, _index):
        self.face_controller.set_servo_angle_time_32(_angle, _index, 200)
        self.scale_factor_dict[_name].set(self.face_controller.angle2factor_from_index(_angle, _index))

    def scale_set_factor(self, _factor, _name, _index):
        self.scale_angle_dict[_name].set(self.face_controller.factor2angle_from_index(_factor, _index))

    def get_yaml_list(self):
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)
        if self.save_path:
            self.yaml_file_list.clear()
            self.listbox.delete(0, tk.END)
            file_list = os.listdir(self.save_path)
            for file in file_list:
                if file.endswith(".yaml"):
                    self.yaml_file_list.append(file)
            for item in self.yaml_file_list:
                self.listbox.insert('end',item) 

    def save_value(self):
        value_list = []
        for servo_name, servo_info in self.face_controller.servo_name2servo_info.items():
            val = self.scale_angle_dict[servo_name].get()
            value_list.append(val)
        save_file_name = self.save_file_entry.get()
        if save_file_name == "":
            print("file name is none, save failed")
            return
        with open(self.save_path + save_file_name + ".yaml", 'w') as file:
            yaml.dump(value_list, file)  # 将数据保存到 YAML 文件
            self.get_yaml_list()
            print("save yaml over...")
    
    
    def reset_button_click(self):
        self.face_controller.set_servo_reset()  # 控制所有舵机复位
        # 更新界面拖动条
        for servo_name, servo_info in self.face_controller.servo_name2servo_info.items():
            self.scale_angle_dict[servo_name].set(servo_info["temp_deg"])
            chan_idx = servo_info["channel_idx"]
            self.scale_factor_dict[servo_name].set(self.face_controller.list_factor[chan_idx])
        self.scale_eyebrow_head_vertical.set(self.face_controller.list_factor[0])
        self.scale_eyebrow_head_horizontal.set(self.face_controller.list_factor[2])
        self.scale_eyebrow_tail.set(self.face_controller.list_factor[4])
        self.scale_upper_eyelid.set(self.face_controller.list_factor[6])
        self.scale_eyeball_vertical.set(self.face_controller.list_factor[8])
        self.scale_eyeball_horizontal.set(self.face_controller.list_factor[10])
        self.scale_cheek.set(self.face_controller.list_factor[12])
        self.scale_upper_lip.set(self.face_controller.list_factor[16])
        self.scale_lower_lip.set(self.face_controller.list_factor[18])
        self.scale_upper_mouth.set(self.face_controller.list_factor[23])
        self.scale_lower_mouth.set(self.face_controller.list_factor[24])


    def delete_button_click(self):
        result = self.listbox.curselection()
        if result == ():
            return
        file_name = str(self.yaml_file_list[result[0]])
        os.remove(self.save_path + file_name)
        self.get_yaml_list()

    def load_button_click(self):
        result = self.listbox.curselection()
        if result == ():
            return
        file_name = str(self.yaml_file_list[result[0]])
        self.load_and_run(file_name)
    
    def load_and_run(self, _file_name):
        load_list_angle = read_yaml(self.save_path + _file_name)
        if load_list_angle == None:
            print("load data is None...")
            return
        for servo_name, servo_info in self.face_controller.servo_name2servo_info.items():
            chan_idx = servo_info["channel_idx"]
            self.scale_angle_dict[servo_name].set(load_list_angle[chan_idx])
            # self.scale_factor_dict[servo_name].set(angle2factor(load_list_angle[chan_idx], 
            #             self.face_controller.list_start_deg[chan_idx], self.face_controller.list_end_deg[chan_idx]))

if __name__ == '__main__' :
    root = tk.Tk()
    face_gui = GUI(root)
    root.mainloop()
