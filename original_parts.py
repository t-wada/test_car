import os
import time
import datetime
# import scipy
import cv2    as cv
import numpy  as np
import pandas as pd

from   imu       import SL_MPU9250
import donkeycar as     dk

#import parts
from donkeycar.parts.throttle_filter import ThrottleFilter
from donkeycar.parts.camera          import BaseCamera, Webcam
from donkeycar.parts.controller      import get_js_controller, PS3JoystickController
from donkeycar.parts.actuator        import PCA9685, PWMSteering, PWMThrottle
from donkeycar.utils                 import *

import Jetson.GPIO as GPIO


class CustomPiCamera(BaseCamera):
      def __init__(self, image_w     = 256,
                         image_h     = 256,
                         rgb_or_gray = True,
                         framerate   = 20,
                         image_dir   = "image"):
                         
            self.rgb_or_gray = rgb_or_gray
            self.framerate   = framerate

            # 画像サイズが小さすぎるとself.frameにNoneが返る
            GST_STR = 'nvarguscamerasrc \
                       ! video/x-raw(memory:NVMM), width=3280, height=2464, format=(string)NV12, framerate=(fraction)30/1 \
                       ! nvvidconv ! video/x-raw, width=(int){}, height=(int){}, format=(string)BGRx \
                       ! videoconvert \
                       ! appsink'.format(image_w, image_h)
            self.cap = cv.VideoCapture(GST_STR, cv.CAP_GSTREAMER)

            cnt      = 0
            while cnt < 20:
                  _, frame = self.cap.read()
                  if frame is not None:
                        cnt += 1
            self.frame = frame

            # initialize the frame and the variable used to indicate
            # if the thread should be stopped
            self.on = True
            
            self.image_num = 0
            self.image_dir = os.path.join(image_dir, datetime.datetime.today().strftime("%Y%m%d-%H%M%S"))
            os.makedirs(self.image_dir, exist_ok = True)

            print('PiCamera loaded... warming camera')
            time.sleep(2)

      def run_threaded(self):
            image_path = os.path.join(self.image_dir, "{:08}.jpg".format(self.image_num))
            cv.imwrite(image_path, self.frame)

            self.image_num += 1

            return self.frame

      def update(self):
            while self.on:
                  start_time = time.time()

                  _, self.frame = self.cap.read()
                  if not self.rgb_or_gray:
                        self.frame = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)

                  elapsed_time_sec = time.time() - start_time
                  sleep_time_sec   = 1 / self.framerate - elapsed_time_sec
                  if sleep_time_sec > 0.0:
                        time.sleep(sleep_time_sec)
                  
      def shutdown(self):
            # indicate that the thread should be stopped
            self.on = False

            print('Stopping PiCamera')
            time.sleep(0.5)

class IPM:
      def __init__(self, uv, 
                         points, 
                         scale = 50, 
                         xlim  = (0, 5), 
                         ylim  = (0, 5)):
            # 変換後の画像が大きくなりすぎなように値を補正する
            # points [m]：unitの値からスケールを補正する
            uv     = np.array(uv,     dtype = np.float32)
            points = np.array(points, dtype = np.float32)
            
            self.xlim   = xlim
            self.ylim   = ylim
            self.scale  = scale
            self.points = self._preprocess(points)

            self.M = cv.getPerspectiveTransform(uv, self.points)

      def _preprocess(self, points):
            points[:, 0] = points[:, 0] + self.xlim[1] / 2
            points[:, 1] = self.ylim[1] - points[:, 1]
            points      *= self.scale

            return points

      def _postprocess(self, points):
            points      /= self.scale
            points[:, 1] = self.ylim[1] - points[:, 1]
            points[:, 0] = points[:, 0] - self.xlim[1] / 2

            return points

      def run(self, img):
            return cv.warpPerspective(img,
                                      self.M, 
                                      (self.xlim[1] * self.scale, self.ylim[1] * self.scale))

      def transform(self, P):
            P = np.array(P)
            P = np.insert(P.T, 2, 1, axis = 0)

            dst = np.dot(self.M, P)
            ret = dst / dst[2, :]
            ret = ret[:-1].T
            
            return ret

class PS3JoystickController_trim(PS3JoystickController):
      def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            self.angle_offset = 0

      def adjust_right_angle(self):
            print("angle offset : {}".format(self.angle_offset))
            self.angle_offset -= 1
      
      def adjust_left_angle(self):
            print("angle offset : {}".format(self.angle_offset))
            self.angle_offset += 1

      def init_trigger_maps(self):
            '''
            init set of mapping from buttons to function calls
            '''
            self.button_down_trigger_map = {
                  'select'    : self.toggle_mode,
                  'circle'    : self.toggle_manual_recording,
                  'triangle'  : self.erase_last_N_records,
                  'cross'     : self.emergency_stop,
                  'dpad_up'   : self.increase_max_throttle,
                  'dpad_down' : self.decrease_max_throttle,
                  'start'     : self.toggle_constant_throttle,
                  "R1"        : self.adjust_right_angle,
                  "L1"        : self.adjust_left_angle,
            }

            #   self.button_up_trigger_map = {
            #       "R1" : self.chaos_monkey_off,
            #       "L1" : self.chaos_monkey_off,
            #   }

            self.axis_trigger_map = {
                  'left_stick_horz'  : self.set_steering,
                  'right_stick_vert' : self.set_throttle,
            }

      def run_threaded(self):
            '''
            process E-Stop state machine
            '''
            if self.estop_state > self.ES_IDLE:
                  if self.estop_state == self.ES_START:
                        self.estop_state = self.ES_THROTTLE_NEG_ONE
                        # print("angle : {}, throttle : {}".format(0.0, -1.0 * self.throttle_scale))

                        return 0.0, -1.0 * self.throttle_scale, self.angle_offset

                  elif self.estop_state == self.ES_THROTTLE_NEG_ONE:
                        self.estop_state = self.ES_THROTTLE_POS_ONE
                        # print("angle : {}, throttle : {}".format(0.0, 0.01))

                        return 0.0, 0.01, self.angle_offset

                  elif self.estop_state == self.ES_THROTTLE_POS_ONE:
                        self.estop_state = self.ES_THROTTLE_NEG_TWO
                        self.throttle    = -1.0 * self.throttle_scale
                        # print("angle : {}, throttle : {}".format(0.0, self.throttle))

                        return 0.0, self.throttle, self.angle_offset

                  elif self.estop_state == self.ES_THROTTLE_NEG_TWO:
                        self.throttle += 0.05
                        if self.throttle >= 0.0:
                              self.throttle    = 0.0
                              self.estop_state = self.ES_IDLE

                        # print("angle : {}, throttle : {}".format(0.0, self.throttle))

                        return 0.0, self.throttle, self.angle_offset

            # print("angle : {}, throttle : {}".format(self.angle, self.throttle))
            return self.angle, self.throttle, self.angle_offset



class PWMSteering_trim(PWMSteering):

      def __init__(self, controller  = None,
                         left_pulse  = 290,
                         right_pulse = 490):

            super().__init__(controller,
                             left_pulse,
                             right_pulse)

      def run(self, angle, angle_offset):
            #map absolute angle to angle that vehicle can implement.
            pulse = dk.utils.map_range(angle,
                                       self.LEFT_ANGLE,
                                       self.RIGHT_ANGLE,
                                       self.right_pulse,
                                       self.left_pulse)
            pulse += angle_offset
            pulse  = np.clip(pulse, self.left_pulse, self.right_pulse)
            
            self.controller.set_pulse(pulse)



class Speed_Logger:
      def __init__(self, framerate      = 20,
                         low_pass_coeff = 0.9):
            # initialize IMU
            self.imu = SL_MPU9250(0x68, 1)
            self.imu.resetRegister()
            self.imu.powerWakeUp()
            self.imu.setAccelRange(8, True)
            self.imu.setGyroRange(1000, True)
            self.imu.setMagRegister('100Hz', '16bit')
            # self.imu.selfTestMag()

            # low pass coefficient
            self.low_pass_coeff = low_pass_coeff
            # timespan per frame
            self.timespan_sec = 1.0 / framerate

            self.initialize_vars()

            # スピードの初期化
            # はじめの数秒間は誤差の蓄積が大きい
            print("Calibrating Speed Logger ...")
            elapsed_time_sec = 0.0
            while elapsed_time_sec < 5.0:
                  elapsed_time_sec += self.accel2speed()
            # self.speed_m_per_s = np.array([0.0] * 3)
            # self.position_change_m  = np.array([0.0] * 3)
            self.initialize_vars()

            self.speed_list = []

            self.on = True

      def initialize_vars(self):
            self.last_low_pass_accel_g_per_s2  = np.array([0.0] * 3)
            self.last_high_pass_accel_m_per_s2 = np.array([0.0] * 3)
            # self.last_speed_m_per_s            = np.array([0.0] * 3)
            self.speed_m_per_s                 = np.array([0.0] * 3)
            # self.position_change_m             = np.array([0.0] * 3)

      def run_threaded(self):
            print(self.speed_m_per_s)
            self.speed_list.append(self.speed_m_per_s.tolist())

            # return self.speed_m_per_s, self.position_change_m
            return self.speed_m_per_s

      def update(self):
            while self.on:
                  elapsed_time_sec = self.accel2speed()
                  sleep_time_sec   = self.timespan_sec - elapsed_time_sec
                  if sleep_time_sec > 0.0:
                        time.sleep(sleep_time_sec)

      # 加速度を速度に、速度を変位に変換
      def accel2speed(self):
            start_time = time.time()

            row_accel_g_per_s2 = np.array(self.imu.getAccel())
      
            # low pass filter
            low_pass_accel_g_per_s2 = self.low_pass_coeff         * self.last_low_pass_accel_g_per_s2 + \
                                      (1.0 - self.low_pass_coeff) * row_accel_g_per_s2
            # high pass filter
            high_pass_accel_g_per_s2 = row_accel_g_per_s2 - low_pass_accel_g_per_s2
            # 単位変換
            # g → m
            # high_pass_accel_m_per_s2 = high_pass_accel_g_per_s2 * scipy.constants.g
            high_pass_accel_m_per_s2 = high_pass_accel_g_per_s2 * 9.8

            # 加速度を(台形)積分して速度に変換
            self.speed_m_per_s     += (self.last_high_pass_accel_m_per_s2 + high_pass_accel_m_per_s2) * self.timespan_sec * 0.5
            # print(self.speed_m_per_s)
            # 速度を(台形)積分して変位量に変換
            # self.position_change_m += (self.last_speed_m_per_s            + self.speed_m_per_s)       * self.timespan_sec * 0.5
            # print(self.position_change_m)

            self.last_low_pass_accel_g_per_s2  = low_pass_accel_g_per_s2
            self.last_high_pass_accel_m_per_s2 = high_pass_accel_m_per_s2
            # self.last_speed_m_per_s            = self.speed_m_per_s

            elapsed_time_sec = time.time() - start_time

            return elapsed_time_sec

      def shutdown(self):
            # indicate that the thread should be stopped
            self.on = False

            num      = 0
            csv_path = "{}.csv".format(num)
            while os.path.isfile(csv_path):
                  num     += 1
                  csv_path = "{}.csv".format(num)
            else:
                  np.savetxt(csv_path, self.speed_list)

            print('Stopping Speed Logger')
            time.sleep(0.5)

class CSV_Logger():
      def __init__(self, csv_path = "log.csv"):
            self.csv_path = csv_path
            self.vals     = []

      def run(self, image_name, angle, throttle):
            self.vals.append([image_name, angle, throttle])

      def shutdown(self):
            col_names = ["image_name", "angle", "throttle"]
            dataframe = pd.DataFrame(self.vals, columns = col_names)
            dataframe.to_csv(self.csv_path)


class WheelEncoder():
      def __init__(self, in_pin=13, wheel_diameter_mm=82, hz=100):
            self._wheel_diameter_mm = wheel_diameter_mm
            self._hz = hz
            self._in_pin = in_pin

            self._on = True
            self._pre_input = 0
            self._sign = 1.0
            self._vals = []
            GPIO.setmode(GPIO.BOARD)
            GPIO.setup(self._in_pin, GPIO.IN)

      def update(self):
            sleep_time = 1.0 / self._hz
            cnt = 0
            while self._on:
                  cur_input = GPIO.input(self._in_pin)
                  if cur_input != self._pre_input and cur_input == GPIO.HIGH:
                        self._vals.append(self._sign)
                  else:
                        self._vals.append(0.0)
                  
                  self._vals = self._vals[self._hz]
                  self._pre_input = cur_input
                  time.sleep(sleep_time)
                  cnt += 1
                  if cnt % self.hz == 0:
                        print("update")

      def run_threaded(self, throttle):
            self._sign = np.sign(throttle)
            print("rps", sum(self._vals))
            return sum(self._vals) * self._wheel_diameter_mm


      def shutdown(self):
            self._on = False
            GPIO.cleanup()

if __name__ == "__main__":
      cfg = dk.load_config()

      threaded = True

      #Initialize car
      V = dk.vehicle.Vehicle()

      # camera
      cam = CustomPiCamera(image_w     = cfg.IMAGE_W,
                           image_h     = cfg.IMAGE_H,
                           rgb_or_gray = True,
                           framerate   = 20)
      V.add(cam, 
            outputs  = ['cam/image_array'],
            threaded = threaded)

      # speed_logger = Speed_Logger()
      # V.add(speed_logger, 
      #       # outputs  = ['speed', 'position_change'],
      #       outputs  = ['speed'],
      #       threaded = threaded)

      # joystick controller
      ctr = PS3JoystickController_trim(throttle_dir            = cfg.JOYSTICK_THROTTLE_DIR,
                                       throttle_scale          = cfg.JOYSTICK_MAX_THROTTLE,
                                       steering_scale          = cfg.JOYSTICK_STEERING_SCALE,
                                       auto_record_on_throttle = cfg.AUTO_RECORD_ON_THROTTLE)
      ctr.set_deadzone(cfg.JOYSTICK_DEADZONE)
      V.add(ctr,
            outputs  = ['angle', 'throttle', 'angle_offset'],
            threaded = True)

      # throttle filter
      th_filter = ThrottleFilter()
      V.add(th_filter, 
            inputs  = ['throttle'],
            outputs = ['throttle'])

      # steering
      # サーボに指示値を与えるクラス
      steering_controller = PCA9685(cfg.STEERING_CHANNEL, cfg.PCA9685_I2C_ADDR, busnum=cfg.PCA9685_I2C_BUSNUM)
      steering            = PWMSteering_trim(controller  = steering_controller,
                                             left_pulse  = cfg.STEERING_LEFT_PWM, 
                                             right_pulse = cfg.STEERING_RIGHT_PWM)
      V.add(steering, 
            inputs = ['angle', 'angle_offset'])
      
      # throttle
      # サーボに指示値を与えるクラス
      throttle_controller = PCA9685(cfg.THROTTLE_CHANNEL, cfg.PCA9685_I2C_ADDR, busnum=cfg.PCA9685_I2C_BUSNUM)
      throttle            = PWMThrottle(controller = throttle_controller,
                                        max_pulse  = cfg.THROTTLE_FORWARD_PWM,
                                        zero_pulse = cfg.THROTTLE_STOPPED_PWM, 
                                        min_pulse  = cfg.THROTTLE_REVERSE_PWM)
      V.add(throttle, 
            inputs = ['throttle'])

      wheel_encoder = WheelEncoder()
      V.add(wheel_encoder,
            inputs = ['throttle'],
            outputs = ['speed_mmps'], # output speed mm/s.
            threaded = True)

      # V.add(CSV_Logger(), inputs = ['image_name', 'angle', 'throttle'])

      #run the vehicle for 20 seconds
      V.start(rate_hz        = cfg.DRIVE_LOOP_HZ,
              max_loop_count = cfg.MAX_LOOPS)