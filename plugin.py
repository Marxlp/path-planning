import rospy
import geometry_msgs.msg
from threading import Thread
from mavros_msgs.msg import HilSensor
from mavros_msgs.msg import HilGPS
from mavros_msgs.msg import HilStateQuaternion
from mavros_msgs.msg import HilActuatorControls
from mavros_msgs.msg import RCIn
from mavros_msgs.msg import HomePosition
from std_msgs.msg import Float32
import math
import numpy as np
import time
# import rc
# import serial


lla = np.zeros(3)
gyro_data = np.zeros(3)
acc_data = np.zeros(3)
mag_data = np.zeros(3)
v_ned = np.zeros(3)
quaternion_data = np.zeros(4)
local_position = np.zeros(3)
angular_velocity = np.zeros(3)
actuators_controls = np.zeros(4)
sim_time = 0.0
v_loc = np.zeros(3)
mode = 0


def plugin():
    rospy.init_node('plugin', anonymous=True)
    rospy.Subscriber('/gps_data', geometry_msgs.msg.Vector3, callback_gps)
    rospy.Subscriber('/gyro_data', geometry_msgs.msg.Vector3, callback_gyro)
    rospy.Subscriber("/acc_data", geometry_msgs.msg.Vector3, callback_acc)
    rospy.Subscriber("/sim_ros_interface/mag_data", geometry_msgs.msg.Vector3, callback_mag)
    rospy.Subscriber("/sim_ros_interface/local_velocity", geometry_msgs.msg.Vector3, callback_vel_loc)
    rospy.Subscriber("/vehicleQuaternion", geometry_msgs.msg.Quaternion, callback_quaternion)
    rospy.Subscriber("/sim_ros_interface/angular_velocity", geometry_msgs.msg.Vector3, callback_vel_a)
    rospy.Subscriber("/simulationTime", Float32, callback_time)
    rospy.Subscriber("/vehiclePosition", geometry_msgs.msg.Point, callback_loc_pos)
    rospy.Subscriber("/sim_ros_interface/velocity_ned", geometry_msgs.msg.Vector3, callback_vel_ned)
    rospy.Subscriber("/mavros/hil/actuator_controls", HilActuatorControls, callback_control)

    pub_gps = rospy.Publisher('/mavros/hil/gps', HilGPS, queue_size=1)
    pub_imu = rospy.Publisher('/mavros/hil/imu_ned', HilSensor, queue_size=1)
    pub_state_quaternion = rospy.Publisher('/mavros/hil/state', HilStateQuaternion, queue_size=1)

    pub_thread_1 = Thread(target=gps_publisher, name='gps_publisher', args=(pub_gps,))
    pub_thread_1.start()

    pub_thread_2 = Thread(target=imu_publisher, name='imu_publisher', args=(pub_imu,))
    pub_thread_2.start()

    pub_thread_3 = Thread(target=state_quaternion_publisher, name='state_quaternion_publisher', args=(pub_state_quaternion,))
    pub_thread_3.start()

    # publish control signal

    pub_control_1 = rospy.Publisher('px4_control_1', Float32, queue_size=1)
    pub_control_2 = rospy.Publisher('px4_control_2', Float32, queue_size=1)
    pub_control_3 = rospy.Publisher('px4_control_3', Float32, queue_size=1)
    pub_control_4 = rospy.Publisher('px4_control_4', Float32, queue_size=1)

    pub_thread_4 = Thread(target=control_publisher, name='control_1_publisher',
                          args=(pub_control_1, pub_control_2, pub_control_3, pub_control_4,))
    pub_thread_4.start()

    pub_home = rospy.Publisher("/mavros/home_position/set", HomePosition, queue_size=1)
    pub_thread_5 = Thread(target=home_publisher, name='home_publisher', args=(pub_home, ))
    pub_thread_5.start()

    rospy.spin()


def callback_gps(data):
    global lla
    """latitude and longitude in degree, altitude in m"""
    lla[0] = data.x
    lla[1] = data.y
    lla[2] = data.z


def callback_gyro(data):

    global gyro_data
    gyro_data[0] = data.x
    gyro_data[1] = data.y
    gyro_data[2] = data.z


def callback_acc(data):
    """Acceleration unit: mG, type: init16 in HilStateQuaternion, but the unit: m/s/s,
    type: float in HilSensor
      """
    global acc_data
    acc_data[0] = data.x
    acc_data[1] = data.y
    acc_data[2] = data.z


def callback_mag(data):
    global mag_data
    """magnetic data in Tesla"""
    mag_data[0] = data.x * 10**(-4)
    mag_data[1] = data.y * 10**(-4)
    mag_data[2] = data.z * 10**(-4)
    # rospy.loginfo(mag_data)


def callback_vel_loc(data):
    """velocity in aircraft frame m/s/s"""
    global v_loc
    v_loc[0] = data.x
    v_loc[1] = data.y
    v_loc[2] = data.z


def callback_vel_ned(data):
    """velocity in NED frame m/s/s"""
    global v_ned
    v_ned[0] = data.x
    v_ned[1] = data.y
    v_ned[2] = data.z


def callback_quaternion(data):
    global quaternion_data
    quaternion_data[0] = data.x
    quaternion_data[1] = data.y
    quaternion_data[2] = data.z
    quaternion_data[3] = data.w


def callback_vel_a(data):
    global angular_velocity
    angular_velocity[0] = data.x
    angular_velocity[1] = data.y
    angular_velocity[2] = data.z


def callback_time(data):
    global sim_time
    sim_time = data.data


def callback_loc_pos(data):
    global local_position
    local_position[0] = data.x
    local_position[1] = data.y
    local_position[2] = data.z


def gps_publisher(pub):
    global lla, v_ned, sim_time, v_loc
    gps_data = HilGPS()
    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        # gps_data.header.seq = int(1)
        t = math.modf(time.time())
        gps_data.header.stamp.secs = int(t[1])
        gps_data.header.stamp.nsecs = int(t[0] * 10**9)
        gps_data.header.frame_id = "NED"
        """the all frame_id can be found in https://mavlink.io/en/messages/common.html#MAV_FRAME_LOCAL_FRD"""
        gps_data.fix_type = 4

        gps_data.geo.latitude = lla[0]
        gps_data.geo.longitude = lla[1]
        gps_data.geo.altitude = -lla[2]

        gps_data.ve = v_ned[0]
        gps_data.vn = v_ned[1]
        gps_data.vd = v_ned[2]

        gps_data.ve = v_loc[0]
        gps_data.eph = 0.8
        gps_data.epv = 1.5
        """eph and epv means horizontal position error and vertical position error,unit is  (m)"""
        gps_data.cog = 65436
        gps_data.satellites_visible = 10

        pub.publish(gps_data)
        rate.sleep()


def barometric(z, v):
    """calculate barometric pressure
    param: z local position z up+, down-
    param: v local velocity"""
    alt_home = 0.0

    lapse_rate = 0.0065
    """"(kelvin/m)"""
    temperature_msl = 288.0
    """mean sea level temperature(kelvin)"""
    alt_msl = alt_home + z
    temperature_local = temperature_msl - lapse_rate * alt_msl
    pressure_ratio = (temperature_msl / temperature_local) ** 5.256
    pressure_msl = 101325.0
    """pressure at mean sea level (Pa)"""
    absolute_pressure = pressure_msl / pressure_ratio + np.random.normal(0, 1) * 5

    density_ratio = (temperature_msl / temperature_local) ** 4.256
    rho = 1.225 / density_ratio

    """calculate temperature in Celsius"""
    baro_temperature = temperature_local - 273.0 + np.random.normal(0, 1) * 0.1
    """calculate differential pressure (Pa)"""
    diff_pressure = 0.5 * rho * v**2 + np.random.normal(0, 1) * 1
    """calculate pressure altitude (m)"""
    pressure_alt = 44307.7 * (1 - (absolute_pressure/100/1013.25) ** 0.190284) + np.random.normal(0, 1) * 0.1

    return baro_temperature, absolute_pressure, diff_pressure, pressure_alt


def imu_publisher(pub):
    """the sensors data is in baselink frame(Forward, Left, Up),
    MAVROS node transfroms them to aircraft frame (Forward, Right, Down). Since the imu frame in coppeliaSim is aircraft
    frame, in this part, it needs to transform to FLU frame, thus x = data.x, y = data.y, z = data.z """
    global gyro_data, acc_data, mag_data, sim_time, local_position, v_loc
    imu_data = HilSensor()

    rate = rospy.Rate(500)
    while not rospy.is_shutdown():
        imu_data.gyro.x = gyro_data[0]
        imu_data.gyro.y = gyro_data[1]
        imu_data.gyro.z = gyro_data[2]

        imu_data.acc.x = acc_data[0]
        imu_data.acc.y = acc_data[1]
        imu_data.acc.z = acc_data[2]

        imu_data.mag.x = mag_data[0]
        imu_data.mag.y = mag_data[1]
        imu_data.mag.z = mag_data[2]

        barometric_sensor = barometric(-local_position[2], v_loc[2])
        imu_data.temperature = barometric_sensor[0]
        imu_data.abs_pressure = barometric_sensor[1]
        imu_data.diff_pressure = barometric_sensor[2]
        """the pressure unit is Pascal"""
        imu_data.pressure_alt = barometric_sensor[3]

        imu_data.header.seq = int(1)
        t = math.modf(time.time())
        imu_data.header.stamp.secs = int(t[1])
        imu_data.header.stamp.nsecs = int(t[0] * 10 ** 9)

        imu_data.header.frame_id = 'FRD'
        imu_data.fields_updated = int(2 ** 18-1)
        # imu_data.fields_updated = 0
        pub.publish(imu_data)
        rate.sleep()


def state_quaternion_publisher(pub):
    global angular_velocity, lla, v_ned, acc_data, quaternion_data, v_loc
    state_quaternion = HilStateQuaternion()
    rate = rospy.Rate(500)
    while not rospy.is_shutdown():
        state_quaternion.angular_velocity.x = angular_velocity[0]
        state_quaternion.angular_velocity.y = angular_velocity[1]
        state_quaternion.angular_velocity.z = angular_velocity[2]

        state_quaternion.geo.latitude = lla[0]
        state_quaternion.geo.longitude = lla[1]
        state_quaternion.geo.altitude = -lla[2]

        state_quaternion.orientation.w = quaternion_data[3]
        state_quaternion.orientation.x = quaternion_data[0]
        state_quaternion.orientation.y = quaternion_data[1]
        state_quaternion.orientation.z = quaternion_data[2]

        state_quaternion.linear_acceleration.x = acc_data[0]
        state_quaternion.linear_acceleration.y = acc_data[1]
        state_quaternion.linear_acceleration.z = acc_data[2]

        state_quaternion.linear_velocity.x = v_loc[0]
        state_quaternion.linear_velocity.y = v_loc[1]
        state_quaternion.linear_velocity.z = v_loc[2]

        state_quaternion.ind_airspeed = v_loc[0]
        state_quaternion.true_airspeed = v_loc[0]

        t = math.modf(time.time())
        state_quaternion.header.stamp.secs = int(t[1])
        state_quaternion.header.stamp.nsecs = int(t[0] * 10**9)
        state_quaternion.header.seq = int(1)
        pub.publish(state_quaternion)
        rate.sleep()


def callback_control(data):
    global actuators_controls, mode
    j = 0
    mode = data.mode

    for i in data.controls:
        if i < 0:
            i = 0
        actuators_controls[j] = i
        j += 1
        if j > 3:
            break
    rospy.loginfo(actuators_controls)



def control_publisher(pub1, pub2, pub3, pub4):
    global actuators_controls, mode
    rate = rospy.Rate(300)
    while not rospy.is_shutdown():
        if mode == 129:
            pub1.publish(actuators_controls[0])
            pub2.publish(actuators_controls[1])
            pub3.publish(actuators_controls[2])
            pub4.publish(actuators_controls[3])
            rate.sleep()


def rc_publisher(pub):
    # publish rc signal
    rc_data = RCIn()
    rc_input = np.zeros(12)
    rate = rospy.Rate(3000)

    while not rospy.is_shutdown():
        rc_input[0:5] = rc.rc(5)

        t = math.modf(time.time())
        rc_data.header.stamp.secs = int(t[1])
        rc_data.header.stamp.nsecs = int(t[0] * 10**9)

        rc_data.channels = rc_input
        rc_data.rssi = 128
        rospy.loginfo(rc_data)
        pub.publish(rc_data)
        rate.sleep()


def home_publisher(pub):
    data = HomePosition()
    rate = rospy.Rate(100)

    while not rospy.is_shutdown():
        t = math.modf(time.time())
        data.header.stamp.secs = int(t[1])
        data.header.stamp.nsecs = int(t[0] * 10**9)

        data.geo.latitude = 39.455
        data.geo.longitude = 116.245
        data.geo.altitude = 0

        data.position.x = 0
        data.position.y = 0
        data.position.z = 0

        data.orientation.x = 0
        data.orientation.y = 0
        data.orientation.z = 0
        data.orientation.w = 1

        data.approach.x = 0
        data.approach.y = 0
        data.approach.z = 0

        pub.publish(data)
        rate.sleep()


if __name__ == '__main__':
    # serial_, baud = '/dev/ttyUSB0', '2000000'
    # ser = serial.Serial(serial_, str(baud))
    plugin()