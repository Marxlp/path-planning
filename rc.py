import serial

serial_, baud = '/dev/ttyUSB0', '2000000'
ser = serial.Serial(serial_, str(baud))


def rc(channel_num):
    """
    param: serial_, type=str, e.g '/dev/ttyUSB0'
    param: baud, type=int e.g 9600
    param: channel_num, type=int
    """
    # ser = serial.Serial('/dev/ttyUSB0', '9600')

    rc_channels_all = 5
    data = ser.read(channel_num * 8)
    rc_data_str = data.split()
    rc_data = [[] for i in range(rc_channels_all)]

    q = 0
    for i in rc_data_str:
        list_i = list(i)
        n = len(list_i)
        flag = 0
        rc_data_ = []
        for j in range(n):
            if list_i[j] is not ":" :
                flag = flag
            else:
                flag = flag + 1
                continue
            if flag > 0:
                rc_data_.append(list_i[j])
                rc_data[q] = int(''.join(rc_data_))
        q += 1

    return rc_data


if __name__ == '__main__':
    import time
    for i in range(1000):
        a = rc(5)
        time.sleep(0.01)
        print(a)
    ser.close()
