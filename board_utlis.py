import re
import subprocess
from enum import Enum


class Board(Enum):
    RASPBERRY_PI_4 = "raspberry_pi_4"
    JETSON_NANO = "jetson_nano"
    JETSON_XAVIER = "jetson_xavier"
    JETSON_XAVIER_NX = "jetson_xavier_nx"
    JETSON_AGX_XAVIER = "jetson_agx_xavier"
    JETSON_TX3 = "jetson_tx3"
    JETSON_TX2 = "jetson_tx2"
    JETSON_TX1 = "jetson_tx1"


def check_RPI_CPU_temp():
    temp = None
    err, msg = subprocess.getstatusoutput('vcgencmd measure_temp')
    if not err:
        m = re.search(r'-?\d\.?\d*', msg)   # a solution with a  regex
        try:
            temp = float(m.group())
        except:
            pass
    return temp, msg
