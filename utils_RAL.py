from enum import Enum

class TaskResult(Enum):
    SUCCESS = 1
    FAILURE = -1
    CRITICAL_BATTERY = -2

class TOD(Enum):
    STARTING = "STARTING"
    POSTER = "POSTER"
    BUFFET = "BUFFET"
    OFF = "OFF"

TODS = {t.value: i for i, t in enumerate(TOD)}

class WP(Enum):
    ROOM1 = "r1"
    ROOM2 = "r2"
    CORRIDOR1 = "c1"
    CORRIDOR2 = "c2"
    CORRIDOR3 = "c3"
    CORRIDOR4 = "c4"
    CORRIDOR5 = "c5"
    CORRIDOR6 = "c6"
    CORRIDOR7 = "c7"
    CORRIDOR8 = "c8"
    CORRIDOR9 = "c9"
    CORRIDOR10 = "c10"
    CORRIDOR11 = "c11"
    CORRIDOR12 = "c12"
    CORRIDOR13 = "c13"
    CORRIDOR14 = "c14"
    CORRIDOR15 = "c15"
    CORRIDOR16 = "c16"
    CORRIDOR17 = "c17"
    CORRIDOR18 = "c18"
    CORRIDOR19 = "c19"
    CORRIDOR20 = "c20"
    CORRIDOR21 = "c21"
    CORRIDOR22 = "c22"
    CORRIDOR23 = "c23"
    CORRIDOR24 = "c24"
    CORRIDOR25 = "c25"
    CORRIDOR26 = "c26"

WPS = {wp.value: i for i, wp in enumerate(WP)}
