from enum import Enum


class TOD(Enum):
    S1 = "S1"
    S2 = "S2"

TODS = {t.value: i for i, t in enumerate(TOD)}

class WP(Enum):
    INB3235 = "inb3235"
    INB3241 = "inb3241"
    POSTER1 = "poster1"
    POSTER2 = "poster2"
    POSTER3 = "poster3"
    POSTER4 = "poster4"
    poster5 = "poster5"
    INB3238 = "inb3238"
    INB3237 = "inb3237"
    L = "L"
    TOILET = "toilet"
    A = "A"
    B = "B"
    AB = "AB"
    C = "C"
    BC = "BC"
    D = "D"
    E = "E"
    F = "F"
    G = "G"
    H = "H"
    I = "I"
    M = "M"
    LM = "LM"
    N = "N"
    O = "O"
    R = "R"
    CR = "CR"
    S = "S"


WPS = {wp.value: i for i, wp in enumerate(WP)}