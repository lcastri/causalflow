from enum import Enum

class TaskResult(Enum):
    SUCCESS = 1
    FAILURE = -1

class Task(Enum):
    DELIVERY = "delivery"
    CLEANING = "cleaning"
    CHARGING = "charging"

class TOD(Enum):
    T1 = "T1"
    T2 = "T2"
    T3 = "T3"
    T4 = "T4"
    T5 = "T5"
    T6 = "T6"
    T7 = "T7"
    T8 = "T8"
    T9 = "T9"
    T10 = "T10"
    T11 = "T11"
    T12 = "T12"
    T13 = "T13"
    T14 = "T14"
    T15 = "T15"
    T16 = "T16"
    T17 = "T17"
    T18 = "T18"
    T19 = "T19"
    T20 = "T20"
    OFF = "off"

TODS = {t.value: i for i, t in enumerate(TOD)}

class WP(Enum):
    PARKING = "parking"            
    DOOR_ENTRANCE = "door-entrance"
    ENTRANCE = "entrance"
    DOOR_CANTEEN = "door-canteen"
    CORRIDOR_0 = "corridor-0" 
    CORRIDOR_1= "corridor-1"       
    CORRIDOR_2_R= "corridor-2-r"
    CORRIDOR_3_R= "corridor-3-r"
    CORRIDOR_4_R= "corridor-4-r"
    CORRIDOR_5_R= "corridor-5-r"   
    CORRIDOR_6_R= "corridor-6-r"   
    CORRIDOR_7_R= "corridor-7-r"   
    CORRIDOR_2_C= "corridor-2-c"
    CORRIDOR_3_C= "corridor-3-c"
    CORRIDOR_4_C= "corridor-4-c"
    CORRIDOR_5_C= "corridor-5-c"   
    CORRIDOR_6_C= "corridor-6-c"
    CORRIDOR_7_C= "corridor-7-c"
    CORRIDOR_3_L= "corridor-3-l"
    CORRIDOR_4_L= "corridor-4-l"
    CORRIDOR_5_L= "corridor-5-l"   
    SUPPORT_1= "support-1"   
    SUPPORT_2= "support-2"   
    SUPPORT_3= "support-3"   
    SUPPORT_4= "support-4"   
    DOOR_OFFICE_1= "door-office-1" 
    DOOR_OFFICE_2= "door-office-2" 
    DOOR_TOILET_1= "door-toilet-1" 
    DOOR_TOILET_2= "door-toilet-2"
    OFFICE_1= "office-1"
    OFFICE_2= "office-2"
    TOILET_1= "toilet-1"
    SUPPORT_5= "support-5"   
    SUPPORT_6= "support-6"   
    SUPPORT_7= "support-7"   
    SUPPORT_8= "support-8"   
    TOILET_2= "toilet-2"
    DOOR_CORRIDOR_1= "door-corridor-1"
    DOOR_CORRIDOR_2= "door-corridor-2"
    DOOR_CORRIDOR_3= "door-corridor-3"
    TABLE_12= "table-12"           
    TABLE_23= "table-23"           
    SUPPORT_0= "support-0"           
    SUPPORT_9= "support-9"           
    CORRIDOR_CANTEEN_1= "corridor-canteen-1"
    CORRIDOR_CANTEEN_2= "corridor-canteen-2"
    CORRIDOR_CANTEEN_3= "corridor-canteen-3"
    CORRIDOR_CANTEEN_4= "corridor-canteen-4"
    WA_1_R= "wa-1-r"
    WA_2_R= "wa-2-r"
    WA_3_R= "wa-3-r"
    WA_3_CR= "wa-3-cr"
    WA_4_R= "wa-4-r"
    WA_5_R= "wa-5-r"
    WA_1_C= "wa-1-c"
    WA_2_C= "wa-2-c"
    WA_3_C= "wa-3-c"
    WA_4_C= "wa-4-c"
    WA_5_C= "wa-5-c"
    WA_1_L= "wa-1-l"
    WA_2_L= "wa-2-l"
    WA_3_L= "wa-3-l"
    WA_3_CL= "wa-3-cl"
    WA_4_L= "wa-4-l"
    WA_5_L= "wa-5-l"
    TARGET_1= "target-1"
    TARGET_2= "target-2"
    TARGET_3= "target-3"
    TARGET_4= "target-4"
    TARGET_5= "target-5"
    TARGET_6= "target-6"
    TARGET_7= "target-7"
    CHARGING_STATION= "charging-station"

WPS = {wp.value: i for i, wp in enumerate(WP)}

class NODES(Enum):
  TOD = 'TOD'
  RV = 'R_V'
  RB = 'R_B'
  BS = 'B_S'
  PD = 'PD'
  BAC = 'BAC'
  WP = 'WP'