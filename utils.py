# from enum import Enum


# class TOD(Enum):
#     STARTING = "starting"
#     MORNING = "morning"
#     LUNCH = "lunch"
#     AFTERNOON = "afternoon"
#     QUITTING = "quitting"
#     OFF = "off"


# class WP(Enum):
#   PARKING = "parking"
#   DOOR_ENTRANCE = "door-entrance"
#   DOOR_ENTRANCE_CANTEEN = "door-entrance-canteen"
#   CORRIDOR_ENTRANCE = "corridor-entrance"
#   DOOR_CORRIDOR1 = "door-corridor1"
#   DOOR_CORRIDOR2 = "door-corridor2"
#   DOOR_CORRIDOR3 = "door-corridor3"
#   SHELF12 = "shelf12"
#   SHELF23 = "shelf23"
#   SHELF34 = "shelf34"
#   SHELF45 = "shelf45"
#   SHELF56 = "shelf56"
#   DOOR_OFFICE1 = "door-office1"
#   DOOR_OFFICE2 = "door-office2"
#   DOOR_TOILET1 = "door-toilet1"
#   DOOR_TOILET2 = "door-toilet2"
#   DELIVERY_POINT = "delivery-point"
#   CORRIDOR0 = "corridor0"
#   CORRIDOR1 = "corridor1"
#   CORRIDOR2 = "corridor2"
#   CORRIDOR3 = "corridor3"
#   CORRIDOR4 = "corridor4"
#   CORRIDOR5 = "corridor5"
#   ENTRANCE = "entrance"
#   OFFICE1 = "office1"
#   OFFICE2 = "office2"
#   TOILET1 = "toilet1"
#   TOILET2 = "toilet2"
#   TABLE2 = "table2"
#   TABLE3 = "table3"
#   TABLE4 = "table4"
#   TABLE5 = "table5"
#   TABLE6 = "table6"
#   CORR_CANTEEN_1 = "corr-canteen-1"
#   CORR_CANTEEN_2 = "corr-canteen-2"
#   CORR_CANTEEN_3 = "corr-canteen-3"
#   CORR_CANTEEN_4 = "corr-canteen-4"
#   CORR_CANTEEN_5 = "corr-canteen-5"
#   CORR_CANTEEN_6 = "corr-canteen-6"
#   KITCHEN_1 = "kitchen1"
#   KITCHEN_2 = "kitchen2"
#   KITCHEN_3 = "kitchen3"
#   CORRIDOR_CANTEEN = "corridor-canteen"
#   SHELF1 = "shelf1"
#   SHELF2 = "shelf2"
#   SHELF3 = "shelf3"
#   SHELF4 = "shelf4"
#   SHELF5 = "shelf5"
#   SHELF6 = "shelf6"
#   CHARGING_STATION = "charging-station"
  
  
# class NODES(Enum):
#   TOD = 'TOD'
#   # A = 'A'
#   # T = 'T'
#   RV = 'R_V'
#   RB = 'R_B'
#   BS = 'B_S'
#   PD = 'PD'
#   BAC = 'BAC'
#   WP = 'WP'


# TODS = {
#     TOD.STARTING.value: 0,
#     TOD.MORNING.value: 1,
#     TOD.LUNCH.value: 2,
#     TOD.AFTERNOON.value: 3,
#     TOD.QUITTING.value: 4,
#     TOD.OFF.value: 5
# }

# WPS = {
#     WP.PARKING.value: 0,
#     WP.DOOR_ENTRANCE.value: 1,
#     WP.DOOR_ENTRANCE_CANTEEN.value: 2,
#     WP.CORRIDOR_ENTRANCE.value: 3,
#     WP.DOOR_CORRIDOR1.value: 4,
#     WP.DOOR_CORRIDOR2.value: 5,
#     WP.DOOR_CORRIDOR3.value: 6,
#     WP.SHELF12.value: 7,
#     WP.SHELF23.value: 8,
#     WP.SHELF34.value: 9,
#     WP.SHELF45.value: 10,
#     WP.SHELF56.value: 11,
#     WP.DOOR_OFFICE1.value: 12,
#     WP.DOOR_OFFICE2.value: 13,
#     WP.DOOR_TOILET1.value: 14,
#     WP.DOOR_TOILET2.value: 15,
#     WP.DELIVERY_POINT.value: 16,
#     WP.CORRIDOR0.value: 17,
#     WP.CORRIDOR1.value: 18,
#     WP.CORRIDOR2.value: 19,
#     WP.CORRIDOR3.value: 20,
#     WP.CORRIDOR4.value: 21,
#     WP.CORRIDOR5.value: 22,
#     WP.ENTRANCE.value: 23,
#     WP.OFFICE1.value: 24,
#     WP.OFFICE2.value: 25,
#     WP.TOILET1.value: 26,
#     WP.TOILET2.value: 27,
#     WP.TABLE2.value: 28,
#     WP.TABLE3.value: 29,
#     WP.TABLE4.value: 30,
#     WP.TABLE5.value: 31,
#     WP.TABLE6.value: 32,
#     WP.CORR_CANTEEN_1.value: 33,
#     WP.CORR_CANTEEN_2.value: 34,
#     WP.CORR_CANTEEN_3.value: 35,
#     WP.CORR_CANTEEN_4.value: 36,
#     WP.CORR_CANTEEN_5.value: 37,
#     WP.CORR_CANTEEN_6.value: 38,
#     WP.KITCHEN_1.value: 39,
#     WP.KITCHEN_2.value: 40,
#     WP.KITCHEN_3.value: 41,
#     WP.CORRIDOR_CANTEEN.value: 42,
#     WP.SHELF1.value: 43,
#     WP.SHELF2.value: 44,
#     WP.SHELF3.value: 45,
#     WP.SHELF4.value: 46,
#     WP.SHELF5.value: 47,
#     WP.SHELF6.value: 48,
#     WP.CHARGING_STATION.value: 49,
# }


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