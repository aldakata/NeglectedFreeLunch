import numpy as np
from scipy.interpolate import make_interp_spline
import os
import xml.etree.ElementTree as ET

REGULAR_TS = np.array([   
        0,  100,  200,  300,  400,  500,  600,  700,  800, 900, 
        1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700,1800, 1900, 
        2000, 2100, 2200, 2300, 2400, 2500, 2600,2700, 2800, 2900, 
        # 3000., 3100., 3200., 3300., 3400., 3500.,3600., 3700., 3800., 3900.
        ])

def regularize_mouse_record(mouse_record, estimate_time, point_fg):
        REGULAR_TS = np.array([   
        0,  100,  200,  300,  400,  500,  600,  700,  800, 900, 
        1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700,1800, 1900, 
        2000, 2100, 2200, 2300, 2400, 2500, 2600,2700, 2800, 2900, 
        # 3000., 3100., 3200., 3300., 3400., 3500.,3600., 3700., 3800., 3900.
        ])

        default_mouse_record = np.array([np.ones_like(REGULAR_TS), np.ones_like(REGULAR_TS)], dtype=np.float32).T
        try:
            t0 = mouse_record[0,0]
            mouse_record[:,0] = mouse_record[:,0] - t0
            tf = min(mouse_record[-1,0], estimate_time)
            idxf = min(int(tf//100) + 1, len(REGULAR_TS)-1)
            if (mouse_record[:,0]<estimate_time).sum() < 4:
                idxf = "Throw an error"

            bspl = make_interp_spline(mouse_record[:,0], mouse_record[:, 1:].T, k=3,axis=1)
            default_mouse_record[:idxf] = bspl(REGULAR_TS[:idxf]).T
            mask_away = np.logical_not(np.any(default_mouse_record > 1, axis=1)|np.any(default_mouse_record < 0, axis=1))[:, None]
            default_mouse_record = mask_away * default_mouse_record
            default_mouse_record[idxf:] *= point_fg
            return default_mouse_record
            # return default_mouse_record

        except Exception as e: 
            # print(f"Error: {e}")#, mouse_record: ts : {mouse_record[:,0]-t0}, x-y: {mouse_record[:,1:]}")
            # continue
            return np.array([np.zeros_like(REGULAR_TS), np.zeros_like(REGULAR_TS)], dtype=np.float32).T * 0.
        
def load_points_interpolated(xml_file):
    DEFAULT_MOUSE_RECORD = np.array([np.ones_like(REGULAR_TS), np.ones_like(REGULAR_TS)], dtype=np.float32).T

    selected_record_default = -1 * np.ones((1,2))
    estimateTime_default = -1
    mouse_record_default = DEFAULT_MOUSE_RECORD*0.
    np.array([np.ones_like(REGULAR_TS), np.ones_like(REGULAR_TS)], dtype=np.float32).T

    if not os.path.isfile(xml_file):
        return selected_record_default, estimateTime_default, mouse_record_default

    tree = ET.parse(xml_file)
    root = tree.getroot()
    selected_record = []
    mouse_record = []
    estimateTime = estimateTime_default
    for obj in root.findall("metadata"):
        if obj.find("selected").text == "True":
            selected_point = obj.find("selectedRecord")
            if len(selected_record) == 0:
                selected_record.append(
                    [
                        float(selected_point.find("x").text),
                        float(selected_point.find("y").text),
                    ]
                )
            estimateTime = float(obj.find("estimateTime").text)

            for mouse_point in  obj.findall("mouseTracking"):
                time = mouse_point.find("time")
                x = mouse_point.find("x")
                y = mouse_point.find("y")
                if (x is not None) and (y is not None) and (time is not None):
                    payload = [float(time.text),float(x.text), float(y.text)]
                    mouse_record.append(payload)


    if len(selected_record) == 0:
        selected_record = selected_record_default
        mouse_record = mouse_record_default
    else:
        selected_record = np.array(selected_record)
        mouse_record = regularize_mouse_record(np.array(mouse_record), estimateTime, selected_record)

    if estimateTime == 0:
        estimateTime = estimateTime_default

    return selected_record, estimateTime, mouse_record

def load_points(xml_file):
    selected_record_default = -1 * np.ones((1,2))
    estimateTime_default = -1
    mouse_record_default = np.array([np.ones_like(REGULAR_TS)*-1, np.ones_like(REGULAR_TS)*-1, np.ones_like(REGULAR_TS)*-1]).T
    if not os.path.isfile(xml_file):
        return selected_record_default, estimateTime_default, mouse_record_default

    tree = ET.parse(xml_file)
    root = tree.getroot()
    selected_record = []
    mouse_record = []
    estimateTime = estimateTime_default
    for obj in root.findall("metadata"):
        if obj.find("selected").text == "True":
            selected_point = obj.find("selectedRecord")
            if len(selected_record) == 0:
                selected_record.append(
                    [
                        float(selected_point.find("x").text),
                        float(selected_point.find("y").text),
                    ]
                )
            estimateTime = float(obj.find("estimateTime").text)

            for mouse_point in  obj.findall("mouseTracking"):
                time = mouse_point.find("time")
                x = mouse_point.find("x")
                y = mouse_point.find("y")
                if (x is not None) and (y is not None) and (time is not None):
                    payload = [float(time.text),float(x.text), float(y.text)]
                    mouse_record.append(payload)


    if len(selected_record) == 0:
        selected_record = selected_record_default

    if estimateTime == 0:
        estimateTime = estimateTime_default

    return np.array(selected_record), estimateTime, np.array(mouse_record)

def is_turd(path):
    ret = True
    ret &= path.endswith('.xml')
    ret &= "._" not in path
    return ret
