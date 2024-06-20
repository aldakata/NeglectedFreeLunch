import numpy as np
import xml.etree.ElementTree as ET


xml_file = '../train_xml/n01440764/n01440764_105.xml'
xml_file = '../train_xml/n01440764/n01440764_10026.xml'

tree = ET.parse(xml_file)
root = tree.getroot()

selected_record = []
selected_record_time = -1
selected = False
estimateTime = -1
worker_id = ''
assignment_id = ''
hovered_record = []
mouse_record = []
for obj in root.findall("metadata"):
    selected = bool(obj.find("selected").text)
    worker_id = obj.find("worker_id").text
    assignment_id = obj.find("assignment_id").text
    
    if obj.find("selected").text == "True":
        selected_point = obj.find("selectedRecord")
        selected_record.append(float(selected_point.find("x").text))
        selected_record.append(float(selected_point.find("y").text))
        selected_record_time = float(selected_point.find("time").text)
        estimateTime = int(obj.find("estimateTime").text)

    for hovered_point in  obj.findall("hoveredRecord"):
        payload = hovered_point.find("action").text,float(hovered_point.find("time").text)
        hovered_record.append(payload)

    for mouse_point in  obj.findall("mouseTracking"):
        payload = float(mouse_point.find("time").text),float(mouse_point.find("x").text), float(mouse_point.find("y").text)
        mouse_record.append(payload)

print(f"""
{selected_record}
{selected_record_time}
{selected}
{estimateTime}
{worker_id}
{assignment_id}
{hovered_record}
{mouse_record}
""")