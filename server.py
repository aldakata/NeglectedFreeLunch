import numpy as np

import torch
import torchvision
import xml.etree.ElementTree as ET
import os

from siamese_dataloader import Siamese_dataset, Siamese_dataloader

import socket

HOST = ''                 # Symbolic name meaning all available interfaces
PORT = 50007              # Arbitrary non-privileged port
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen(1)
while 1:
    conn, addr = s.accept()
    print 'Connected by', addr
    data = conn.recv(1024)
    if not data: break
    conn.sendall("Working on file" + data)
    
    conn.close()