import numpy as np

from forex import Forex
from parse_data import ParseData

file = 'eur_usd_m5.csv'
data = ParseData(file).get_file()

forex = Forex(data=data)
for i in range(200000):
    forex.play(2)
    print(forex.get_score())
    print(forex.get_actual_bar())
    print(forex.get_state().shape)
