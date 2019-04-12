import numpy as np
import matplotlib.pyplot as mp

value = [14,23,22,45,10]
types = ["apple","banana","strawberry","pear","watermelan"]

mp.figure().set_facecolor(np.ones(3) * 240/255)
mp.title("Pie",fontsize = 24)
mp.pie(value,(0.05, 0.01, 0.01, 0.01, 0.01),types,
       ["dodgerblue","limegreen",'orangered',"yellow","violet"],
       "%d%%",shadow = True, startangle = 180)
mp.show()