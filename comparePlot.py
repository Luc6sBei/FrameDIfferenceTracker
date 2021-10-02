import matplotlib.pyplot as plt
import numpy as np
import datetime
import csv

RELATIVE_DESTINATION_PATH = 'compare/'
total_error_array_diff = np.zeros(0)
total_error_array_Otsu = np.zeros(0)
error_array_diff = np.zeros(0)
error_array_Otsu = np.zeros(0)
speed_array_diff = np.zeros(0)
speed_array_Otsu = np.zeros(0)
dis_array_diff = np.zeros(0)
dis_array_Otsu = np.zeros(0)
time_array = np.zeros(0)

with open(RELATIVE_DESTINATION_PATH + "Otsu_results.csv", "r") as csvFile:
    reader = csv.reader(csvFile)
    rows = [row for row in reader]

for n in range(1, len(rows)):
    time_array = np.append(time_array, float(rows[n][1]))
    dis_array_Otsu = np.append(dis_array_Otsu, float(rows[n][2]))
    speed_array_Otsu = np.append(speed_array_Otsu, float(rows[n][3]))
    error_array_Otsu = np.append(error_array_Otsu, float(rows[n][6]))
    total_error_array_Otsu = np.append(total_error_array_Otsu, float(rows[n][7]))

with open(RELATIVE_DESTINATION_PATH + "diff_results.csv", "r") as csvFile:
    reader = csv.reader(csvFile)
    rows = [row for row in reader]

for n in range(1, len(rows)):
    dis_array_diff = np.append(dis_array_diff, float(rows[n][2]))
    speed_array_diff = np.append(speed_array_diff, float(rows[n][3]))
    error_array_diff = np.append(error_array_diff, float(rows[n][6]))
    total_error_array_diff = np.append(total_error_array_diff, float(rows[n][7]))


# plot the total error
save_path = '/Users/lucas/Desktop/images/'
plt.plot(time_array[1::], total_error_array_Otsu[1::], label='Otsu')
plt.plot(time_array[1::], total_error_array_diff[1::], label='DFM')
plt.xlabel('Time(s)')
plt.ylabel('Total error')
plt.title('Total error plot')
plt.savefig(save_path+'totalError')
plt.legend(loc='best')
plt.show()
# plot the error
plt.plot(time_array[1::], error_array_Otsu[1::], label='Otsu')
plt.plot(time_array[1::], error_array_diff[1::], label='DFM')
plt.xlabel('Time(s)')
plt.ylabel('Error')
plt.title('Error plot')
plt.savefig(save_path+'error')
plt.legend(loc='best')
plt.show()
# plot the speed
plt.plot(time_array[1::], speed_array_Otsu[1::], label='Otsu')
plt.plot(time_array[1::], speed_array_diff[1::], label='DFM')
plt.xlabel('Time(s)')
plt.ylabel('Speed(unit/s)')
plt.title('Speed plot')
plt.savefig(save_path+'speed')
plt.legend(loc='best')
plt.show()
# plot the distance
plt.plot(time_array[1::], dis_array_Otsu[1::], label='Otsu')
plt.plot(time_array[1::], dis_array_diff[1::], label='DFM')
plt.xlabel('Time(s)')
plt.ylabel('Distance(unit)')
plt.title('Distance plot')
plt.savefig(save_path+'distance')
plt.legend(loc='best')
plt.show()
