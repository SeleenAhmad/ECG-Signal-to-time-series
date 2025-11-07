import cv2
import numpy as np
import matplotlib.pyplot as plt
img= cv2.imread(r"C:\Users\DELL\Documents\ECG_images\ecg_1.png" , cv2.THRESH_BINARY)
_,thresh= cv2.threshold(img,100,255,cv2.THRESH_BINARY_INV)
edges= cv2.Canny(thresh,50 , 150)
plt.imshow(edges, cmap='gray')
plt.title("Detected Edges")
plt.show()

time_series=[]

height, width= edges.shape
for col in range(width):
   if col%100 ==0:
       print(f"Processing column {col}/{width}")
   column=edges[:,col]
   white_pixels= np.where(column > 0 )[0]
   if len(white_pixels)>0:
        y=int(np.mean(white_pixels))
   else:
        y= height // 2
   time_series.append(height-y)
plt.figure(figsize=(10, 4))
plt.plot(time_series, color='green')
plt.xlabel("Time (Pixels)")
plt.ylabel("Amplitude(Pixels)")
plt.title("Extracted ECG Signal")
plt.show()
from scipy.interpolate import interp1d
time= np.arange(len(time_series))
f_interp= interp1d(time,time_series,kind= 'cubic')
time_smooth=np.linspace(time.min(), time.max(), len(time_series)*5)
signal_interp= f_interp(time_smooth)
from scipy.signal import savgol_filter
signal_smooth = savgol_filter(signal_interp, window_length=51, polyorder=3)

plt.figure(figsize=(10,4))
plt.plot(time_smooth, signal_interp, color='orange', alpha=0.5, label='Interpolated')
plt.plot(time_smooth,signal_smooth, color='red', linewidth= 2, label='smoothed')
plt.title("Interpolated and smoothed  ECG-like signal")
plt.xlabel("Time(pixels)")
plt.ylabel("Amplitude")
plt.show()
