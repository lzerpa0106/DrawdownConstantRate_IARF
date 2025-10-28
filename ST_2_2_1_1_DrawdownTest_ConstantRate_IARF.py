#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 08:55:43 2025

Drawdown Well Test - Semilog Analysis

Constant rate production and Infinite Acting Radial Flow

@author: luiszerpa
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Well Test Data
# "t, hr",
time = np.array([0.001,0.0021,0.0034,0.0048,0.0064,0.0082,0.0102,0.0125,0.0151,0.018,0.0212,0.0249,0.029,0.0336,0.0388,0.0447,0.0512,0.0587,0.067,0.0764,0.0869,0.0988,0.1121,0.1271,0.144,0.163,0.1844,0.208,0.236,0.266,0.3,0.339,0.382,0.431,0.486,0.547,0.617,0.695,0.783,0.882,0.993,1.118,1.259,1.417,1.595,1.795,2.021,2.275,2.56,2.881,3.242,3.648,4.105,4.619,5.198,5.848,6.58,7.404,8.331,9.373,10.545,11.865,13.349,15.018,16.897,19.010,21.387,24.061,27.070,30.455,34.262,38.546,43.366,48.787,54.787,60.787,66.787,72.000])
# "pwf, psia",
pressure = np.array([2748.95,2745.62,2744.63,2745.49,2741.7,2742,2736.69,2737.26,2733.72,2729.13,2724.23,2720.57,2715.83,2710.7,2706.63,2698.17,2692.75,2684.56,2676.82,2665.33,2655.67,2642.29,2627.5,2614.76,2598.79,2582.16,2564.54,2545.27,2523.21,2501.07,2475.93,2451.83,2422.8,2397.61,2367.5,2338.18,2309.21,2277.84,2251.46,2222.09,2196.92,2170.7,2148.33,2126.44,2108.5,2090.87,2080.73,2066.59,2054.29,2048.25,2039.49,2035.32,2029.91,2025.01,2018.87,2016.4,2011.11,2007.46,2003.24,2000.53,1995.75,1991.15,1988.67,1984.74,1979.34,1981.14,1973.78,1970.58,1967.59,1965.5,1961.64,1957.61,1955.9,1951.21,1949.05,1945.7,1942.51,1941.14])

# Rock and fluid properties
q = 125.0 # Constant flow rate, STB/day
porosity = 0.22
h = 32 # Reservoir thickness, ft	
rw = 0.25 # Wellbore radius, ft
pi = 2750.0 # Initial reservoir pressure, psia
B = 1.152 # Oil formation volume factor, RB/STB
viscosity = 2 # Oil viscosity, cp.122
ct = 10.9e-6 # Total compressibility, 1/psi


st.title("Drawdown Well Test - Semilog Analysis")

st.text('Constant rate production and Infinite Acting Radial Flow')


# ---- User-defined range for fitting ----
start_index = st.slider("Start index", 0, np.size(time), 55)   # e.g. fit starting from index 10
end_index   = st.slider("End index", 0, np.size(time), 77)   # e.g. fit ending at index 50


# Select data range
x = np.log10(time[start_index:end_index])
y = pressure[start_index:end_index]

# Fit straight line model
coefficients, residuals, _, _, _ = np.polyfit(x, y, deg=1, full=True)

slope = coefficients[0]
intercept = coefficients[1]

SSE = residuals[0]
mean_y = np.mean(y)
SST = np.sum((y - mean_y)**2)
R_squared = 1 - (SSE / SST)

y_pred = intercept + slope*np.log10(time)


p_1hr = intercept

''' Calculation '''

# Permeability
k = -162.6*q*B*viscosity/(slope*h)

# Skin Factor
Skin = 1.151*((pi - p_1hr)/np.abs(slope) - np.log10(k/(porosity*viscosity*ct*rw**2)) +3.23)

# Radius of investigation at the start and end of the semilog straight line
ri_start = np.sqrt(k*time[start_index]/(948.0*porosity*viscosity*ct))
ri_end = np.sqrt(k*time[end_index]/(948.0*porosity*viscosity*ct))

# Print results

st.text(f"Fitted line: y = {slope:.4f} * x + {intercept:.4f}")
st.text(f"Coefficient of determination (R²): {R_squared:.4f}")
st.text(f"Permeability): {k:.4f}")
st.text(f"Skin Factor): {Skin:.4f}")
st.text(f"Radius of investigation at the start): {ri_start:.4f}")
st.text(f"Radius of investigation at the end): {ri_end:.4f}")


fig, ax = plt.subplots(figsize=(8, 5))  # Create a figure containing a single axes.
ax.plot(time,pressure,'ob',linewidth=3.0,label='Original data') # Plot data on the axes.
ax.plot(time, y_pred, color='red', linewidth=2, label='Fitted line')
ax.plot(1.0, intercept, '.k', markersize=10, label='Fitted line')
ax.set_xlabel('Time (hr)',fontsize='12',fontweight='bold', fontname='Verdana')
ax.set_ylabel('Pressure (psia)',fontsize='12',fontweight='bold',fontname='Verdana')
ax.set_xscale('log')
ax.tick_params(direction='in',width=1.5,length=8.0, labelsize=12.0)
# ax.set(xlim=(0, 4500), ylim=(1.0, 1.3))
for tick in ax.get_xticklabels():
    tick.set_fontname('Verdana')
for tick in ax.get_yticklabels():
    tick.set_fontname('Verdana')
ax.grid(True)     
st.pyplot(fig)
