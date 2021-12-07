import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import skfuzzy.membership as mf


x = np.arange(-4, 7, 1)

#DEFINITIONS OF MEMBERSHIP FUNCTIONS
err_negative = mf.trimf(x, [-4, -2, 0])
err_zero = mf.trimf(x, [-2, 0, 2])

d_err_negative = mf.trimf(x, [-4, -2, 0])
d_err_zero = mf.trimf(x, [-2, 0, 2])
d_err_positive = mf.trimf(x, [0, 2, 4])

c_signal_negative = mf.trimf(x, [-4, -2, 0])
c_signal_zero = mf.trimf(x, [-2, 0, 2])
c_signal_positive = mf.trimf(x, [0, 2, 4])
c_signal_muchPositive = mf.trimf(x, [2, 4, 6])

#DRAWING MEMBERSHIP FUNCTIONS
fig, (ax0, ax1, ax2) = plt.subplots(nrows = 3, figsize = (8, 10))

ax0.plot(x, err_negative, 'r', linewidth = 2, label = 'Neg')
ax0.plot(x, err_zero, 'b', linewidth = 2, label = 'Zer')
ax0.set_title("ERROR")
ax0.legend()

ax1.plot(x, d_err_negative, 'r', linewidth = 2, label = 'Neg')
ax1.plot(x, d_err_zero, 'b', linewidth = 2, label = 'Zer')
ax1.plot(x, d_err_positive, 'y', linewidth = 2, label = 'Poz')
ax1.set_title("ERROR CHANGE")
ax1.legend()

ax2.plot(x, c_signal_negative, 'r', linewidth = 2, label = 'Neg')
ax2.plot(x, c_signal_zero, 'b', linewidth = 2, label = 'Zer')
ax2.plot(x, c_signal_positive, 'y', linewidth = 2, label = 'Poz')
ax2.plot(x, c_signal_muchPositive, 'g', linewidth = 2, label = 'LPoz')
ax2.set_title("CONTROL SIGNAL")
ax2.legend()

#INPUT VALUES
input_err = -1
input_d_err = 1.75

#FINDING MEMBERSHIP GRADES ACCORDING TO THE VALUES
err_fit_negative = fuzz.interp_membership(x, err_negative, input_err)
err_fit_zero = fuzz.interp_membership(x, err_zero, input_err)

d_err_fit_negative = fuzz.interp_membership(x, d_err_negative, input_d_err)
d_err_fit_zero = fuzz.interp_membership(x, d_err_zero, input_d_err)
d_err_fit_positive = fuzz.interp_membership(x, d_err_positive, input_d_err)

#DRAWING INTERSECTION POINTS
ax0.vlines(input_hata, 0, err_fit_negative, linestyles = '--', linewidth = 1, color = 'black')
ax0.vlines(input_hata, 0, err_fit_zero, linestyles = '--', linewidth = 1, color = 'black')

ax1.vlines(input_d_hata, 0, d_err_fit_negative, linestyles = '--', linewidth = 1, color = 'black')
ax1.vlines(input_d_hata, 0, d_err_fit_zero, linestyles = '--', linewidth = 1, color = 'black')
ax1.vlines(input_d_hata, 0, d_err_fit_positive, linestyles = '--', linewidth = 1, color = 'black')

#RULES
rule1 = np.fmin(np.fmin(err_fit_zero, d_err_fit_positive),c_signal_negative)  #RULE-1: IF ERROR IS "ZERO" AND ERROR CHANGE IS "POSITIVE" THEN CONTROL SIGNAL IS "NEGATIVE"
rule2 = np.fmin(np.fmin(err_fit_zero, d_err_fit_zero), c_signal_zero)  #RULE-2: IF ERROR IS "ZERO" AND ERROR CHANGE IS "ZERO" THEN CONTROL SIGNAL IS "ZERO"
rule3 = np.fmin(np.fmin(err_fit_negative, d_err_fit_negative),c_signal_positive)  #RULE-3: IF ERROR IS "NEGATIVE" AND ERROR CHANGE IS "NEGATIVE" THEN CONTROL SIGNAL IS "POSITIVE"
rule4 = np.fmin(np.fmin(err_fit_negative, d_err_fit_zero), c_signal_muchPositive)  #RULE-4: IF ERROR IS "NEGATIVE" AND ERROR CHANGE IS "ZERO" THEN CONTROL SIGNAL IS "MUCH POSITIVE"

#DRAWING FUZZIED CONTROL SIGNAL
fig, ax0 = plt.subplots(figsize = (7, 4))
ax0.plot(x, rule1, 'r', label = "Kural1")
ax0.plot(x, rule2, 'b', label = "Kural2")
ax0.plot(x, rule3, 'y', label = "Kural3")
ax0.plot(x, rule4, 'g', label = "Kural4")
ax0.legend()
ax0.set_title("Fuzzy Inference")

#DEFUZZIFICATION
result = ((np.max(rule1)*fuzz.centroid(x,rule1)) + (np.max(rule2)*fuzz.centroid(x, rule2)) + (np.max(rule3)*fuzz.centroid(x, rule3)) + (np.max(rule4)*fuzz.centroid(x, rule4))) / ((np.max(rule1)+np.max(rule2)+np.max(rule3)+np.max(rule4)))
print("Result of Defuzzification Process:", result)

plt.show()
