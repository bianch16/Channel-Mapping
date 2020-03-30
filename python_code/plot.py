#####  Painting the Communication Rate    #####
import numpy as np 
import matplotlib.pyplot as plt
SNR = [-10,-5,0,5,10,15,20]

overall_upper = [0.133,0.26,0.411,0.576,0.70,0.785,0.831]
overall_lower = [0.1268,0.2466,0.4088,0.5656,0.7056,0.7825,0.8284]
pilot_aided = [0.60,0.809,0.897,0.925,0.939,0.945,0.947]
fusion = [0.8026352075992033, 0.9131913589704305, 0.951647004749502, 0.96182013176038, 0.9647924007966907, 0.9675808181400337, 0.969112915581431]
plt.grid()
plt.xlabel('SNR')
plt.ylabel('Accuracy')
plt.xlim(-10, 20)
plt.ylim(0, 1)

#plt.plot(SNR,overall_pred,'bo-',label='')
plt.title('TOP-1 ACCURACY FOR SUB-6GHZ BASED MMWAVE BEAM PREDICTION', fontsize=12, color='r')
plt.plot(SNR,overall_upper,'bo-',label='thesis')
plt.plot(SNR,pilot_aided,'rs-',label='pilot_aided')
plt.plot(SNR,fusion,'gs-',label='fusion network')

plt.legend(loc = 'center right')
plt.show()