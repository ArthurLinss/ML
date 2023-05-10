import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
plt.style.use('seaborn')
import matplotlib.dates as mdates


#csv_file = "groupby.csv"
csv_file = "groupby_month.csv"
#csv_file = "groupby_day.csv"

data = pd.read_csv(csv_file)


plt.figure(figsize=(17, 8))

#dates = [datetime.strptime(date, '%Y-%M-%d').date() for date in data["Buchungsdatum (BL)_datetime"]]
#print(dates)


plt.title('MERCEDES ST')
plt.ylabel('Number of entries')
plt.xlabel('Time')
#plt.grid(True)

plt.plot(data["Buchungsdatum (BL)_datetime"], data.ST)
#plt.plot(dates, data.ST)
#plt.plot(data.ST)

plt.show()
