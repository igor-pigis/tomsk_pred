NAME = 'smoothfunction'
import statistics
import pickle
#with open('smooth_dict.pkl', 'rb') as f:
#    smooth_dict = pickle.load(f)

def smooth_1(series):
    result = series
    for i in range(4,len(result)-6):
        neighbors_avg_1 = result.iloc[i-3:i-1].median()
        neighbors_avg_2 = result.iloc[i+1:i+3].median()
        if (abs(result.iloc[i] - neighbors_avg_1) > 0.3 * neighbors_avg_1) and (abs(result.iloc[i] - neighbors_avg_2) > 0.3 * neighbors_avg_2) :
            result.iloc[i] = statistics.median([neighbors_avg_1,neighbors_avg_2])
        if result.iloc[i] < 80:
            result.iloc[i] = result.iloc[i-10:i+5].median()
    return result

def smooth_2(series):
    result = series
    for i in range(4,len(result)-4):
        neighbors_avg_1 = result.iloc[i-3:i-1].median()
        neighbors_avg_2 = result.iloc[i+1:i+3].median()
        if (abs(result.iloc[i] - neighbors_avg_1) > 0.5 * neighbors_avg_1) and (abs(result.iloc[i] - neighbors_avg_2) > 0.5 * neighbors_avg_2) :
            result.iloc[i] = statistics.median([neighbors_avg_1,neighbors_avg_2])
    return result

def smooth_3(series):
    result = series
    for i in range(5,len(result)-7):
        neighbors_avg_1 = result.iloc[i-7:i-1].median()
        neighbors_avg_2 = result.iloc[i+1:i+7].median()
        if (abs(result.iloc[i] - neighbors_avg_1) > 0.5 * neighbors_avg_1) and (abs(result.iloc[i] - neighbors_avg_2) > 0.5 * neighbors_avg_2) :
            result.iloc[i] = statistics.median([neighbors_avg_1,neighbors_avg_2])
        if result.iloc[i] < 20 :
            result.iloc[i] = result.iloc[i-10:i+5][result.iloc[i-10:i+5] != 0].median()
    return result

def smooth_4(series):
    result = series
    for i in range(5,len(result)-7):
        neighbors_avg_1 = result.iloc[i-7:i-1].median()
        neighbors_avg_2 = result.iloc[i+1:i+7].median()
        if (abs(result.iloc[i] - neighbors_avg_1) > 0.5 * neighbors_avg_1) and (abs(result.iloc[i] - neighbors_avg_2) > 0.5 * neighbors_avg_2) :
            result.iloc[i] = statistics.median([neighbors_avg_1,neighbors_avg_2])
        if result.iloc[i] < 5 :
            result.iloc[i] = result.iloc[i-10:i+5][result.iloc[i-10:i+5] != 0].median()
    return result

def smooth_7(series):
    result = series
    for i in range(5,len(result)-7):
        neighbors_avg_1 = result.iloc[i-7:i-1].median()
        neighbors_avg_2 = result.iloc[i+1:i+7].median()
        if (abs(result.iloc[i] - neighbors_avg_1) > 0.5 * neighbors_avg_1) and (abs(result.iloc[i] - neighbors_avg_2) > 0.5 * neighbors_avg_2) :
            result.iloc[i] = statistics.median([neighbors_avg_1,neighbors_avg_2])
        if result.iloc[i] == 0:
            result.iloc[i] = result.iloc[i-5:i+5][result.iloc[i-5:i+5] != 0].median()
    return result

def smooth_8(series):
    result = series
    for i in range(5,len(result)-8):
        neighbors_avg_1 = result.iloc[i-7:i-1].median()
        neighbors_avg_2 = result.iloc[i+1:i+7].median()
        if (abs(result.iloc[i] - neighbors_avg_1) > 0.5 * neighbors_avg_1) and (abs(result.iloc[i] - neighbors_avg_2) > 0.5 * neighbors_avg_2) :
            result.iloc[i] = statistics.median([neighbors_avg_1,neighbors_avg_2])
        if result.iloc[i] < 2:
            result.iloc[i] = result.iloc[i-5:i+5].median()
    return result

def smooth_10(series):
    result = series
    for i in range(5,len(result)-6):
        neighbors_avg_1 = result.iloc[i-6:i-3].median()
        neighbors_avg_2 = result.iloc[i+3:i+6].median()
        if (abs(result.iloc[i] - neighbors_avg_1) > 0.3 * neighbors_avg_1) and (abs(result.iloc[i] - neighbors_avg_2) > 0.3 * neighbors_avg_2) :
            result.iloc[i] = statistics.median([neighbors_avg_1,neighbors_avg_2])
        if result.iloc[i] == 0:
            result.iloc[i] = result.iloc[i-10:i+10][result.iloc[i-10:i+10] != 0].median()
    return result

def smooth_16(series):
    result = series
    for i in range(4,len(result)-5):
        neighbors_avg_1 = result.iloc[i-4:i-3].median()
        neighbors_avg_2 = result.iloc[i+3:i+4].median()
        if (abs(result.iloc[i] - neighbors_avg_1) > 0.3 * neighbors_avg_1) and (abs(result.iloc[i] - neighbors_avg_2) > 0.3 * neighbors_avg_2) :
            result.iloc[i] = statistics.median([neighbors_avg_1,neighbors_avg_2])
    return result

def smooth_21(series):
    result = series
    for i in range(5,len(result)-11):
        if result.iloc[i] < 20:
            result.iloc[i] = result.iloc[i-10:i+10][result.iloc[i-10:i+10] > 40].median()
    return result

def smooth_24(series):
    result = series
    for i in range(7,len(result)-6):
        neighbors_avg_1 = result.iloc[i-6:i-3].median()
        neighbors_avg_2 = result.iloc[i+3:i+6].median()
        if (abs(result.iloc[i] - neighbors_avg_1) > 0.3 * neighbors_avg_1) and (abs(result.iloc[i] - neighbors_avg_2) > 0.3 * neighbors_avg_2) :
            result.iloc[i] = statistics.median([neighbors_avg_1,neighbors_avg_2])
    return result

def smooth_37(series):
    result = series
    for i in range(7,len(result)-11):
        neighbors_avg_1 = result.iloc[i-7:i-1].median()
        neighbors_avg_2 = result.iloc[i+1:i+7].median()
        if (abs(result.iloc[i] - neighbors_avg_1) > 0.3 * neighbors_avg_1) and (abs(result.iloc[i] - neighbors_avg_2) > 0.3 * neighbors_avg_2) :
            result.iloc[i] = statistics.median([neighbors_avg_1,neighbors_avg_2])
        if result.iloc[i] == 0:
            result.iloc[i] = result.iloc[i-10:i+10][result.iloc[i-10:i+10] != 0].median()
    return result

def smooth_38(series):
    result = series
    for i in range(7,len(result)-11):
        neighbors_avg_1 = result.iloc[i-7:i-1].median()
        neighbors_avg_2 = result.iloc[i+1:i+7].median()
        if (abs(result.iloc[i] - neighbors_avg_1) > 0.5 * neighbors_avg_1) and (abs(result.iloc[i] - neighbors_avg_2) > 0.5 * neighbors_avg_2) :
            result.iloc[i] = statistics.median([neighbors_avg_1,neighbors_avg_2])
        if result.iloc[i] == 0:
            result.iloc[i] = result.iloc[i-10:i+10][result.iloc[i-10:i+10] != 0].median()
    return result

def smooth_79(series):
    result = series
    for i in range(7,len(result)-7):
        neighbors_avg_1 = result.iloc[i-7:i-1].median()
        neighbors_avg_2 = result.iloc[i+1:i+7].median()
        if (abs(result.iloc[i] - neighbors_avg_1) > 0.3 * neighbors_avg_1) and (abs(result.iloc[i] - neighbors_avg_2) > 0.3 * neighbors_avg_2) :
            result.iloc[i] = statistics.median([neighbors_avg_1,neighbors_avg_2])
    return result
    
def smooth_142(series):
    result = series
    for i in range(6,len(result)-6):
        neighbors_avg_1 = result.iloc[i-6:i-3].median()
        neighbors_avg_2 = result.iloc[i+3:i+6].median()
        if (abs(result.iloc[i] - neighbors_avg_1) > 0.3 * neighbors_avg_1) and (abs(result.iloc[i] - neighbors_avg_2) > 0.3 * neighbors_avg_2) :
            result.iloc[i] = statistics.median([neighbors_avg_1,neighbors_avg_2])
        if result.iloc[i] == 0:
            result.iloc[i] = result.iloc[i-20:i+20].median()
    return result

def smooth_anomalies(series, threshold=0.3):
    series_smoothed = series.copy()
    for i in range(1, len(series) - 2):
        left = series.iloc[i - 1]
        right = series.iloc[i + 1]
        current = series.iloc[i]
        left_left = series.iloc[i - 2]
        right_right = series.iloc[i + 2]
        current = series.iloc[i]
        
        if abs(current - left) / left > threshold and abs(current - right) / right > threshold:
            series_smoothed.iloc[i] = (left_left + right_right) / 2      
    for i in range(1, len(series_smoothed) - 1):
        left = series_smoothed.iloc[i - 1]
        right = series_smoothed.iloc[i + 1]
        current = series_smoothed.iloc[i]
        
        if abs(current - left) / left > threshold and abs(current - right) / right > threshold:
            series_smoothed.iloc[i] = (left + right) / 2   
            
    for i in range(1, len(series_smoothed) - 2):
        left_left = series_smoothed.iloc[i - 2]
        right_right = series_smoothed.iloc[i + 2]
        current = series_smoothed.iloc[i]
        
        if abs(current - left_left) / left_left > threshold and abs(current - right_right) / right_right > threshold:
            series_smoothed.iloc[i] = (left_left + right_right) / 2
    
    return series_smoothed