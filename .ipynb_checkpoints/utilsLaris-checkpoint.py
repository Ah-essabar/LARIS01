import pandas as pd
import numpy as np
import os
import wget

def separteSensors(data, filename, save=False):
    '''
    this function separates the data for a room, dataframs are created by sensor in the form of a dictionary. the call to the separteSensors(data, filename, save=False) function: filename is the name  to save the dictionary if save is True'''
    
    # Number of sensors
    nb_sensors = len(pd.unique(data['sensor'])) 
    sensors_list = data.sensor.unique()
    print("We have ",nb_sensors," sensors. Their Id are ", [i_sensor for i,i_sensor in enumerate(sensors_list)])
    # Separate each sensor data and create a dictionary for all sensors 
    # the dataframe of each senor can be extracted from dictionary ex : DataSensors["sensor_100"]
    DataSensors={}
    for i,i_sensor in enumerate(sensors_list):
        globals()['sensor_%s' % i_sensor] = data.loc[data["sensor"]==i_sensor]    
        globals()['sensor_%s' % i_sensor] = globals()['sensor_%s' % i_sensor].set_index('date')
        globals()['sensor_%s' % i_sensor].drop(["id","sensor","room"],axis=1, inplace = True)
        globals()['sensor_%s' % i_sensor].columns = globals()['sensor_%s' % i_sensor].columns+'_'+str(i_sensor)    
        globals()['sensor_%s' % i_sensor].index = pd.to_datetime(globals()['sensor_%s' % i_sensor].index,dayfirst=True)
        globals()['sensor_%s' % i_sensor].sort_index(inplace=True) # index sorted
        print('sensor_{}'.format(i_sensor))
        dictTemp= {'sensor_%s' % i_sensor: globals()['sensor_%s' % i_sensor] }
        DataSensors.update(dictTemp)
    # Save all sensors in dictionary
    if save:
        np.save(filename+'.npy', DataSensors)
    return DataSensors  
    

## fusion des données par master and all
def dataFusionA(dictSensors, salle=219, all_df = False):
    '''merging of data by Master and all
    ex: df1,df2,df3,df4,df = dataFusion(dictSensors, room=219)
    salle = int(salle)'''
    if salle == 219:
        dict=dictSensors.copy()
        dfs =[dict['sensor_100'],dict['sensor_101'],dict['sensor_102'], dict['sensor_103']]
        df1 = dfs[0].join(dfs[1:])
        dfs =[dict['sensor_104'],dict['sensor_105'],dict['sensor_106'], dict['sensor_107']]
        df2 = dfs[0].join(dfs[1:])
        dfs =[dict['sensor_108'],dict['sensor_109'],dict['sensor_110']]
        df3 = dfs[0].join(dfs[1:])
        dfs =[dict['sensor_111'],dict['sensor_112'],dict['sensor_113']]
        df4 = dfs[0].join(dfs[1:])
        dfs = [df1,df2,df3,df4]
        df = dfs[0].join(dfs[1:])
        
    if salle == 114:
        dict=dictSensors.copy()
        dfs =[dict['sensor_118'],dict['sensor_119'],dict['sensor_120']]
        df1 = dfs[0].join(dfs[1:])
        dfs =[dict['sensor_121'],dict['sensor_122'],dict['sensor_123']]
        df2 = dfs[0].join(dfs[1:])
        dfs =[dict['sensor_124'],dict['sensor_125'],dict['sensor_126']]
        df3 = dfs[0].join(dfs[1:])
        dfs =[dict['sensor_127'],dict['sensor_128'],dict['sensor_129']]
        df4 = dfs[0].join(dfs[1:])
        dfs = [df1,df2,df3,df4]
        df = dfs[0].join(dfs[1:])
        # return all df if all_df=True of not return juste df
    if all_df:
        return df1,df2,df3,df4,df
    else:
        return df
    

    #windows = pd.read_csv('windows',parse_dates=True, index_col='date')
def resampleWindows(windows, period ='5T'):
    
    #a= windows.resample('5T',label='left', closed='left').bfill()
    a = windows.resample(period,label='left', closed='left').bfill()
    timestamp = a.index[1]-a.index[0]
    for i in range(len(windows)-1):
        dt = windows.index[i+1]-windows.index[i]
        if dt > timestamp :
            a[windows.index[i] : windows.index[i+1]] = windows.iloc[:,0][windows.index[i]]
            b = a[windows.index[i] : windows.index[i+1]]
            t= b.index[b.index.shape[0]-1].strftime("%Y-%m-%d %H:%M:%S")
            a[t : t] = windows.iloc[:,0][windows.index[i+1]]
    return a
   

def resampleSensors(dictSensors,period='5T',categorical = False):
    ''' 
    This function makes it possible to aggregate the data according to a given period (5T: for 5 min)'''
    if categorical :
        dict=dictSensors.copy()
        for cle, valeur in dict.items():       
            sensortemp = resampleWindows(valeur, period = period)
            dictTemp= {cle: sensortemp }
            dict.update(dictTemp) 
        return dict

    else :
        dict=dictSensors.copy()
        for cle, valeur in dict.items():       
            sensortemp = valeur.resample(period).mean()
            dictTemp= {cle: sensortemp }
            dict.update(dictTemp) 
        return dict



def outliersToNan(data):
    ''' This function replaces outliers with np.nan'''
    outlier_temp = np.where((data['temperature'] >= (60)) ) # 60°C
    outlier_humidity = np.where(data['humidity'] >= (100)) # 100 %
    outlier_tvoc = np.where(data['tvoc'] >= (10000)) # 10 000 ppb
    #outlier_light = np.where(data['light'] >= (100000)) # 10 000
    outlier_light = np.where(data['light'] >= (65535)) # 10 000
    outlier_sound = np.where(data['sound'] >= (5000)) # 1 000
    outlier_co2 = np.where(data['co2'] >= (10000)) # 65535
    outliers = np.unique(np.concatenate((outlier_temp[0],outlier_humidity[0],outlier_tvoc[0],
                                        outlier_light[0],outlier_sound[0],outlier_co2[0]),0))
    data.loc[outlier_temp[0],"temperature"] = np.nan
    data.loc[outlier_humidity[0],"humidity"] = np.nan
    data.loc[outlier_tvoc[0],"tvoc"] = np.nan
    data.loc[outlier_light[0],"light"] = np.nan
    data.loc[outlier_sound[0],"sound"] = np.nan
    data.loc[outlier_co2[0],"co2"] = np.nan
    return data, outliers



def seperateGrandeurs(df,grandeurs = ["temperature","co2","humidity","sound","tvoc"]):
    '''This function separates the data of a dataFrame by garndeur defined in the variable grandeurs. To call this function use grandeursTemp = seperateGrandeurs(df,grandeurs = {"temperature":,"co2":,"humidity":,"sound":,"tvoc":}).'''
    grandeurs = {grandeurs[i]: [] for i in range(len(grandeurs))}
    grandeursTemp=grandeurs.copy()
    for grandeursTemp_key in  grandeursTemp:
        grandeursTemp[grandeursTemp_key] = []        
    colonnesName=df.columns
    #grandeursTemp = {"temperature": [],"co2": [],"humidity": [],"sound": [],"tvoc": []}
    for grandeursTemp_key in  grandeursTemp:
        for name in colonnesName:            
            if name.find(grandeursTemp_key)==0:
                grandeursTemp[grandeursTemp_key].append(name)
                
    return grandeursTemp


def df_column_uniquify(df):
    df_columns = df.columns
    new_columns = []
    for item in df_columns:
        counter = 0
        newitem = item
        while newitem in new_columns:
            counter += 1
            newitem = "{}_{}".format(item, counter)
        new_columns.append(newitem)
    df.columns = new_columns
    return df

def importData():    
    if os.path.isfile("s114.php")==True:
        os. remove("s114.php")
    if os.path.isfile("s219.php")==True:
        os. remove("s219.php")
    if os.path.isfile("shelly.php")==True:
        os. remove("shelly.php")    
    wget.download("https://biot.u-angers.fr/s219.php")
    wget.download("https://biot.u-angers.fr/s114.php")
    wget.download("https://biot.u-angers.fr/shelly.php")

    for salle in ["s114",'s219', 'shelly']:
        #raw_data = pd.read_csv("test.txt", sep=";")
        sallePhp = salle
        raw_data = pd.read_csv(sallePhp+".php", sep=";")
        if sallePhp != 'shelly':
            data,outliers = outliersToNan(raw_data)
        else :
            data = raw_data
        # Separate sensors and save as dictionary
        filename = sallePhp
        # separteSensors(data, filename, save=False)
        DataSensors = separteSensors(data,filename, save = True )