import pandas as pd
import janitor
import numpy as np
import os
import wget
import glob
import functools
from datetime import datetime
from datetime import timedelta

prefixFiles = {"ElecS219" :"S219*.csv","ElecS114" :"S114*.csv", "Weather" :"WeatherFile*.txt", "Ambiance114": "s114*.txt", "Ambiance219": "s219*.txt"}
prefixFile = "ElecS219"


def separteSensors(data, filename, save=False):
    '''
    this function separates the data for a room, dataframs are created by sensor in the form of a dictionary. the call to the separteSensors(data, filename, save=False) function: filename is the name  to save the dictionary if save is True''', 
    
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
def dataFusionAmbiance(dictSensors, salle=219, all_df = False):
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
            ## Ajout de une ligne close au début
            data = [[valeur.index[0] - timedelta(days=25), "close"]]
            # Create the pandas DataFrame
            df1 = pd.DataFrame(data, columns=['date', valeur.columns[0]])
            df1.set_index ('date', inplace= True)
            df1.index=pd.DatetimeIndex(df1.index)
            valeur = pd.concat([df1,valeur], ignore_index=False)
            ## ajout 2 lignes at the end
            data = [[valeur.index[-1] + timedelta(minutes=1), "close"], [datetime.now(), "close"]]
            # Create the pandas DataFrame
            df1 = pd.DataFrame(data, columns=['date', valeur.columns[0]])
            df1.set_index ('date', inplace= True)
            df1.index=pd.DatetimeIndex(df1.index)
            valeur = pd.concat([valeur,df1], ignore_index=False)
            #print(valeur)
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


def importData(annee ="2022", n_monthStart=2,n_monthEnd=5 ):    
    if os.path.isfile("s114.php")==True:
        os. remove("s114.php")
    if os.path.isfile("s219.php")==True:
        os. remove("s219.php")
    if os.path.isfile("shelly.php")==True:
        os. remove("shelly.php")
   #####################   Vider le dossier   ImportedData
    py_files = glob.glob('./ImportedData/*.txt')
    for py_file in py_files:
        try:
            os.remove(py_file)
        except OSError as e:
            print(f"Error:{ e.strerror}")
    ###################################        
    wget.download("https://biot.u-angers.fr/shelly.php")    
    for mois in range (n_monthStart,n_monthEnd+1):
        wget.download("https://biot.u-angers.fr/data/s114/"+annee+"/"+str(mois), out="ImportedData/s114_"+annee+"_"+str(mois)+".txt")
        wget.download("https://biot.u-angers.fr/data/s219/"+annee+"/"+str(mois), out="ImportedData/s219_"+annee+"_"+str(mois)+".txt")
        print(mois)
    print("start merging")    
    df_114 = mergeMultipleCSV_Files(dirctory = "./ImportedData", prefixFile = prefixFiles["Ambiance114"])
    #df_114.se_index("id", inplace = True)
    df_114.to_csv("s114.php",  sep=';', index=False)
    df_219 = mergeMultipleCSV_Files(dirctory = "./ImportedData", prefixFile = prefixFiles["Ambiance219"])
    #df_219.se_index("id", inplace = True)
    df_219.to_csv("s219.php", sep=';', index=False)
    for salle in ["s114",'s219', 'shelly']:
        #raw_data = pd.read_csv("test.txt", sep=";")
        sallePhp = salle
        raw_data = pd.read_csv(sallePhp+".php", sep=";")
        if sallePhp != 'shelly':
            data,outliers = outliersToNan(raw_data)
        else :
            data = raw_data
            #print(data.head())
        # Separate sensors and save as dictionary
        filename = sallePhp
        # separteSensors(data, filename, save=False)
        DataSensors = separteSensors(data,filename, save = True )
        
        
def readData(period='5T'):
    tab=[]
    for fileNpy in ["s114","s219","shelly"]:
        filename = fileNpy+'.npy'    
        dictionary = np.load(filename,allow_pickle='TRUE').item()
        dictSensors = dictionary.copy()
        dict = dictSensors .copy()
        if fileNpy != "shelly" :
            categorical = False
        else :
            categorical = True
        tab.append(resampleSensors(dictSensors, period = period,categorical = categorical))
    return tab

def dataPreparationElec(data, period = "5T"):
    data = data.set_index("date") 
    data.index = pd.to_datetime(data.index)
    data = data.clean_names() # janitor
    data.columns="elec_"+data.columns
    data.fillna(0,inplace=True)
    data = data.resample(period).mean()
    return data

def dataPreparationWeather(weatherData, period = "5T"):
    weatherData = weatherData.rename({'Date_Time':'date'}, axis=1)
    weatherData  = weatherData .set_index("date")
    # we use import janitor to clean colonne's name
    weatherData = weatherData .clean_names()
    # df_column_uniquify : allows to have unique column names
    weatherData  = df_column_uniquify(weatherData )
    weatherData  = weatherData[['out','hum','bar_','rad_']]
    weatherData.columns="weather_"+weatherData.columns
    weatherData  = weatherData .sort_index()
    # missing data in raw fil is coded as "---" . so we change this chaine by np.nan
    x = {'---':np.nan}
    weatherData = weatherData.replace(x)
    weatherData = weatherData.astype(float)
    weatherData = weatherData.resample(period,label='left', closed='left').mean()
    weatherData = weatherData.interpolate(method='linear', limit_direction='both', axis=0)
    return weatherData


def mergeMultipleCSV_Files(dirctory="./Data", prefixFile = prefixFile): 
    # merging the files
    joined_files = os.path.join(dirctory, prefixFile)
    # A list of all joined files is returned
    joined_list = glob.glob(joined_files)
    # Finally, the files are joined
    if prefixFile == "S219*.csv":
        li_mapper = map(lambda filename: pd.read_csv(filename, sep=",", skiprows=(1),names = ['date','general_219_w', 'eclairage_219_w']),joined_list)
    if prefixFile == "S114*.csv":
        li_mapper = map(lambda filename: pd.read_csv(filename, sep=",", skiprows=(1),names =["date","Prises_114_W","General_114_W","Eclairage_114_W","Videoproj_114_W"]),joined_list)
    if prefixFile == "WeatherFile*.txt" :
        li_mapper = map(lambda filename: pd.read_csv(filename, sep="\t",skiprows=(1),parse_dates=[['Date','Time']],dayfirst=True),joined_list)
    if prefixFile in ["s114*.txt", "s219*.txt"]: # merge data imported from BIot
        li_mapper = map(lambda filename: pd.read_csv(filename, sep=";"),joined_list)        
    li_2 = list(li_mapper)    
    df = pd.concat(li_2, axis=0, ignore_index= True)        
    return df
        
        

def dataFusionAll(dfs, shelly_sensors,ResampledDict_shelly):
    for sensor_name in shelly_sensors: 
        dfs.append(ResampledDict_shelly[sensor_name])
    df = functools.reduce(lambda left,right: pd.merge(left,right,on='date'), dfs)
    return df         
