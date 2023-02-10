import folium
from geopandas import GeoDataFrame
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib
import datetime
import geopy.distance
from folium import plugins
import plotly.graph_objects as go
from CNOSSOS import *
from collections import defaultdict, OrderedDict
from IPython.display import display_html
from itertools import chain,cycle
from branca.element import Figure

positions = np.array([[37.99174325689546, 23.731320327878496],
[37.99159316923085, 23.73234123268387],
[37.991339021048596, 23.734039084803616],
[37.991331432695844, 23.734073800158672],
[37.99089072041197, 23.736842672159554],
[37.9909113833555, 23.736746212821014],
[37.99032001805405, 23.740375798507234],
[37.98994148312128, 23.730840211656457],
[37.98961862165934, 23.730765809461428],
[37.98858493362708, 23.730459986236557],
[37.98754668026814, 23.730229037800676],
[37.9929105175198, 23.731531949450392],
[37.99468785729667, 23.732020673145563],
[37.994679170936905, 23.732008712117196],
[37.99563131589096, 23.732237194628677],
[37.99676656009595, 23.732520894496048],
[37.99803603945146, 23.73284739373721],
[38.00009190308469, 23.73336335097613],
[37.991076471465455, 23.735695398061246]])
la = positions[:,0]
lo = positions[:,1]

def dataReading():
    """read the pneum data from csv file

    Returns:
        data: list containing the file
    """
    data = []
    with open('data.csv') as f:
        lines=f.readlines()
        for line in lines:
            if line.startswith('track_id'):
                continue
            read_line = line.split('; ')
            track_id = int(read_line[0])
            type = read_line[1]
            traveled_d = float(read_line[2])
            avg_speed = float(read_line[3])
            lat = np.array(list(map(lambda i: 0.0 if i=='' or i.isspace() else float(i), read_line[4::6][:-1][::25])))
            lon = np.array(list(map(lambda i: 0.0 if i=='' or i.isspace() else float(i), read_line[5::6][::25])))
            speed = np.array(list(map(lambda i: 0.0 if i=='' or i.isspace() else float(i), read_line[6::6][::25])))
            lon_acc = np.array(list(map(lambda i: 0.0 if i=='' or i.isspace() else float(i), read_line[7::6][::25])))
            lat_acc = np.array(list(map(lambda i: 0.0 if i=='' or i.isspace() else float(i), read_line[8::6][::25])))
            time =np.array(list(map(lambda i: 0.0 if i=='' or i.isspace() else float(i), read_line[9::6][::25])))
            data.append([track_id,type,traveled_d,avg_speed,lat,lon,speed,lon_acc,lat_acc,time])
    return data

def lon_lat_acc(a,b):
    """calculating the acceleration according to lon_acc and lat_acc

    Args:
        a (float): lon_acc
        b (float): lat_acc

    Returns:
        float: acceleration
    """
    if a >= 0:
        return np.sqrt(a**2+b**2)
    else: 
        return -np.sqrt(a**2+b**2)

def Preprocess(data,drop_lowspeed=True,drop_highspeed=True):
    """prepare the dataframe for calculating and analysing

    Args:
        data (list[list]): pneum data list
        consider_low_range : whether to consider speed < 20km/h
        drop_highspeed: bool -> whether to drop speed over 130km/h

    Returns:
        df_time: dataframe for analysing
    """
    df = pd.DataFrame(data, columns=["track_id", "type", "traveled_d", "avg_speed", 'lat','lon','speed','lon_acc','lat_acc','time'])
    #To make the dataframe indexed by time, we split the old form of the df
    df_time = df.explode('time')
    for i in ['speed','lon','lat','lon_acc','lat_acc']:
        df_time[i] = df[i].explode()

    # Adjusting the form and remove the NAN
    df_time = df_time.sort_values(by='time')
    df_time = df_time.reset_index(drop=False)
    df_time=df_time.dropna(how='any',axis=0)
    df_time.drop('index', axis=1, inplace=True)
    df_time.time = df_time.time.apply(lambda x:round(x))
    col = df_time.pop("time")
    df_time.insert(0, col.name, col)
            
    df_time['acc'] = df_time.apply(lambda x : lon_lat_acc(x.lon_acc,x.lat_acc),axis=1) 
    # get a new column for acceleration
    # select the appropriate data
    if drop_lowspeed:
        df_time = df_time[df_time.speed>=20]
    if  drop_highspeed:
        df_time = df_time[df_time.speed<=130]


    return df_time

def cal_distance(lat1,lat2_list,lon1,lon2_list):
    """calculate the distance to the nearst intersection

    Args:
        lat1 (float): position for latitute
        lat2_list (List[float]): [intersection positions]
        lon1 (float): position for lontitute
        lon2_list (List[float]): [intersection positions]

    Returns:
        float: distance
    """
    id = np.argmin((lat1 - lat2_list)**2 + (lon1 - lon2_list)**2)
    coords_1 = (lat1, lon1)
    coords_2 = (lat2_list[id], lon2_list[id])

    return geopy.distance.geodesic(coords_1, coords_2).km * 1000 

def NoiseCalculation(df_time):
    """calculating the noise emissions and construct new dataframe for documenting the noise by two models

    Args:
        df_time (_type_): _description_

    """
    df_time['distance'] = df_time.apply(lambda x:cal_distance(x.lat,la,x.lon,lo),axis=1)
    df_test_CI = df_time.copy(deep=True)
    df_test_CI['datetime'] = df_test_CI.time.apply(lambda x: datetime.datetime.fromtimestamp(int(x+1541062800.0)))
   
    a = CNOSSOS_NOISE_CAL(df_test_CI,2)
    a.conssos_cal()
    df_test_CI['LWA'] = a.LWA.copy()
    df_test_CI['LW'] = a.Lw.copy()
    df_test_CI['Lwr'] = a.Lwr.copy()
    df_test_CI['Lwp'] = a.Lwp.copy()
    df_test_CI['Lwr_acc'] = 0
    df_test_CI['Lwp_acc'] = a.acc_p.copy()
    df_test_CI['Lwp_acc'] = df_test_CI['Lwp_acc'].apply(lambda x:acoustic_average(x))

    df_test_CN = df_time.copy(deep=True)
    df_test_CN['datetime'] = df_test_CN.time.apply(lambda x: datetime.datetime.fromtimestamp(int(x+1541062800.0)))
    
    b = CNOSSOS_NOISE_CAL(df_test_CN,1)
    b.conssos_cal()
    df_test_CN['LWA'] = b.LWA.copy()
    df_test_CN['LW'] = b.Lw.copy()
    df_test_CN['Lwr'] = b.Lwr.copy()
    df_test_CN['Lwp'] = b.Lwp.copy()
    df_test_CN['Lwr_acc'] = b.acc_r.copy()
    df_test_CN['Lwp_acc'] = b.acc_p.copy()


    df_time['datetime'] = df_test_CN.time.apply(lambda x: datetime.datetime.fromtimestamp(int(x+1541062800.0)))
    
    res = ['LW63', 'LW125', 'LW250', 'LW500', 'LW1000', 'LW2000', 'LW4000', 'LW8000']
    for id,i in enumerate(res):
        df_test_CI[i] = df_test_CI.LW.apply(lambda x:x[id])
        df_test_CN[i] = df_test_CN.LW.apply(lambda x:x[id])
    

    # res = ['Lwr_acc_63', 'Lwr_acc_125', 'Lwr_acc_250', 'Lwr_acc_500', 'Lwr_acc_1000', 'Lwr_acc_2000', 'Lwr_acc_4000', 'Lwr_acc_8000']
    # for id,i in enumerate(res):
    #     df_test_CI[i] = df_test_CI.Lwr_acc # acc_rolling=0
    #     print(df_test_CN.Lwr_acc)
    #     print(df_test_CN.Lwp_acc)
    #     df_test_CN[i] = df_test_CN.Lwr_acc.apply(lambda x:x[id])
    
    # res = ['Lwp_acc_63', 'Lwp_acc_125', 'Lwp_acc_250', 'Lwp_acc_500', 'Lwp_acc_1000', 'Lwp_acc_2000', 'Lwp_acc_4000', 'Lwp_acc_8000']
    # for id,i in enumerate(res):
    #     df_test_CI[i] = df_test_CI.Lwp_acc.apply(lambda x:x[id])
    #     df_test_CN[i] = df_test_CN.Lwp_acc.apply(lambda x:x[id])

    return df_time,df_test_CI,df_test_CN

def acoustic_average(values):
    """
    Acoustic average
    :param list values: list of decibels to average
    :return: acoustic_avg
    """

    # new_values = [x for x in values if np.percentile(values, 10) < x < np.percentile(values, 90)]
    # sum_ = 0
    # for val in values:
    #     sum_ += 10 ** (0.1 * val)
    # acoustic_avg = float(10 * np.log10(sum_ / len(values)))

    values = np.array(values)
    acoustic_avg = 10*np.log10(np.sum(10**(0.1*values))/len(values))

    return acoustic_avg

# def isInRectangle(x): # right_top,right_bottom,left_top,left_bottom
#     """to determine whether the point lies inside the rectangle

#     Args:
#         x (array(latitude,lontitude)): position

#     Returns:
#         bool: True indicate the points inside the rectangle
#     """
#     AB = right_top - left_top
#     AO = x - left_top
#     CD = left_bottom - right_bottom
#     CO = x - right_bottom
#     DA = left_top - left_bottom
#     DO = x - left_bottom
#     BC = right_bottom - right_top
#     BO = x - right_top
#     return np.cross(AB,AO)*np.cross(CD,CO) >= 0 and np.cross(DA,DO)*np.cross(BC,BO) >= 0

def get_acc_phase(df_input):

    """get the dataframe with documents for different driving phase

    Returns:
        set(acceleration dataframe, deceleration dataframe, steady states driving dataframe): 
    """

    # split the acceleration time >= 3s phase
    threshold = 3
    df_combine = []
    df_left = []
    for type in [0,1,2]:
        if type == 0:
            df_in = df_input[df_input.acc>=0.44].sort_values(by=['track_id','datetime'])
        elif type == 1:
            df_in = df_input[df_input.acc<=-0.44].sort_values(by=['track_id','datetime'])
        i = 1
        pre_id = 1 # the first row id in dataframe
        pre_time = 56 # the first row time in dataframe
        mask = np.array([False]*len(df_in))

        while i < len(df_in):
            count = 1 # already consider the first row
            row = df_in.iloc[i]
            cur_id = row.track_id
            cur_time = row.time
            while cur_id == pre_id and cur_time == pre_time + 1:
                pre_time = cur_time
                count += 1
                i+= 1

                row = df_in.iloc[i]
                cur_id = row.track_id
                cur_time = row.time
            
            # document the phase
            if count >= threshold:
                mask[i-count:i] = [True] # get the slice of the phase

            # after we need to reset the pre_id and cur_id
            pre_id = cur_id
            pre_time = cur_time
            i += 1

        if type == 0 or type == 1:
            df_left.append(df_in[pd.Series([not i for i in mask],df_in.index)])
            df_result = df_in[pd.Series(mask,df_in.index)]
        else:
            df_stable = df_input[df_input.acc.between(-0.44,0.44)]
            df_result = pd.concat([df_left[0], df_left[1],df_stable])

        df_combine.append(df_result)

    return df_combine[0],df_combine[1],df_combine[2]

def drivingStatePlot(df_CI,df_CN,state):

    """
    plot the comparation for two models on different octave band
    """

    fig, axes = plt.subplots(3, 3, sharex=True, figsize=(20,16))
    fig.suptitle('plot LW distribution for different groups of vehicle ' + state +' driving state')
    res = ['LW63', 'LW125', 'LW250', 'LW500', 'LW1000', 'LW2000', 'LW4000', 'LW8000','LWA']
    for i,name in enumerate(res):
        rows = int(i/3)
        cols = i%3
        plot_title = name
        plt.title(plot_title)
        x1 = df_CI[[plot_title,'type']].groupby('type').apply(acoustic_average)
        x2 = df_CN[[plot_title,'type']].groupby('type').apply(acoustic_average)
        df_bar = pd.concat([x1,x2],axis=1).rename(columns={0: 'CI', 1: 'CN'}).reset_index().melt(id_vars=["type"])
        g = sns.barplot(data=df_bar,hue='type',x="variable", y="value",ax=axes[rows][cols])
        g.set(ylim=(70, None))
        sns.move_legend(g,"lower right")

        axes[rows][cols].set_title(name)

def map_plot(df,type = "OpenStreetMap"):
    """plot the heatmap with time

    Args:
        df (data frame): the dataframe containing [LWA,lat,lon,time] columns you want to plot
        type (str, optional): the type of map you want to use. Defaults to "OpenStreetMap".

    Returns:
        object: folum map
    """
    data = defaultdict(list)
    max_LWA = df.LWA.max()
    for r in df.itertuples():
        data[r.time].append([r.lat, r.lon,r.LWA/max_LWA])
    data = OrderedDict(sorted(data.items(), key=lambda t: t[0]))
    m = folium.Map(location=[37.991490141138, 23.732255933280655], tiles=type, zoom_start=30)
    hm = plugins.HeatMapWithTime(data=list(data.values()),index=list(data.keys()),radius=25,auto_play=True, width = '50%', height = '50%',\
        max_opacity=1,gradient={'0':'Navy', '0.25':'Blue','0.5':'Green', '0.75':'Yellow','1': 'Red'})
    hm.add_to(m)
    return m

def display_side_by_side(*args,titles=cycle([''])):
    html_str=''
    for df,title in zip(args, chain(titles,cycle(['</br>'])) ):
        html_str+='<th style="text-align:center"><td style="vertical-align:top">'
        html_str+=f'<h2 style="text-align: center;">{title}</h2>'
        html_str+=df._repr_html_().replace('table','table style="display:inline"')
        html_str+='</td></th>'
    display_html(html_str,raw=True)

def map_plot(df,type = "OpenStreetMap",single=True):
    """plot the heatmap with time

    Args:
        df (data frame): the dataframe containing [LWA,lat,lon,time] columns you want to plot
        type (str, optional): the type of map you want to use. Defaults to "OpenStreetMap".

    Returns:
        object: folum map
    """
    data = defaultdict(list)
    max_LWA = df.LWA.max()
    for r in df.itertuples():
        data[r.time].append([r.lat, r.lon,r.LWA/max_LWA])
    data = OrderedDict(sorted(data.items(), key=lambda t: t[0]))
    if single:
        fig = Figure(width=1000, height=800)
    else:
        fig = Figure(width=1000, height=300)
    m = folium.Map(location=[37.991490141138, 23.732255933280655], tiles=type, zoom_start=30)
    hm = plugins.HeatMapWithTime(data=list(data.values()),index=list(data.keys()),radius=25,auto_play=True,\
        max_opacity=1,gradient={'0':'Navy', '0.25':'Blue','0.5':'Green', '0.75':'Yellow','1': 'Red'})
    hm.add_to(m)
    fig.add_child(m)
    return m