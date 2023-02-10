# in the following class, we provide a way to calculate the noise of vehicle by CONSSOS
import numpy as np
import pandas as pd

class CNOSSOS_NOISE_CAL:
    def __init__(self,df,acc_term_type) -> None:
        """we provide a way to calculate the noise

        Args:
            df : DataFrame Object arranged by time
            acc_term_type : int, 0 refers to no acceleration term concerned, 1 for considering distance to intersection, 2 for consider Imagine acc term and facotr
        """

        self.speed = df.speed.apply(lambda x:x if x>0.01 else 0.01) # noise calculation in log
        self.vehicle_type = df.type
        self.A_factor = np.array([-26.2,-16.1,-8.6,-3.2,0,1.2,1,-1.1]) ## A factor for 8 octave band from 63HZ to 8000HZ
        self.band = 8 # number of octave band
        self.v_ref = 70.0 # reference speed for calculating noise
        self.acc_term_type = acc_term_type
        # initialize the acceleration term 
        if self.acc_term_type == 0:
            pass
        elif self.acc_term_type == 1:
            self.distance = df.distance
            self.acc_param_crossing = [{1: -4.5, 2: -4.0, 3: -4.0,4: -4.5},{1: 5.5, 2: 9.0, 3: 9.0,4: 5.5}] # CR and CP respectivly
            self.acc_param_roundabout = [{1: -4.4, 2: -2.3, 3: -2.3,4: -4.4},{1: 3.1, 2: 6.7, 3: 6.7,4: 3.1}]

        elif self.acc_term_type == 2:
            self.acc = df.acc
            self.Cp = {
            1:np.array([5,5,2,2,2,2,2,2]),
            2:np.array([7,7,3,3,3,3,3,3]),
            3:np.array([7,7,3,3,3,3,3,3]),
            4:np.array([5,5,2,2,2,2,2,2])
        }

        self.type_to_category = {
            'Motorcycle': 4,
            'Taxi' : 1, 
            'Car' : 1, 
            'Heavy Vehicle' : 3, 
            'Medium Vehicle' : 2,
            'Bus' : 3 # TODO
        }
        # key is the vehicle type name, value is the corresponding type category

        self.sound_power_factor = {
            1 : np.array([[79.7,	85.7,	84.5,	90.2,	97.3,	93.9,	84.1,	74.3], # Ar
                [30,	41.5,	38.9,	25.7,	32.5,	37.2,	39,	40], # Br
                [94.5,	89.2,	88,	85.9,	84.2,	86.9,	83.3,	76.1], # Ap
                [-1.3,	7.2,	7.7,	8,	8,	8,	8,	8]]) # Bp
                ,
            
            2 :np.array([[84,	88.7,	91.5,	96.7,	97.4,	90.9,	83.8,	80.5],
                [30,	35.8,	32.6,	23.8,	30.1,	36.2,	38.3,	40.1],
                [101,	96.5,	98.8,	96.8,	98.6,	95.2,	88.8,	82.7],
                [-1.9,	4.7,	6.4,	6.5,	6.5,	6.5,	6.5,	6.5]]),
            
            3 : np.array([[87,	91.7,	94.1,	100.7,	100.8,	94.3,	87.1,	82.5],
                [30,	33.5,	31.3,	25.4,	31.8,	37.1,	38.6,	40.6],
                [104.4,	100.6,	101.7,	101	,100.1,	95.9,	91.3,	85.3],
                [0,	3,	4.6,	5,	5,	5,	5,	5]]),
            
            4 : np.array([[0,	0,	0,	0,	0,	0,	0,	0],
                [0,	0,	0,	0,	0,	0,	0,	0],
                [88,	87.5,	89.5,	93.7,	96.6,	98.8,	93.9,	88.7],
                [4.2,	7.4,	9.8,	11.6,	15.7,	18.9,	20.3,	20.6]])
        }
        # key is the vehicle type, value is the list of the rolling noise and propulsion noise factor
        
        self.vehicle_type = self.vehicle_type.replace(self.type_to_category)
        self.Lwr = None
        self.Lwp = None
        self.Lw = None
        self.LWA = None
        self.LWA_eq = None
        self.acc_r = None
        self.acc_p = None
        self.acc_contribution_Lw = None
    
    def conssos_cal(self):
        
        self.Lwr = self.cal_lwr()
        self.Lwr = self.Lwr.apply(lambda x: np.array([0 if a < 0  else a for a in x]))  
        # ensure rolling noise > 0
        self.Lwp = self.cal_lwp()
        
        self.acc_contribution_Lw = self.cal_lw()
        if self.acc_term_type == 1: # cnossos
            self.acc_r,self.acc_p = self.cal_acc_cno()
            self.Lwr += self.acc_r
            self.Lwp += self.acc_p
            

        elif self.acc_term_type == 2: # imagine
            self.acc_p = self.cal_acc_Img_propulsion()
            self.Lwp += self.acc_p

        self.Lw = self.cal_lw()
        self.acc_contribution_Lw = self.Lw - self.acc_contribution_Lw
        self.LWA = self.cal_LWA()
        # self.LWA_eq = self.cal_LWA_eq()

    def cal_lwr(self):
        return self.vehicle_type.apply(lambda x:self.sound_power_factor[x][0]) +\
               self.vehicle_type.apply(lambda x:self.sound_power_factor[x][1])*np.log10(((self.speed)/self.v_ref).astype(float)) # TODO:enlarge speed range
        # N*1 1->np.array of length 8 : [[1...8],[1...8],...,[1...8]]
    
    def cal_lwp(self):
        return self.vehicle_type.apply(lambda x:self.sound_power_factor[x][2]) +\
               self.vehicle_type.apply(lambda x:self.sound_power_factor[x][3])*((self.speed-self.v_ref)/self.v_ref)
    
    def cal_lw(self):
        return 10*(((10**(self.Lwr/10) + 10**(self.Lwp/10)).apply(lambda x: np.log10(x))))
    
    def cal_LWA(self):
        return self.Lw.apply(lambda x:10**((x + self.A_factor)/10)).apply(lambda x:10*(np.log10(x.sum()))) # N*1
    
    # def cal_LWA_eq(self):
    #     return 10 * ((10**(self.LWA/10)).sum()/max(df.time).apply(lambda x: np.log10(x)))

    def cal_acc_cno(self):
        param_cr = self.acc_param_crossing[0]
        param_cp = self.acc_param_crossing[1]
        return (self.vehicle_type.apply(lambda x:param_cr[x]) *(1-abs(self.distance)/100).apply(lambda x:x if x>0 else 0),\
             self.vehicle_type.apply(lambda x:param_cp[x]) * (1-abs(self.distance)/100).apply(lambda x:x if x>0 else 0))
    
    def cal_acc_Img_propulsion(self): 

        # self.vehicle_type
        def cal_a(a,type):
            dcit = {1:[-2,2],2:[-1,1],3:[-1,1],4:[-4,4]}
            min_a = dcit[type][0]
            max_a = dcit[type][1]
            if a < min_a : return -1
            if a > max_a: return max_a
            if a >= -1: return a
            else: return -1

        return pd.DataFrame({'acc':self.acc,'type':self.vehicle_type}).apply(lambda x:cal_a(x.acc,x.type),axis=1) *\
               self.vehicle_type.apply(lambda x:self.Cp[x]) 
        
        # return self.acc.apply(lambda a: a if a>=-1 and a<=1 else(0 if a>1 else -1)) * self.vehicle_type.apply(lambda x:self.Cp[x]) 
        # return a list of acc term for propulsion noise
