import torch
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import math

class UNSW_NB15_quantized(torch.utils.data.Dataset):
    def __init__(self, file_path_train, file_path_test, quantization=True, onehot=False, train=True, sequence_length=25, transform=None):   
        self.dataframe = pd.concat([pd.read_csv(file_path_train), pd.read_csv(file_path_test)]).reset_index().drop(columns = ['index'])
        self.transform = transform
        self.sequence_length = sequence_length
        
        self.categorical_column_values = {"proto":None, "state":None, "service":None, "attack_cat":None}

        #load all the unique values of categorical features at the start
        #and make these accessible via a fast function call.
        for key in self.categorical_column_values:
            self.categorical_column_values[key] = self.dataframe[key].unique()
        
        #------------------------------------------------
        self.dataframe = self.dataframe.drop(['id', 'attack_cat'],1)   
        
        # -------------APPLY 1HOT ENCODING and Quantization---------
        self.to_be_tensor_df = self.dataframe
        if onehot :
            self.to_be_tensor_df = self.one_hot_encoding(self.to_be_tensor_df)
        if quantization:            
            _, self.to_be_tensor_train, self.to_be_tensor_test = self.quantize_df(self.to_be_tensor_df)        
        
        if train:
            self.data = torch.FloatTensor(self.to_be_tensor_train.astype('float')) #create tensor
        else:
            self.data = torch.FloatTensor(self.to_be_tensor_test.astype('float')) #create tensor
            
        print(self.data.shape)
       
    
    def get_dataframe(self):
        return self.dataframe
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        target = self.data[index][-1]
        data_val = self.data[index][:-1]
        return data_val,target
    
    #get a list of all the unique labels in the dataset
    def get_labels(self):
        return self.dataframe['label'].unique().tolist()
    
    #get a list of all the unique attack categories in the dataset
    def get_attack_categories(self):
        return self.dataframe['attack_cat'].unique().tolist()
    
    def get_list_of_categories(self, column_name):
        pass #TODO

    #limit the dataset to only examples in the specified category
    def use_only_category(self, category_name):
        if category_name not in self.get_attack_categories():
            return False
        
        new_dataframe = self.dataframe[self.dataframe['attack_cat'] == category_name]
        new_dataframe = new_dataframe.reset_index()
        self.dataframe = new_dataframe
        return True
    
    #limit the dataset to only examples with the specified label
    def use_only_label(self, label):
        if label not in self.get_labels():
            return False
        
        new_dataframe = self.dataframe[self.dataframe['label'] == label]
        new_dataframe = new_dataframe.reset_index()
        self.dataframe = new_dataframe
        return True
    
    def dec2bin(self,column: pd.Series, number_of_bits: int, left_msb:bool= True )-> pd.Series: 
        """Convert a decimal pd.Series to binary pd.Series with numbers in their base-2 equivalents.
        The output is a numpy nd array.   
        # adapted from here https://stackoverflow.com/questions/51471097/vectorized-conversion-of-decimal-integer-array-to-binary-array-in-numpy
        Parameters
        ----------
         column: pd.Series
            Series wit all decimal numbers that will be cast to binary
         number_of_bits: str
            The desired number of bits for the binary number. If bigger than what is needed then those bits will be 0.
            The number_of_bits should be >= than what is needed to express the largest decimal input 
         left_msb: bool
            Specify that the most significant digit is the leftmost element. If this is False, it will be the rightmost element.
        Returns
        -------
        numpy.ndarray
           Numpy array with all elements in binary representation of the input.

        """
        my_binary_repr = lambda number, nbits:  np.binary_repr(number, nbits)[::-1]
        func = my_binary_repr if left_msb else np.binary_repr

        return np.vectorize(func)(column.values, number_of_bits)
    
    def round_like_matlab_number(self, n:np.float64)->int:
        """Round the input "n" like matlab uint32(n) cast (which also rounds) 0.5->1;  1.5->2; 2.3->2;   2.45->2 """
        if n - math.floor(n) < 0.5:
            return math.floor(n)
        return math.ceil(n)

    def round_like_matlab_series(self, series:pd.Series)->pd.Series:
        rounded_values_list = []
        for value in series:
            rounded_values_list.append(self.round_like_matlab_number(value))
        return pd.Series(rounded_values_list) 

    def integer_encoding(self, df):
        """Applies integer encoding to the object columns of the dataframe"""
        le = preprocessing.LabelEncoder()
        for column in df.select_dtypes('object').columns.tolist():
            df[column] = le.fit_transform(df[column])
        return df

    def quantize_df(self, df):
        """Quantized the input dataframe. The scaling is done by multiplying every column by the inverse of the minimum of that column"""
        # gets the smallest positive number of a vector
        get_min_positive_number = lambda vector: vector[vector > 0].min()
        # computes the maximum required bits necessary to represent each number from a vector of numbers
        get_max_bits = lambda vector: math.ceil(math.log2(float(vector.max())+1.0))
        # splits a string into a list of all characters
        char_split = lambda s: np.array([ch for ch in s]) 

        df_encoded = self.integer_encoding(df)
        python_quantized_df = df_encoded.copy()
        dict_correct_rate_values= {715:34716, 11691:25278, 27417:5259117, 45319:60744, 73620:9039, 74498:15070, 86933:1024485,
 89021:1689027, 90272:5259117, 103372:1562102, 118192:1759777, 122489:246327, 159266:18853, 190473:18423}

        for column in python_quantized_df.columns:       
            column_data = df_encoded[column]

            m = get_min_positive_number(column_data)
            m_inv = 1.0/m    
            if m_inv>1:
                column_data = column_data * np.float64(m_inv)
   
            maxbits = get_max_bits(column_data) 
            #CLIP, ROUND and CAST to UINT32
            column_data = np.clip(column_data, 0, 4294967295)         #clip due to overflow of uint32 of matlab code
            column_data = self.round_like_matlab_series(column_data)       #round like matlab
            column_data = column_data.astype(np.uint32)               #cast like matlab
            
            if column == "rate":
                column_data.update(pd.Series(dict_correct_rate_values))
        
            python_quantized_df[column] = self.dec2bin(column_data, maxbits, left_msb=False).reshape((-1,1)).flatten()
               
        for column in python_quantized_df.columns:
            python_quantized_df[column] = python_quantized_df[column].apply(char_split).values
       
        python_quantized_df_separated = pd.DataFrame(np.column_stack(python_quantized_df.values.T.tolist()))
        python_train = python_quantized_df_separated.iloc[:175341]
        python_test  = python_quantized_df_separated.iloc[175341:]

        return python_quantized_df_separated.values, python_train.values, python_test.values
    
    def one_hot_encoding(self, df):
        dataframe = df.copy()
        """Applies 1 hot encoding to the proto, service and state columns """

        string_columns= ["proto","service","state"]
        string_categories= [[['tcp', 'udp', 'arp', 'ospf', 'icmp', 'igmp', 'rtp', 'ddp',
           'ipv6-frag', 'cftp', 'wsn', 'pvp', 'wb-expak', 'mtp', 'pri-enc',
           'sat-mon', 'cphb', 'sun-nd', 'iso-ip', 'xtp', 'il', 'unas',
           'mfe-nsp', '3pc', 'ipv6-route', 'idrp', 'bna', 'swipe',
           'kryptolan', 'cpnx', 'rsvp', 'wb-mon', 'vmtp', 'ib', 'dgp',
           'eigrp', 'ax.25', 'gmtp', 'pnni', 'sep', 'pgm', 'idpr-cmtp',
           'zero', 'rvd', 'mobile', 'narp', 'fc', 'pipe', 'ipcomp', 'ipv6-no',
           'sat-expak', 'ipv6-opts', 'snp', 'ipcv', 'br-sat-mon', 'ttp',
           'tcf', 'nsfnet-igp', 'sprite-rpc', 'aes-sp3-d', 'sccopmce', 'sctp',
           'qnx', 'scps', 'etherip', 'aris', 'pim', 'compaq-peer', 'vrrp',
           'iatp', 'stp', 'l2tp', 'srp', 'sm', 'isis', 'smp', 'fire', 'ptp',
           'crtp', 'sps', 'merit-inp', 'idpr', 'skip', 'any', 'larp', 'ipip',
           'micp', 'encap', 'ifmp', 'tp++', 'a/n', 'ipv6', 'i-nlsp',
           'ipx-n-ip', 'sdrp', 'tlsp', 'gre', 'mhrp', 'ddx', 'ippc', 'visa',
           'secure-vmtp', 'uti', 'vines', 'crudp', 'iplt', 'ggp', 'ip',
           'ipnip', 'st2', 'argus', 'bbn-rcc', 'egp', 'emcon', 'igp', 'nvp',
           'pup', 'xnet', 'chaos', 'mux', 'dcn', 'hmp', 'prm', 'trunk-1',
           'xns-idp', 'leaf-1', 'leaf-2', 'rdp', 'irtp', 'iso-tp4', 'netblt',
           'trunk-2', 'cbt']],[['-', 'ftp', 'smtp', 'snmp', 'http', 'ftp-data', 'dns', 'ssh',
           'radius', 'pop3', 'dhcp', 'ssl', 'irc']],[['FIN', 'INT', 'CON', 'ECO', 'REQ', 'RST', 'PAR', 'URN', 'no',
           'ACC', 'CLO']]]
    

        for column, categories in zip(string_columns, string_categories):       
            column_df = dataframe.loc[:, [column]]

            one_hot_encoder = OneHotEncoder(sparse=False, categories = categories)
            # Fit OneHotEncoder to dataframe
            one_hot_encoder.fit(column_df)  
            # Transform the dataframe
            column_df_encoded = one_hot_encoder.transform(column_df)
            #Create dataframe from the 2-d array
            column_df_encoded = pd.DataFrame(data=column_df_encoded, columns=one_hot_encoder.categories_[0])
            dataframe = pd.concat([column_df_encoded, dataframe], axis=1, sort=False)

        #delete proto,service and state columns
        dataframe = dataframe.drop(string_columns,1)

        return dataframe
    
    def one_hot_encoding_select_categ(self, df):
        dataframe = df.copy()
        """Applies 1 hot encoding to the proto, service and state columns but to some selected categories which are more influetial acording to seaborn countplot"""

        string_columns= ["proto","service","state"]
        string_categories= [[['tcp', 'udp', 'arp', 'ospf']],[['-', 'ftp', 'smtp', 'snmp', 'http', 'ftp-data', 'dns', 'ssh']],[['FIN', 'INT', 'CON', 'ECO', 'REQ']]]
    

        for column, categories in zip(string_columns, string_categories):       
            column_df = dataframe.loc[:, [column]]

            one_hot_encoder = OneHotEncoder(sparse=False, categories = categories,handle_unknown='ignore')
            # Fit OneHotEncoder to dataframe
            one_hot_encoder.fit(column_df)  
            # Transform the dataframe
            column_df_encoded = one_hot_encoder.transform(column_df)
            #Create dataframe from the 2-d array
            column_df_encoded = pd.DataFrame(data=column_df_encoded, columns=one_hot_encoder.categories_[0])
            dataframe = pd.concat([column_df_encoded, dataframe], axis=1, sort=False)

        #delete proto,service and state columns
        dataframe = dataframe.drop(string_columns,1)

        return dataframe