import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

class UNSW_NB15(torch.utils.data.Dataset):
    def __init__(self, file_path, sequence_length=25, transform=None, onehot=True):   
        self.dataframe = pd.read_csv(file_path)
        self.transform = transform
        self.sequence_length = sequence_length
        
        self.categorical_column_values = {"proto":None, "state":None, "service":None, "attack_cat":None}

        #load all the unique values of categorical features at the start
        #and make these accessible via a fast function call.
        for key in self.categorical_column_values:
            self.categorical_column_values[key] = self.dataframe[key].unique()
        
        #------------------------------------------------
        self.dataframe = self.dataframe.drop(['id', 'attack_cat'],1)   
        
        # -------------APPLY 1HOT ENCODING---------
        if onehot:
            #self.one_hot_encoded_df = self.one_hot_encoding(self.dataframe)
            self.one_hot_encoded_df = self.one_hot_encoding_select_categ(self.dataframe)
        else:
            #self.dataframe = self.dataframe.drop(["proto","service","state"],1)
            self.one_hot_encoded_df = pd.DataFrame()
                
        #normalize df
        #self.one_hot_encoded_df.apply(lambda x: x/x.max(), axis=0)
         
        self.data = torch.FloatTensor(self.one_hot_encoded_df.values.astype('float')) #create tensor
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
    
    

    def integer_encoding(self):
        """Applies integer encoding to the proto, service and state columns of the dataframe"""
        le = preprocessing.LabelEncoder()

        #self.dataframe['attack_cat'] = le.fit_transform(self.dataframe['attack_cat'])
        self.dataframe['proto'] = le.fit_transform(self.dataframe['proto'])
        self.dataframe['service'] = le.fit_transform(self.dataframe['service'])
        self.dataframe['state'] = le.fit_transform(self.dataframe['state']) 
        return self.dataframe
    
    # Apply 1 hot encoding to the dataframe
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