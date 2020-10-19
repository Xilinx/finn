import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

class UNSW_NB15(torch.utils.data.Dataset):
    def __init__(self, file_path, sequence_length=25, transform=None):
        #TODO have a sequence_overlap=True flag? Does overlap matter?
        self.transform = transform
        self.sequence_length = sequence_length
        self.columns = ['id', 'dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 'sbytes',
       'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss',
       'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin',
       'tcprtt', 'synack', 'ackdat', 'smean', 'dmean', 'trans_depth',
       'response_body_len', 'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm',
       'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm',
       'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm',
       'ct_srv_dst', 'is_sm_ips_ports', 'attack_cat', 'label']
        self.dtypes = dtypes = {"id":"int32",
                                "scrip": "object",
                                #"sport": "int32",
                                "dstip": "object",
                                #"dsport": "int32",
                                "proto": "object",
                                "state": "object",
                                "dur": "float64",
                                "sbytes": "int32",
                                "dbytes": "int32",
                                "sttl": "int32",
                                "dttl": "int32",
                                "sloss": "int32",
                                "dloss": "int32",
                                "service": "object",
                                "sload": "float64",
                                "dload": "float64",
                                "spkts": "int32",
                                "dpkts": "int32",
                                "swin": "int32",
                                "dwin": "int32",
                                "stcpb": "int32",
                                "dtcpb": "int32", 
                                #"smeansz": "int32",
                                #"dmeansz": "int32",
                                "trans_depth": "int32",
                                #"res_bdy_len": "int32",
                                "sjit": "float64",
                                "djit": "float64",
                                #"stime": "int64",
                                #"ltime": "int64",
                                #"sintpkt": "float64",
                                #"dintpkt": "float64",
                                "tcprtt": "float64",
                                "synack": "float64",
                                "ackdat": "float64",

                                #commenting these because they have mixed values and we aren't going to generate them anyway
                                #"is_sm_ips_ports": "int32",
                                #"ct_state_ttl": "int32",
                                #"ct_flw_httpd_mthd": "int32",
                                #"is_ftp_login": "int32",
                                #"is_ftp_cmd": "int32",
                                #"ct_ftp_cmd": "int32",
                                #"ct_srv_src": "int32",
                                ##"ct_dst_ltm": "int32", 
                                #"ct_src_ltm": "int32",
                                #"ct_src_dport_ltm": "int32",
                                #"ct_dst_sport_ltm": "int32",
                                #"ct_dst_src_ltm": "int32",
                                "attack_cat": "object",
                                "label": "int32"}
        self.categorical_column_values = {"proto":None, "state":None, "service":None, "attack_cat":None}

        self.dataframe = pd.read_csv(file_path, encoding="latin-1", names=self.columns,header=0, dtype=self.dtypes)
        #self.dataframe.sort_values(by=['stime']) #sort chronologically upon loading
        
        #load all the unique values of categorical features at the start
        #and make these accessible via a fast function call.
        for key in self.categorical_column_values:
            self.categorical_column_values[key] = self.dataframe[key].unique()

        #cache all the maximum values in numeric columns since we'll be using these for feature extraction
        self.maximums = {}
        for key in self.dtypes:
            if "int" in self.dtypes[key] or "float" in self.dtypes[key]:
                self.maximums[key] = max(self.dataframe[key])
        
        #------------------------------------------------
        self.dataframe = self.dataframe.drop(['id', 'attack_cat'],1)   
        
        # -------------APPLY 1HOT ENCODING
        self.one_hot_encoded_df = self.dataframe
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
            
            column_df = self.one_hot_encoded_df.loc[:, [column]]

            one_hot_encoder = OneHotEncoder(sparse=False, categories = categories)
            # Fit OneHotEncoder to dataframe
            one_hot_encoder.fit(column_df)  
            # Transform the dataframe
            column_df_encoded = one_hot_encoder.transform(column_df)
            #Create dataframe from the 2-d array
            column_df_encoded = pd.DataFrame(data=column_df_encoded, columns=one_hot_encoder.categories_[0])
            self.one_hot_encoded_df = pd.concat([column_df_encoded,self.one_hot_encoded_df], axis=1, sort=False)

        #delete proto,service and state columns
        self.one_hot_encoded_df = self.one_hot_encoded_df.drop(string_columns,1)
        #create tensor
        self.data = torch.FloatTensor(self.one_hot_encoded_df.values.astype('float'))

        print(self.data.shape)
       
    
    def get_dataframe(self):
        return self.dataframe
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        target = self.data[index][-1]
        data_val = self.data[index] [:-1]
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
    def one_hot_encoding_df(self):
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
            column_df = self.dataframe.loc[:, [column]]

            one_hot_encoder = OneHotEncoder(sparse=False, categories = categories)
            # Fit OneHotEncoder to dataframe
            one_hot_encoder.fit(column_df)  
            # Transform the dataframe
            column_df_encoded = one_hot_encoder.transform(column_df)
            #Create dataframe from the 2-d array
            column_df_encoded = pd.DataFrame(data=column_df_encoded, columns=one_hot_encoder.categories_[0])
            self.dataframe = pd.concat([column_df_encoded, self.dataframe], axis=1, sort=False)

        #delete proto,service and state columns
        self.dataframe = self.dataframe.drop(string_columns,1)

        return self.dataframe