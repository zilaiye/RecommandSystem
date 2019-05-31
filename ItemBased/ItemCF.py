# -*- coding: UTF-8 -*-
# add by Kylin 
import pandas as pd 
import numpy as np 

class ItemCF:
    """this is a class tend to do item_based recommandation
        df is a dataframe with three columns , user_id , item_id and rating 
        column_map is a dict , 
        which map your dataframe to a standard user_id , item_id , rating dataframe 
        you must specify the user_id column ,item_id column , rating column 
        all you need is just put into a dataframe and specify the column relations 
    """

    def __init__(self,uir_df,similarity="adjusted_cosine",column_map={'user_id':'user_id','item_id':'item_id','rating':'rating'}):
        """firstly we need a dataframe , which is you user_item_rating dataframe """
        self.df = uir_df.copy()
        self.column_map = column_map
        self.check_data()
        self.column_map = {}
        self.n_users = 0 
        self.n_items = 0 
        self.user_df = pd.DataFrame()
        self.item_df = pd.DataFrame()
        self.matrix = None 
        self.sims = None 
        self.similarity = similarity
        self.uid_to_uiid = {}
        self.iid_to_iiid = {}
        self.iiid_to_iid = {}
        self.uiid_to_uid = {} 
        self.result_list = [] 

    def check_data(self):
        if  self.df.shape[0] < 1 or self.df.shape[1] < 1  :
            raise Exception("the input uir_df must have at least 2 row and 2 column ,but now with shape = ", self.df.shape)

    def assign_matrix(self,r):
        """
            构建完整的评分矩阵
        """
        i = int(r['uiid']) 
        j = int(r['iiid'])
        self.matrix[i,j] = r['rating']
        return         

    def preprocessing(self):
        self.df.rename(columns=self.column_map,inplace=True)
        users = self.df['user_id'].unique() 
        items = self.df['item_id'].unique() 
        self.n_users = len(users)
        self.n_items = len(items)
        self.user_df = pd.DataFrame( data={'user_id':users,'uiid':range(len(users))} )
        self.item_df = pd.DataFrame( data={'item_id':items,'iiid':range(len(items))} )

        # dataframe 的值进行替换
        df2 = pd.merge(self.df,self.user_df,on='user_id')
        df3 = pd.merge(df2,self.item_df,on='item_id') 
        self.df = df3[['uiid','iiid','rating']]
        self.matrix = np.zeros(shape=[self.n_users,self.n_items])
        self.sims = np.zeros(shape=[self.n_items,self.n_items])
        dummy = self.df.apply(lambda r: self.assign_matrix(r),axis=1)
        dummy = self.user_df.apply(lambda x : self.uid_to_uiid.update({x['user_id']:x['uiid']}),axis=1)
        dummp = self.user_df.apply(lambda x : self.uiid_to_uid.update({x['uiid']:x['user_id']}),axis=1)
        dummp = self.item_df.apply(lambda x : self.iid_to_iiid.update({x['item_id']:x['iiid']}),axis=1)
        dummp = self.item_df.apply(lambda x : self.iiid_to_iid.update({x['iiid']:x['item_id']}),axis=1)

    def cosin_dis(self,x,y):
        div = (np.linalg.norm(x) * np.linalg.norm(y))
        if div == 0 : 
            return 0 
        else:
            return np.dot(x, y) / div

    def compute_similarity(self):
        if self.similarity == 'cosine' :
            matrix_t = self.matrix.T
            for i in range(self.n_items):
                for j in range(i,self.n_items):
                    if i == j :
                        self.sims[i,j] = 1 
                    else:
                        self.sims[j,i] = self.cosin_dis(matrix_t[j],matrix_t[i])
                        self.sims[i,j] = self.sims[j,i]
        elif self.similarity == "adjusted_cosine":
            #adjusted cosine 获取每个用户的均值
            user_mean =  self.df.groupby('uiid',as_index=True)['rating'].mean().rename(columns={'rating':'avg_rating'})
            #现在开始减去均值获取矩阵
            matrix_adjusted = self.matrix.copy()
            for j in range(matrix_adjusted.shape[0]):
                arr = matrix_adjusted[j]
                mean_rating = user_mean[j]
                arr = np.where( arr > 0 , arr - mean_rating , 0 ) 
                matrix_adjusted[j] = arr
            #这里完成了均值的调整
            matrix_t = matrix_adjusted.T

            for i in range(self.n_items):
                for j in range(i,self.n_items):
                    if i == j :
                        self.sims[i,j] = 1 
                    else:
                        self.sims[j,i] = self.cosin_dis(matrix_t[j],matrix_t[i])
                        self.sims[i,j] = self.sims[j,i]
        else:
            raise Exception("{} distance is not supported !" , self.similarity)

    def train(self):
        self.preprocessing()
        self.compute_similarity()

    def predict(self):
        result = np.dot(self.matrix,self.sims) 
        result_l = []
        # iiid 转化成初步的评分
        for i in range(self.n_users):
            for j in range(self.n_items):
                result_l.append( (self.uiid_to_uid[i],self.iiid_to_iid[j],result[i,j]) )
        self.result_list = result_l

    def save(self,path):
        df = pd.DataFrame(data = self.result_list , columns = ['user_id','item_id','score'])
        df.to_csv(path,sep = ',', header=True,index=None)
