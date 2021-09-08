
from math import inf
import numpy as np
from random import randint

from numpy.core.defchararray import not_equal
from numpy.core.numeric import zeros_like


class Node(object):
    """
    结点
    """

    def __init__(self):
        self.father = None          #父节点
        self.leftchild = None       #左子节点
        self.rightchild = None      #右子节点
        self.split_feature = None   #分离特征
        self.split_value = None     #分离点数值
        self.label = None           #标签

    def __str__(self) -> str:
        return "feature:%s ,split node:%s ,label:%s " %(str(self.split_feature),str(self.split_value),str(self.label))
        

    def brother(self):
        if not self.father:
            return None
        else :
            if self.father.leftchild is self:
                return self.father.rightchild
            else :
                return self.father.leftchild
    
class KDtree(object):
    """
    KD树
    """
    def __init__(self) :
        self.root = Node()

    def __str__(self) -> str:
        ret = []
        i = 0
        que = [(self.root,-1)] #啥意思？
        print('que',que)
        while que :
            now ,idx_father = que.pop(0)
            #print("now",now,"idx",idx_father)
            ret.append("%d -> %d:%s" %(idx_father,i,str(now)))
            if now.leftchild:
                que.append((now.leftchild,i))
            if now.rightchild:
                que.append((now.rightchild,i))
            i = i + 1
        return "\n".join(ret)

    def _get_variance(self,X,feature_num):
        '''
        计算方差,考虑多维情况

        Arguments：
            X [list] : 一行一个数据，列为特征
            feature_num [int] : 计算哪一个特征

        Return:
            方差 
        '''
        n = len(X)
        sum = 0
        sum_square = 0
        for i in range(n):
            xi = X[i][feature_num]
            #Var(x)=E(x^2)-[E(x)]^2
            sum += xi
            sum_square += xi**2
        return  sum_square/n - (sum/n)**2

    def get_median_x(self,X,y,feature_num):
        '''
        参数
            X List：

            feature_num ： 分离的特征

        输出：
            从小到大排序的X

            分离的下标

        '''
        n = len(X)
        k = n//2
        L = []                      #将数据与标签填入，格式为：[ [ data ],[ label ] ]
        for i in range(n):
            now = [X[i],y[i]]
            L.append(now)
        print('test1\n',type(L))
        minmax_feature_L = sorted(L,key=lambda x :x[:][0][feature_num])
        print('test2\n',type(minmax_feature_L))
        return minmax_feature_L,k

    def _choose_feature(self,X):
        """
        遍历所有特征，选出方差最大的那个

        Arguments:
            X [List] : 一行一个数据，列为特征

        Return:
            特征值下标
        """
        variances = map(lambda i: [ i,self._get_variance(X,i)], range(len(X[1])))
        return max(list(variances), key=lambda x: x[1])[0]
    
    def split_list(self,sorted_L,split_num,feature_num):
        '''
        分离列表

        参数
            sorted_L List：已排列的X

            split_num：分离点下标

            feature_num:分离特征下标

        输出： 
            2个子列表 : 小于放左边，>=放右边
            一个分离点 : [ [data] , [label] ]
        '''
        split_left_data = []
        split_left_label = []
        split_right_data = []
        split_right_label = []
        split_value = sorted_L[split_num][0][feature_num]
        split_item_useless = sorted_L.pop(split_num) # 排除已使用节点
        n = len(sorted_L)
        for i in range(n):
            if sorted_L[i][0][feature_num] < split_value :
                
                split_left_data.append(sorted_L[i][0])
                label = sorted_L[i][1]
                split_left_label.append(label)
            else:
                split_right_data.append(sorted_L[i][0])
                label = sorted_L[i][1]
                split_right_label.append(label)
        
        return split_left_data,split_left_label,split_right_data,split_right_label

    def build_tree(self,X,y,tree_node=None):
        '''
        递归 ，构建树
        参数：
            L [ [ data ] , [ label ] ]
            -> L[][0] --> X List ： 输入列表
            -> L[][1] --> y List :  类别

        输出：
            2个列表，左、右
        '''
        if tree_node == None:
            now = self.root #当前节点
        else:
            now = tree_node
        
        n = len(X)
        
        if n <= 1:#这里？
            now.split_value = X[0]
            now.label       = y[0]
        else :
            feature_num = self._choose_feature(X)
            now.split_feature = feature_num
            minmax_sort_L , split_num   = self.get_median_x(X,y,feature_num)#加一个y ，然后L.append( [x[],y[]] )
            now.split_value = minmax_sort_L[split_num][0]
            now.label       = minmax_sort_L[split_num][1]
            split_left_X,split_left_X_label , split_right_X,split_right_X_label  =  self.split_list(minmax_sort_L,split_num,feature_num)
            #now.split_value = split_node
            if split_left_X !=[] :
                now.leftchild = Node()
                now.leftchild.father = now  
                self.build_tree(split_left_X,split_left_X_label,now.leftchild)
            if split_right_X !=[] :
                now.rightchild = Node()
                now.rightchild.father = now  
                self.build_tree(split_right_X,split_right_X_label,now.rightchild)
        
    
    def _get_distance(self,X1,now):
        '''
        计算两点之间的距离
        
        输入：
            X : 新数据
            now : 当前节点

        输出：
            distance : 距离
        '''
        X2 = now.split_value
        return sum((x1 - x2) ** 2 for x1, x2 in zip(X1, X2)) ** 0.5

    def _get_dim_distance(self,X,now):
        '''
        计算X与分界线的距离
        '''
        return abs(X[now.split_feature ] - now.split_value[now.split_feature])

    def search_down(self,X,now = None):
        '''
        搜索
        输入：
            X : 数据
            now : 开始搜索的节点 ，若无则为根节点

        输出：
            now : 与X最接近的最底部节点
        '''
        #先将数据与最底部叶子节点匹配
        if now == None:
            now = self.root
        
        while True :
            
            if now.split_feature != None :
                if X[now.split_feature] < now.split_value[now.split_feature]:
                    if now.leftchild != None:
                        
                        now = now.leftchild
                    else:
                        return now #若无左子树，则当前节点为最底部
                else:
                    if now.rightchild != None:
                        
                        now = now.rightchild
                    else:
                        break
            else:
                break   
        return now     #已获取与数据最近的底部叶子节点
        
    def search_up(self,X,stop_node = None , k=1 , max_distance = np.inf ):
        '''
        先给定节点，向下搜索至底部，再向上搜索
        输入：
            X ： 搜索数据            
            stop_node : 结束节点
            k : k个最近邻
        
        输出：
            list [ [data] , [label] , distance ] :  包含k个最近邻的列表
        '''
        #设置从哪个节点开始搜索
        if stop_node == None:
            stop_node = self.root
         
        now = self.search_down(X,stop_node) #搜索到最底部节点，从下往上搜索，逐个计算距离       
        distance = float(self._get_distance(X,now))

        list_k = [] #存放k个最近点 [ [data] , [label] , distance ]  ,dis从大到小排列  
        list_k.insert(0, [now.split_value , now.label , distance] ) #将最底部的节点存入
        if distance < max_distance :
            max_distance = distance
        
        ii = 0
        Running = True
        while Running:
            ii = ii+1
            len_list_k = len(list_k)
            if now != stop_node: 
                last_now = now  #保存节点
                now = now.father    #去往父节点  

                if len_list_k < k : #如果 list k 不满，则填入                    
                    distance = float(self._get_distance(X,now))
                    index = 0   #距离大小序号
                    
                    for i in range(len_list_k):
                        if distance < list_k[i][2] : #当前距离较小，index+1
                            index += 1
                        if distance > list_k[i][2] : #当前距离大的时候
                            break
                    #插入新节点
                    list_k.insert(index, [now.split_value , now.label , distance] )
                    
                    max_distance = list_k[0][2]
                   
                else :  #如果满了，计算节点与X距离
                    t_distance = float(self._get_distance(X,now)) #计算X与节点距离
                    if t_distance < max_distance :
                        index = 0   #距离大小序号
                        
                        for i in range(len_list_k):
                            if t_distance < list_k[i][2] : #当前距离较小，index+1
                                index += 1
                            else: #当前距离小的时候
                                break
                    #插入新节点
                    list_k.insert(index, [now.split_value , now.label , t_distance] )
                    len_list_k = len(list_k)
                    if len_list_k > k : 
                        list_k.pop(0)
                    max_distance = list_k[0][2]

                #如果X与分界线的距离小于max_distance               
                if float(self._get_dim_distance(X,now)) < max_distance:
                    a = []                
                    if now.leftchild ==  last_now and now.rightchild != None:
                        next_now = now.rightchild
                        a = self.search_up(X,stop_node = next_now,k=k,max_distance=max_distance)
                    elif now.rightchild ==  last_now and now.leftchild != None:
                        next_now = now.leftchild
                        a = self.search_up(X,stop_node = next_now,k=k,max_distance=max_distance)
                    if a !=[]:
                        for a_x in a:                      
                            x = a_x[0]
                            index = 0   #距离大小序号
                            distance = float(sum((X - x) ** 2 for X, x in zip(X, x)) ** 0.5)#直接求距离
                            if distance < max_distance:
                                for i in range(len_list_k):
                                    if distance < list_k[i][2] : #当前距离较小，index+1
                                        index += 1
                                    else :  #当前距离大的时候
                                        break
                                
                                list_k.insert(index, [a_x[0] , a_x[1] , distance] )
                                len_list_k = len(list_k)
                                if len_list_k > k : 
                                    list_k.pop(0)
                        max_distance = list_k[0][2]
            #退出        
            if now == stop_node : 
                Running = False                            
        return list_k
                

    def search(self,X,k=1):
        '''
        搜索函数
        输入：
            X ： 搜索数据
            k ： 最近邻个数

        输出：
            label ： 预测类别
        '''
        listk = self.search_up(X,k=k)
        label_list = []
        for  label_x in listk:
            label = label_x[1]
            label_list.append(label)
        label_decision = max(label_list,key=label_list.count) # 决定类别
        return label_decision
    

def gen_data(low, high, n_rows, n_cols=1):
    '''
    随机生成数据
    '''
    ret = [[randint(low, high) for _ in range(n_cols)]
               for _ in range(n_rows)]
    return ret

if __name__ == '__main__':
    min_num = 0
    max_num = 100
    n_rows = 5
    n_cols = 3
    #X = gen_data(min_num, max_num, n_rows, n_cols)
    #y = gen_data(min_num, max_num, n_rows)
    #X = [[50,50],[2,2],[99,99],[111,111],[200,200] ]
    X = []
    y = [1,0,1,1,1]
    print('x',np.array(X))
    print('y',np.array(y))
     
    tree = KDtree()
    tree.build_tree(X,y)
    print('tree',tree)
    listk = tree.search([99,99],2)
    print('X label is :',listk)
    
    























