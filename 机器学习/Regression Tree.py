from math import inf
import numpy as np
from numpy.core.fromnumeric import nonzero
from numpy.lib.function_base import append


class Node(object):
    def __init__(self):
        self.parent = None  #父节点
        self.leftchild = None
        self.rightchild = None
        
        self.feature_num = None   #特征
        self.feature_value = None #特征值
        self.label = None   #y值

    def __str__(self) -> str:
        return "feature:%s ,split node:%s ,mean:%s " %(str(self.feature_num),str(self.feature_value),str(self.label))
     

class RegressionTree(object):
    def __init__(self) -> None:
        self.root = Node()

    def __str__(self) -> str:
        '''
        树的形状
        '''
        ret = []
        i = 0
        que = [(self.root,-1)] #啥意思？
        print('que',que)
        while que :
            now ,idx_father = que.pop(0)
            ret.append("%d -> %d:%s" %(idx_father,i,str(now)))
            if now.leftchild:
                que.append((now.leftchild,i))
            if now.rightchild:
                que.append((now.rightchild,i))
            i = i + 1
        return "\n".join(ret)

    def _loss_func(self,sorted_X,X_id,feature_id):
        '''
        损失函数
        输入：
            

        输出：
            loss：损失函数值
        '''
        
        '''
        最小二乘法
        切分点左边，切分点右边
        E（yi左-c1)^2 + E(yi右-c2)^2
        c = E yi / N
        '''
        #直接填入值
        
        left_Y,right_Y = [],[]
        n = len(sorted_X)
        
        for i in range(n):
            if  sorted_X[i,feature_id]  <= sorted_X[X_id,feature_id]:
                
                left_Y.append(sorted_X[i,-1])
            else:
                
                right_Y.append(sorted_X[i,-1])
        c1 = np.sum(left_Y) / len(left_Y)
        c2 = np.sum(right_Y) / len(right_Y)
        loss = np.sum((left_Y-c1)**2) + np.sum((right_Y-c2)**2)

        return loss
    
    

    def choose_feature(self,X):
        '''
        遍历所有特征，选择最佳切分点
        输入：
            X ：数据,最后一列为标签

        输出：
            best_x_id: X的下标
            best_feature:特征下标
            best_sorted_X : 排序好的X
        '''
        n = len(X)
        feature_n = X.shape[1]-1
        best_loss = np.inf
        best_feature, best_x_id = 0, 0
        best_sorted_X = []
        
        
        for feature_id in range(feature_n):    
            # X 格式为 [ data , ... , data , label ]          
            sorted_X = np.array(sorted(X ,key=lambda x :x[feature_id] )  ) 
            for x_id in range(n):
                loss = self._loss_func(sorted_X,x_id,feature_id)
                if loss < best_loss:
                    best_loss = loss
                    best_sorted_X = sorted_X
                    best_feature, best_x_id = feature_id, x_id
        return best_x_id,best_feature,best_sorted_X

    def build_tree(self,X,tree_node=None):
        '''
        构建树
        输入：
            X：数据 , 最后一列为标签y
            
            tree_node：结点
        
        后续可以扩展功能
        '''
        X=np.array(X)
        
        if tree_node == None:
            now = self.root #当前节点
        else:
            now = tree_node
        
        n,col = X.shape[0],X.shape[1]-1
        if n <= 1:  #这里？   
            now.feature_value = [X[0][0]]
            now.feature_num   = 0
            now.label         = X[0][1]
        
        else:
            x_id,feature_num  ,sorted_X = self.choose_feature(X)#选择特征
            now.feature_num   = feature_num
            now.feature_value = sorted_X[:,feature_num]
            now.label         = np.sum(sorted_X[:,-1]) / n #这里求当前数组的均值
            
            #拆分数组
            left_X , right_X =[],[]
            for x_id in range(n) : 
                if sorted_X[x_id,feature_num] <= now.feature_value[0] :
                    left_X.append(sorted_X[x_id])
                else:
                    right_X.append(sorted_X[x_id])
            
            #递归生成树
            if left_X !=[] :
                now.leftchild = Node()
                now.leftchild.parent = now  
                self.build_tree( left_X ,now.leftchild)
            if right_X !=[] :
                now.rightchild = Node()
                now.rightchild.parent = now  
                self.build_tree( right_X ,now.rightchild)

    def search(self,X,now=None):
        '''
        搜索：
        输入：
            X：

        输出：
            mean：均值
        
        '''
        if now == None:
            now = self.root
        while True : 
            if X[now.feature_num] <= now.feature_value[0]:
                if now.leftchild != None:
                    now = now.leftchild
                else:
                    break #若无左子树，则当前节点为最底部
            else:
                if now.rightchild != None:
                    now = now.rightchild
                else:
                    break
        mean = now.label
        return mean     #已获取与数据最近的底部叶子节点

        

if __name__ == '__main__':
    X = [[1,4.5],[2,4.75],[3,4.91],[4,5.34],[5,5.8],[6,7.05],[7,7.9],[8,8.23],[9,8.7],[10,9]]
    
    test = [1]
    print('x',X)
    tree = RegressionTree()
    tree.build_tree(X)
    print('tree',tree)
    a = tree.search(test)
    print('result',a)
    