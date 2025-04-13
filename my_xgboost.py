import numpy as np
from tqdm import tqdm 
import bisect

def compute_weighted_quantiles(eps, hess_sorted):
    """
    计算加权分位数候选点（特征值分割点）
    
    参数:
        eps (float): 近似分位数的误差阈值，分桶数量约为 1/eps
        hess_sorted (list): 已按特征值升序排列的样本二阶导数值（作为权重）
    
    返回:
        list: 分位点对应的索引列表（特征值分割点的候选位置）
    """
    total_weight = sum(hess_sorted)
    if total_weight <= 0:
        return []
    
    target_interval = eps * total_weight  # 每个分桶的目标权重和
    current_sum = 0.0
    quantiles = []
    next_target = target_interval
    
    # 遍历排序后的样本，计算累积权重
    for idx, h in enumerate(hess_sorted):
        current_sum += h
        
        # 当累积权重超过当前目标阈值时，记录分割点
        while current_sum >= next_target:
            quantiles.append(idx)
            next_target += target_interval
    
    # 确保最后一个特征值的索引被包含（结束点）
    last_idx = len(hess_sorted) - 1
    if not quantiles or quantiles[-1] != last_idx:
        quantiles.append(last_idx)
    
    # 去重（避免重复添加同一索引）
    seen = set()
    seen.add(0)
    unique_quantiles = [0]
    for idx in quantiles:
        if idx not in seen:
            seen.add(idx)
            unique_quantiles.append(idx)
    
    return unique_quantiles
def find_quantile_indices(sketch, sorted_x):
    """
    根据预先生成的分位点特征值（sketch），返回其在已排序特征值序列（sorted_x）中的索引。
    
    Args:
        sketch (list): 全局分位点特征值列表（已排序）。
        sorted_x (list): 已排序的特征值序列（从小到大）。
        
    Returns:
        list: 分位点对应索引列表 quid。
    """
    quid = []
    for q_value in sketch:
        # 使用 bisect_left 查找分位值的插入位置（左侧索引）
        idx = bisect.bisect_left(sorted_x, q_value)
        idx = min(max(idx, 0), len(sorted_x) - 1)
        quid.append(idx)
    seen = set()
    seen.add(0)
    unique_quantiles = [0]
    for idx in quid:
        if idx not in seen:
            seen.add(idx)
            unique_quantiles.append(idx)
    return unique_quantiles
class XGBoostTree:
    def __init__(self, max_depth=3, lambda_=1.0, gamma=0.0,eps=None):
        self.max_depth = max_depth    # 树的最大深度
        self.lambda_ = lambda_       # L2正则化系数
        self.gamma = gamma           # 分裂最小增益阈值
        self.tree = None             # 树结构
        self.default_direction = {}  # 各特征的默认方向
        self.eps=eps
    class Node:
        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, default_dir=0):
            self.feature = feature     # 分裂特征
            self.threshold = threshold # 分裂阈值
            self.left = left           # 左子树
            self.right = right         # 右子树
            self.value = value         # 叶子节点值
            self.default_dir = default_dir  # 默认方向(0:左,1:右)

    def _calculate_gain(self, G, H, G_left, H_left, G_right, H_right):
        """ 计算分裂增益（公式7） """
        return 0.5 * ( (G_left**2)/(H_left + self.lambda_) + 
                      (G_right**2)/(H_right + self.lambda_) - 
                      (G**2)/(H + self.lambda_) ) - self.gamma




## 更改了分裂点各自方向的计算以及缺失值样本的分配
    def _find_best_split(self, X, grad, hess, feature_mask,global_split_sketch=None):
        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        best_left_indices = None
        best_right_indices = None
        best_default_dir = 0

        n_samples, n_features = X.shape
        G_total = np.sum(grad)
        H_total = np.sum(hess)

        for fid in range(n_features):
            if feature_mask is not None and fid not in feature_mask:
                continue

            missing_mask = np.isnan(X[:, fid])
            present_mask = ~missing_mask
            present_grad = grad[present_mask]
            present_hess = hess[present_mask]
            present_indices = np.where(present_mask)[0]
            missing_indices = np.where(missing_mask)[0]

            # 遍历默认方向，计算最优方向
            max_default_gain = -np.inf
            current_default_dir = 0
            for default_dir in [0, 1]:
                if default_dir == 0:
                    G_left = np.sum(present_grad) + (G_total - np.sum(present_grad))
                    H_left = np.sum(present_hess) + (H_total - np.sum(present_hess))
                    G_right = 0
                    H_right = 0
                else:
                    G_left = np.sum(present_grad)
                    H_left = np.sum(present_hess)
                    G_right = G_total - G_left
                    H_right = H_total - H_left

                gain = self._calculate_gain(G_total, H_total, G_left, H_left, G_right, H_right)
                if gain > max_default_gain:
                    max_default_gain = gain
                    current_default_dir = default_dir

            # 更新全局最佳
            if max_default_gain > best_gain:
                best_gain = max_default_gain
                best_feature = fid
                best_default_dir = current_default_dir
                # 分配索引（此时仅基于默认方向，无实际分裂点）
                if current_default_dir == 0:
                    best_left_indices = np.concatenate([present_indices, missing_indices])
                    best_right_indices = np.array([], dtype=int)
                else:
                    best_left_indices = present_indices
                    best_right_indices = missing_indices

            # 排序存在的特征值
            sorted_idx = np.argsort(X[present_mask, fid])
            sorted_X = X[present_mask, fid][sorted_idx]
            sorted_grad = present_grad[sorted_idx]
            sorted_hess = present_hess[sorted_idx]

            G_left, H_left = 0.0, 0.0
            if global_split_sketch is not None:
                sketch = global_split_sketch[fid]
                quit = find_quantile_indices(sketch,sorted_X)
            else:
                quit = compute_weighted_quantiles(self.eps,sorted_hess) if self.eps is not None else np.arange(1, len(sorted_X)) # 分位值数组

            for i in range(1,len(quit)):
                G_left += np.sum(sorted_grad[quit[i-1]:quit[i]])
                H_left += np.sum(sorted_hess[quit[i-1]:quit[i]])
                G_right_split = np.sum(sorted_grad) - G_left
                H_right_split = np.sum(sorted_hess) - H_left

                if sorted_X[quit[i]] == sorted_X[quit[i-1]]:
                    continue
                
                # 计算两种默认方向的增益
                max_gain_split = -np.inf
                current_split_dir = 0
                for default_dir in [0, 1]:
                    if default_dir == 0:
                        G_left_total = G_left + (G_total - (G_left + G_right_split))
                        H_left_total = H_left + (H_total - (H_left + H_right_split))
                        gain = self._calculate_gain(G_total, H_total, G_left_total, H_left_total, G_right_split, H_right_split)
                    else:
                        G_right_total = G_right_split + (G_total - (G_left + G_right_split))
                        H_right_total = H_right_split + (H_total - (H_left + H_right_split))
                        gain = self._calculate_gain(G_total, H_total, G_left, H_left, G_right_total, H_right_total)

                    if gain > max_gain_split:
                        max_gain_split = gain
                        current_split_dir = default_dir

                if max_gain_split > best_gain:
                    best_gain = max_gain_split
                    best_feature = fid
                    best_threshold = (sorted_X[quit[i]] + sorted_X[quit[i]-1]) / 2
                    best_default_dir = current_split_dir
                    # 分配索引
                    if current_split_dir == 0:
                        best_left_indices = np.concatenate([present_indices[sorted_idx[:quit[i]]], missing_indices])
                        best_right_indices = present_indices[sorted_idx[quit[i]:]]
                    else:
                        best_left_indices = present_indices[sorted_idx[:quit[i]]]
                        best_right_indices = np.concatenate([present_indices[sorted_idx[quit[i]:]], missing_indices])

        return best_feature, best_threshold, best_left_indices, best_right_indices, best_default_dir,best_gain



    def _build_tree(self, X, grad, hess, depth=0, feature_mask=None,global_split_sketch=None):
        """ 递归构建树（支持特征采样） """
        # 叶子节点计算（公式5）
        if depth >= self.max_depth or len(X) < 2:
            w = -np.sum(grad) / (np.sum(hess) + self.lambda_)
            return self.Node(value=w)

        # 特征采样（论文Sec2.3）
        if feature_mask is None:
            feature_mask = np.random.choice(X.shape[1], int(np.sqrt(X.shape[1])), replace=False)

        # 寻找最佳分裂
        fid, thres, left_idx, right_idx, default_dir,gain = self._find_best_split(X, grad, hess, feature_mask,global_split_sketch)
        
        # 增益不足则停止分裂
        if fid is None or gain <= 0:
            w = -np.sum(grad) / (np.sum(hess) + self.lambda_)
            return self.Node(value=w)

        # 递归构建子树
        left = self._build_tree(X[left_idx], grad[left_idx], hess[left_idx], depth+1, feature_mask,global_split_sketch=global_split_sketch)
        right = self._build_tree(X[right_idx], grad[right_idx], hess[right_idx], depth+1, feature_mask,global_split_sketch=global_split_sketch)
        return self.Node(feature=fid, threshold=thres, 
                        left=left, right=right, default_dir=default_dir)

    def fit(self, X, grad, hess,global_split_sketch=None):
        """ 训练单棵树 """
        self.tree = self._build_tree(X, grad, hess,global_split_sketch=global_split_sketch)
    
    def predict(self, X):
        """ 预测 """
        return np.array([self._predict_single(x, self.tree) for x in X])
    
    def _predict_single(self, x, node):
        """ 递归预测单个样本 """
        if node.value is not None:
            return node.value
        if np.isnan(x[node.feature]):  # 处理缺失值
            if node.default_dir == 0:
                return self._predict_single(x, node.left)
            else:
                return self._predict_single(x, node.right)
        elif x[node.feature] < node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)


class XGBoost:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, 
                 lambda_=1.0, gamma=0.0, subsample=0.8,objective='regression',num_clasees=None,eps=None,global_split=False):
        self.n_estimators = n_estimators
        self.eta = learning_rate
        self.max_depth = max_depth
        self.lambda_ = lambda_
        self.gamma = gamma
        self.subsample = subsample
        self.trees = []
        self.base_score = None
        self.objective =objective
        self.eps=eps
        self.global_split=global_split
        if self.global_split==False:
            self.global_split_sketch=None

        if self.objective=='regression':
            self.loss='squared_error'
        elif self.objective=='binaryclass':
            self.loss='logistic'
        elif self.objective=='multiclass':
            self.loss='softmax'
            self.num_classes=num_clasees
        elif self.objective=='rank':
            self.loss='rank'
    def _gradient(self, y_true, y_pred):
        """ 一阶梯度（平方损失） """
        return y_pred - y_true
    
    def _hessian(self, y_true, y_pred):
        """ 二阶梯度（平方损失） """
        return np.ones_like(y_true)
    def _softmax(self, x):
        """ Softmax 函数 """
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)
    def safe_sigmoid(self,x):
        x = np.clip(x, -50, 50)  # 限制输入范围
        return 1.0 / (1.0 + np.exp(-x))
    def _gradient_and_hessian_rank(self, y_true, y_pred, qid, group_indices):
        """ 排序任务的梯度计算（Pairwise Loss + NDCG 偏置） """
        n = len(y_true)
        grad = np.zeros(n)
        hess = np.zeros(n)
        
        for start, end in group_indices:
            # 提取当前组的标签和预测值
            group_y_true = y_true[start:end]
            group_y_pred = y_pred[start:end]
            group_size = end - start
            
            # 计算当前组的理想DCG（IDCG）用于归一化
            ideal_sorted = np.argsort(-group_y_true)  # 按真实标签降序排列
            idcg = self._compute_dcg(group_y_true[ideal_sorted])
            
            if idcg == 0:
                continue  # 避免除以零
            
            # 当前预测值的排序（按预测得分降序）
            current_rank = np.argsort(-group_y_pred)
            
            # 组内两两比较
            for i in range(group_size):
                for j in range(group_size):
                    if group_y_true[i] <= group_y_true[j]:
                        continue  # 只处理 y_true[i] > y_true[j] 的文档对
                    
                    s_diff =  group_y_pred[i] - group_y_pred[j]
                    sigma = self.safe_sigmoid(s_diff)
                    # sigma = 1 / (1 + np.exp(-s_diff))  # 建议替换为 safe_sigmoid
                    
                    # 计算交换i和j位置后的ΔNDCG
                    delta_ndcg = self._compute_delta_ndcg(
                        group_y_true, group_y_pred, i, j, current_rank, idcg
                    )
                    
                    # 梯度更新（用ΔNDCG加权）
                    grad[start + i] += sigma 
                    grad[start + j] -= sigma 
                    
                    # Hessian更新（通常不变）
                    hess_ij = sigma * (1 - sigma)
                    hess[start + i] += hess_ij
                    hess[start + j] += hess_ij
        print("梯度范围:", grad.min(), grad.max())
        print("Hessian范围:", hess.min(), hess.max())
        print("ΔNDCG示例:", delta_ndcg)
        return grad, hess

    def _compute_dcg(self, sorted_labels, k=10):
        """ 计算 DCG@k（修复版本） """
        # 1. 截断 sorted_labels 到前 k 个元素
        sorted_labels_truncated = sorted_labels[:k]
        m = len(sorted_labels_truncated)
        if m == 0:
            return 0.0
        
        # 2. 生成对应前 m 个位置的折扣因子（位置从 1 到 m）
        discounts = 1 / np.log2(np.arange(2, m + 2))  # 排名 i 对应 log2(i+1)
        gains = 2** sorted_labels_truncated - 1
        
        # 3. 确保 gains 和 discounts 长度相同
        return np.sum(gains * discounts)

    def _compute_delta_ndcg(self, y_true, y_pred, i, j, current_rank, idcg, k=10):
        """ 计算交换文档i和j后的ΔNDCG """
        # 当前NDCG
        current_dcg = self._compute_dcg(y_true[current_rank])
        current_ndcg = current_dcg / idcg if idcg > 0 else 0
        
        # 交换i和j的位置后的新排序
        new_rank = current_rank.copy()
        pos_i = np.where(new_rank == i)[0][0]
        pos_j = np.where(new_rank == j)[0][0]
        new_rank[pos_i], new_rank[pos_j] = new_rank[pos_j], new_rank[pos_i]
        
        # 新NDCG
        new_dcg = self._compute_dcg(y_true[new_rank])
        new_ndcg = new_dcg / idcg if idcg > 0 else 0
        
        return new_ndcg - current_ndcg
    def _gradient_and_hessian(self, y_true, y_pred,p_0=False,qid=None):
        """ 一阶和二阶梯度 """
        if self.loss=='squared_error':
            return y_pred - y_true, np.ones_like(y_true)
        elif self.loss=='logistic':
            if p_0:
                return y_pred - y_true, y_pred*(1-y_pred)
            p = 1 / (1 + np.exp(-y_pred))
            return p-y_true, p*(1-p)
        elif self.loss=='softmax':
            p = self._softmax(y_pred)
            grad = p - y_true  # 梯度公式
            hess = p * (1 - p)    # Hessian 公式
            return grad, hess
        elif self.loss=='rank':
            if qid is not None:
                # 检查qid格式
                assert len(qid) == len(y_pred), "qid长度必须与样本数一致"
                unique_qids, group_counts = np.unique(qid, return_counts=True)
                assert np.all(np.diff(qid) >= 0), "qid必须连续排列（如[1,1,2,2,...]）"
                
                # 预处理：按qid分组，记录每组起始索引
                group_indices = []
                start = 0
                for count in group_counts:
                    group_indices.append((start, start + count))
                    start += count
            else:
                print("qid is not provided, using all samples as one group")
                group_indices = [(0, len(y_true))]
            grad, hess = self._gradient_and_hessian_rank(y_true, y_pred, qid, group_indices=group_indices)
            return grad, hess
        return self._gradient(y_true, y_pred), self._hessian(y_true, y_pred)
    
    def _predict_trees(self, X, iteration):
        """ 根据当前迭代的树生成预测值 """
        pred = np.zeros((len(X), self.num_classes))
        # 获取当前迭代为每个类别训练的树
        for k in range(self.num_classes):
            tree = self.trees[iteration * self.num_classes + k]
            pred[:, k] += tree[1].predict(X)
        return pred
    def calculate_global_weighted_quantiles(self,X,hess):
        """ 计算每个特征的分位点 """
        sketch = {}
        n_samples, n_features = X.shape
        for fid in range(n_features):
            missing_mask = np.isnan(X[:, fid])
            present_mask = ~missing_mask
            present_hess = hess[present_mask]

            # 排序存在的特征值
            sorted_idx = np.argsort(X[present_mask, fid])
            sorted_X = X[present_mask, fid][sorted_idx]
            sorted_hess = present_hess[sorted_idx]

            quit = compute_weighted_quantiles(self.eps,sorted_hess)  # 分位值数组
            sketch[fid]=[sorted_X[i] for i in quit]

        return sketch
    def fit_multiclass(self, X, y):
        """ 多分类训练函数 """
        # 确定类别数
        if self.num_classes is None:
            self.num_classes = len(np.unique(y))
        
        # 将标签转换为 one-hot 编码
        y_one_hot = np.eye(self.num_classes)[y]
        
        # 初始化预测值为均匀分布（或实际类别频率）
        self.base_score = np.mean(y_one_hot, axis=0)  # 每个类别的平均概率
        y_pred = np.full((len(y), self.num_classes), self.base_score, dtype=np.double)
        '''多分类的Global'''
        # 训练多分类模型
        for _ in tqdm(range(self.n_estimators),
                     desc="Training XGBoost",
                     unit="tree",
                     ncols=80,  # 进度条宽度
                     bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed} remaining: {remaining}]"):
            # 计算梯度与 Hessian（多分类版本）
            grad, hess = self._gradient_and_hessian(y_one_hot, y_pred)
            
            # 行采样
            idx = np.random.choice(len(X), int(len(X)*self.subsample), replace=True)
            
            # 为每个类别训练一棵树
            for k in range(self.num_classes):
                tree = XGBoostTree(max_depth=self.max_depth, lambda_=self.lambda_, gamma=self.gamma,eps=self.eps)
                tree.fit(X[idx], grad[idx, k], hess[idx, k])  # 每个类别的梯度单独训练树
                self.trees.append((k, tree))  # 存储树及其对应类别
                
            # 更新所有类别的预测值
            y_pred += self.eta * self._predict_trees(X, iteration=_)
            
    def fit(self, X, y,qid=None):
        """ 训练模型 """

        if self.objective=='multiclass':
            self.fit_multiclass(X,y)
            return 
        self.base_score = np.mean(y)
        y_pred = np.full_like(y, self.base_score,dtype=np.double)# 这里修改了数据类型
        if self.global_split:
            grad, hess = self._gradient_and_hessian(y, y_pred,p_0=True,qid=qid)
            self.global_split_sketch=self.calculate_global_weighted_quantiles(X,hess)

        for _ in tqdm(range(self.n_estimators),
                     desc="Training XGBoost",
                     unit="tree",
                     ncols=80,  # 进度条宽度
                     bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed} remaining: {remaining}]"):
            # 计算梯度
            # grad = self._gradient(y, y_pred)
            # hess = self._hessian(y, y_pred)
            grad, hess = self._gradient_and_hessian(y, y_pred,p_0=(_==0),qid=qid)
            
            # 行采样（论文Sec2.3）
            idx = np.random.choice(len(X), int(len(X)*self.subsample), replace=False)
            
            # 训练树
            tree = XGBoostTree(max_depth=self.max_depth, 
                              lambda_=self.lambda_, gamma=self.gamma,eps=self.eps)
            tree.fit(X[idx], grad[idx], hess[idx],global_split_sketch=self.global_split_sketch)
            
            # 更新预测值
            y_pred += self.eta * tree.predict(X)
            
            self.trees.append(tree)
        
    def predict(self, X):
        """ 预测 """
        if self.objective=='multiclass':
            """ 多分类预测 """
            y_pred = np.zeros((len(X), self.num_classes))
            for k, tree in self.trees:
                y_pred[:, k] += self.eta * tree.predict(X)
            return np.argmax(self._softmax(y_pred), axis=1)
        y_pred = np.full(X.shape[0], self.base_score)
        for tree in self.trees:
            y_pred += self.eta * tree.predict(X)
        return y_pred
    



    ##test
from sklearn.model_selection import train_test_split

# ----------------------
# 数据加载与预处理
# ----------------------
def load_yahoo_data(path, num_features=700):
    """加载Yahoo LTRC数据集"""
    X, y, qids = [], [], []
    with open(path, 'r') as f:
        for line in f:
            if not line.strip(): continue
            
            # 解析行数据
            parts = line.split()
            label = int(parts[0])
            qid = parts[1].split(':')[1]
            
            # 初始化特征向量
            features = np.zeros(num_features)
            for pair in parts[2:]:
                if ':' in pair:
                    fid, val = pair.split(':')
                    features[int(fid)-1] = float(val)  # 特征ID从1开始
                    
            X.append(features)
            y.append(label)
            qids.append(qid)
    
    return np.array(X), np.array(y), np.array(qids)

def load_multiclass_data(path):
    """加载二分类或多分类数据集"""
    X, y = [], []
    count=0
    with open(path, 'r') as f:
        for line in f:
            if not line.strip(): continue
            
            # 解析行数据
            parts = line.split(',')
            label = int(float(parts[0]))
            features = [float(i) for i in parts[1:] ]
                    
            X.append(features)
            y.append(label)
            count+=1
            if count>1000000:break
    k=len(np.unique(y))
    return np.array(X), np.array(y), k


import csv

def read_csv_header(path):
    with open(path, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)  # 读取第一行（表头）
    return header
if __name__ == '__main__':
    
    # 示例
    header = read_csv_header("./higgs\HIGGS.csv\HIGGS.csv")
    # 加载分类数据集（）
    X,y,k = load_multiclass_data(path="./higgs\HIGGS.csv\HIGGS.csv")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # X, y, qids = load_yahoo_data("./higgs\HIGGS.csv\HIGGS.csv")

    # # 划分训练测试集（保持查询组完整）
    # unique_qids = np.unique(qids)
    # train_qids, test_qids = train_test_split(unique_qids, test_size=0.2, random_state=42)

    # train_mask = np.isin(qids, train_qids)
    # test_mask = np.isin(qids, test_qids)

    # X_train, y_train, qids_train = X[train_mask], y[train_mask], qids[train_mask]
    # X_test, y_test, qids_test = X[test_mask], y[test_mask], qids[test_mask]

    # ----------------------
    # 评估指标实现
    # ----------------------
    def calculate_ndcg(y_true, y_pred, qids, k=10):
        """计算NDCG@k"""
        ndcg_scores = []
        for qid in np.unique(qids):
            mask = (qids == qid)
            labels = y_true[mask]
            preds = y_pred[mask]
            
            # 按预测分数排序
            order = np.argsort(preds)[::-1][:k]  # 降序排列取top-k
            sorted_labels = labels[order]
            
            # 计算DCG
            dcg = sum( (2**l - 1) / np.log2(i+2) for i, l in enumerate(sorted_labels) )
            
            # 计算IDCG
            ideal_order = np.argsort(labels)[::-1][:k]
            ideal_labels = labels[ideal_order]
            idcg = sum( (2**l - 1) / np.log2(i+2) for i, l in enumerate(ideal_labels) )
            
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcg_scores.append(ndcg)
        
        return np.mean(ndcg_scores)

    # ----------------------
    # 模型训练与评估
    # ----------------------
    # 初始化模型（参数与论文一致）
    model = XGBoost(
        n_estimators=500,
        learning_rate=0.1,
        max_depth=8,
        lambda_=1.0,
        gamma=0.0,
        subsample=0.8,
        objective='binaryclass',
        eps=0.5
    )

    # 训练模型
    model.fit(X_train, y_train)

    # 预测测试集
    y_pred = model.predict(X_test)
    y_pred_proba = 1 / (1 + np.exp(-y_pred))
    # 将概率转换为类别预测（默认阈值为0.5）
    y_pred_class = (y_pred_proba > 0.5).astype(int)
    # 计算Accuracy
    accuracy = np.mean(y_pred_class == y_test)
    print(f"Accuracy: {accuracy:.4f}")
    # # 评估NDCG@10
    # ndcg_score = calculate_ndcg(y_test, y_pred, qids_test, k=10)
    # print(f"NDCG@10: {ndcg_score:.4f}")

    # # ----------------------
    # # 与论文结果对比
    # # ----------------------
    # print("\n与论文结果对比：")
    # print(f"当前实现: {ndcg_score:.4f}")
    # print("论文XGBoost: 0.7892")


    '''绘制二分类的AUC曲线'''
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt

    # 计算 ROC 曲线的 FPR, TPR 和阈值
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

    # 计算 AUC 值
    roc_auc = auc(fpr, tpr)
    print(f"AUC: {roc_auc:.4f}")
    # 绘制 ROC 曲线
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # 绘制对角线
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()