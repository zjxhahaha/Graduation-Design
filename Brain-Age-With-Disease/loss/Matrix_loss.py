import torch
import numpy as np 
import torch.nn as nn
import scipy.stats.stats as stats
import torch.nn.functional as F


class Matrix_distance_loss(torch.nn.Module):
    def __init__(self, p=1):
        super().__init__()
        self.p = p
        self.criterion_mse = torch.nn.MSELoss()

    def forward(self, pred, gt):
        '''
        Computes the P-norm distance between every pair of row vector in the input.
        This is identical to the upper triangular portion, excluding the diagnoal,
        of
        torch.norm(input[:, None] - input, dim=2, p=p).
        This function will be faster if the row are contiguous.
        '''
        Matrix_Distance_pred = torch.norm(pred[:, None] - pred, dim=2, p=self.p)
        Matrix_Distance_gt = torch.norm(gt[:, None] - gt, dim=2, p=self.p)
        
        loss = self.criterion_mse(Matrix_Distance_gt, Matrix_Distance_pred)
       
        return loss


class Matrix_distance_L2_loss(torch.nn.Module):
    def __init__(self, p=1):
        super().__init__()
        self.p = p
        self.criterion_mse = torch.nn.MSELoss()

    def forward(self, pred, gt):
        '''
        Computes the P-norm distance between every pair of row vector in the input.
        This is identical to the upper triangular portion, excluding the diagnoal,
        of
        torch.norm(input[:, None] - input, dim=2, p=p).
        This function will be faster if the row are contiguous.
        '''
        Matrix_Distance_pred = torch.norm(pred[:, None] - gt, dim=2, p=self.p)
        Matrix_Distance_gt = torch.norm(gt[:, None] - gt, dim=2, p=self.p)
        
        loss = self.criterion_mse(Matrix_Distance_gt, Matrix_Distance_pred)
       
        return loss

class Matrix_distance_L3_loss(torch.nn.Module):
    def __init__(self, p=1):
        super().__init__()
        self.p = p
        self.criterion_mse = torch.nn.MSELoss()

    def forward(self, pred, gt):
        '''
        Computes the P-norm distance between every pair of row vector in the input.
        This is identical to the upper triangular portion, excluding the diagnoal,
        of
        torch.norm(input[:, None] - input, dim=2, p=p).
        This function will be faster if the row are contiguous.
        '''
        Matrix_Distance_pred = torch.norm(pred[:, None] - pred, dim=2, p=self.p)
        Matrix_Distance_gt = torch.norm(gt[:, None] - pred, dim=2, p=self.p)
        
        loss = self.criterion_mse(Matrix_Distance_gt, Matrix_Distance_pred)
       
        return loss

def test_loss():
       
    # ======== use a simple neural network to test loss function =========
    input_size = 1
    output_size = 1
    num_epochs = 4000
    learning_rate = 0.001
    
    x_train = np.array(np.ones((100,1)),dtype=np.float32)
    print(x_train.shape)
    y_train = np.array(np.ones((100,1)) * 10 + np.random.rand(100,1),dtype=np.float32)
    print(y_train.shape)
    
    # ========  Linear regression model  ======== 
    model = nn.Sequential(nn.Linear(input_size, 128),
                          nn.Linear(128,1))

    criterion = Matrix_distance_loss()

    #  ======== 定义迭代优化算法， 使用的是随机梯度下降算法 =========
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  
    loss_dict = []

    # ========  Train the model 
    for epoch in range(num_epochs):
        inputs = torch.from_numpy(x_train)
        targets = torch.from_numpy(y_train)

    # ========  前向传播计算网络结构的输出结果 ========  #
        outputs = model(inputs)
        loss = criterion(outputs, targets)
       
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #  =====  打印训练信息和保存 loss ==========  #
        loss_dict.append(loss.item())
        if (epoch+1) % 100 == 0:
            print ('Epoch [{}/{}], Loss: {:.6f}'.format(epoch+1, num_epochs, loss.item()))
    
if __name__ == "__main__":  
    
    X = torch.Tensor(((1.,),(2.,),(3.,)))
    Y = torch.Tensor(((3.,),(3.,),(3.,)))
    
    matrix_A = torch.norm(X[:, None] - Y, dim=2, p=1)
    matrix_B = torch.norm(Y[:, None] - Y, dim=2, p=1)

    criterion = torch.nn.MSELoss()

    print(X.shape, Y.shape)
    print(X[:None].shape)
    print(matrix_A)
    print(matrix_B)
    print(matrix_A - matrix_B)
    print(criterion(matrix_A, matrix_B))
