from torch import nn

class SimpleDenseNet_gauss(nn.Module):
    def __init__(
        self,
        input_size: int = 784,
        lin1_size: int = 256,
        lin2_size: int = 256,
        lin3_size: int = 256,
        output_size: int = 10,
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, lin1_size),
            nn.BatchNorm1d(lin1_size),
            nn.ReLU(),
            nn.Linear(lin1_size, lin2_size),
            nn.BatchNorm1d(lin2_size),
            nn.ReLU(),
            nn.Linear(lin2_size, lin3_size),
            nn.BatchNorm1d(lin3_size),
            nn.ReLU(),
            nn.Linear(lin3_size, output_size),
        )

    def forward(self, x):
        
        return self.model(x)
    
class SimpleDenseNet_ReLU_gauss(nn.Module):
    def __init__(
        self,
        input_size: int = 784,
        lin1_size: int = 256,
        lin2_size: int = 256,
        lin3_size: int = 256,
        output_size: int = 10,
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, lin1_size),
            nn.ReLU(),
            nn.Linear(lin1_size, lin2_size),
            nn.ReLU(),
            nn.Linear(lin2_size, lin3_size),
            nn.ReLU(),
            nn.Linear(lin3_size, output_size),
        )

    def forward(self, x):
        
        return self.model(x)
    
class SimpleDenseNet_SiLU_gauss(nn.Module):
    def __init__(
        self,
        input_size: int = 784,
        lin1_size: int = 256,
        lin2_size: int = 256,
        lin3_size: int = 256,
        output_size: int = 10,
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, lin1_size),
            nn.BatchNorm1d(lin1_size),
            nn.SiLU(),
            nn.Linear(lin1_size, lin2_size),
            nn.BatchNorm1d(lin2_size),
            nn.SiLU(),
            nn.Linear(lin2_size, lin3_size),
            nn.BatchNorm1d(lin3_size),
            nn.SiLU(),
            nn.Linear(lin3_size, output_size),
        )

    def forward(self, x):
        
        return self.model(x)

class SimpleDenseNet_ELU_gauss(nn.Module):
    def __init__(
        self,
        input_size: int = 784,
        lin1_size: int = 256,
        lin2_size: int = 256,
        lin3_size: int = 256,
        output_size: int = 10,
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, lin1_size),
            nn.BatchNorm1d(lin1_size),
            nn.ELU(),
            nn.Linear(lin1_size, lin2_size),
            nn.BatchNorm1d(lin2_size),
            nn.ELU(),
            nn.Linear(lin2_size, lin3_size),
            nn.BatchNorm1d(lin3_size),
            nn.ELU(),
            nn.Linear(lin3_size, output_size),
        )

    def forward(self, x):
        
        return self.model(x)

class SimpleDenseNet_SS_gauss(nn.Module):
    def __init__(
        self,
        input_size: int = 784,
        lin1_size: int = 256,
        lin2_size: int = 256,
        lin3_size: int = 256,
        output_size: int = 10,
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, lin1_size),
            nn.BatchNorm1d(lin1_size),
            nn.Softshrink(),
            nn.Linear(lin1_size, lin2_size),
            nn.BatchNorm1d(lin2_size),
            nn.Softshrink(),
            nn.Linear(lin2_size, lin3_size),
            nn.BatchNorm1d(lin3_size),
            nn.Softshrink(),
            nn.Linear(lin3_size, output_size),
        )

    def forward(self, x):
        
        return self.model(x)
    
class SimpleDenseNet_LL_gauss(nn.Module):
    def __init__(
        self,
        input_size: int = 784,
        lin1_size: int = 256,
        lin2_size: int = 256,
        lin3_size: int = 256,
        output_size: int = 10,
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, lin1_size),
            nn.BatchNorm1d(lin1_size),
            nn.Linear(lin1_size, lin2_size),
            nn.BatchNorm1d(lin2_size),
            nn.Linear(lin2_size, lin3_size),
            nn.BatchNorm1d(lin3_size),
            nn.Linear(lin3_size, output_size),
        )

    def forward(self, x):
        
        return self.model(x)

class SimpleDenseNet_LR_gauss(nn.Module):
    def __init__(
        self,
        input_size: int = 784,
        lin1_size: int = 256,
        lin2_size: int = 256,
        lin3_size: int = 256,
        output_size: int = 10,
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, lin1_size),
            nn.BatchNorm1d(lin1_size),
            nn.LeakyReLU(),
            nn.Linear(lin1_size, lin2_size),
            nn.BatchNorm1d(lin2_size),
            nn.LeakyReLU(),
            nn.Linear(lin2_size, lin3_size),
            nn.BatchNorm1d(lin3_size),
            nn.LeakyReLU(),
            nn.Linear(lin3_size, output_size),
        )

    def forward(self, x):
        
        return self.model(x)
    
class SimpleDenseNet_LR2_gauss(nn.Module):
    def __init__(
        self,
        input_size: int = 784,
        lin1_size: int = 256,
        lin2_size: int = 256,
        lin3_size: int = 256,
        output_size: int = 10,
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, lin1_size),
            nn.BatchNorm1d(lin1_size),
            nn.LeakyReLU(),
            nn.Linear(lin1_size, lin2_size),
            nn.BatchNorm1d(lin2_size),
            nn.LeakyReLU(),
            nn.Linear(lin2_size, lin3_size),
            nn.BatchNorm1d(lin3_size),
            nn.LeakyReLU(),
            nn.Linear(lin3_size, lin3_size),
            nn.BatchNorm1d(lin3_size),
            nn.LeakyReLU(),
            nn.Linear(lin3_size, lin3_size),
            nn.BatchNorm1d(lin3_size),
            nn.LeakyReLU(),
            nn.Linear(lin3_size, output_size),
        )

    def forward(self, x):
        
        return self.model(x)

class CNN_gauss(nn.Module):
    def __init__(
        self,
        input_size: int = 784,
        lin1_size: int = 256,
        lin2_size: int = 256,
        lin3_size: int = 256,
        output_size: int = 10,
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Unflatten(1, (1, input_size)),
            nn.Conv1d(1, 2, kernel_size=3, padding=1, stride=2),
            nn.Conv1d(2, 1, kernel_size=3, padding=1, stride=2),
            nn.Flatten(),
            nn.Linear(input_size //4, input_size //4),
            nn.BatchNorm1d(input_size //4),
            nn.ReLU(),
            nn.Linear(input_size //4, output_size),
            nn.BatchNorm1d(output_size),
            nn.Linear(output_size, output_size),
        )

    def forward(self, x):

        return self.model(x)

class CNN2_gauss(nn.Module):
    def __init__(
        self,
        input_size: int = 784,
        lin1_size: int = 256,
        lin2_size: int = 256,
        lin3_size: int = 256,
        output_size: int = 10,
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Unflatten(1, (1, input_size)),
            nn.Conv1d(1, 2, kernel_size=3, padding=1, stride=2),
            nn.Flatten(),
            nn.Linear(input_size, lin1_size),
            nn.BatchNorm1d(lin1_size),
            nn.LeakyReLU(),
            nn.Linear(lin1_size, lin2_size),
            nn.BatchNorm1d(lin2_size),
            nn.LeakyReLU(),
            nn.Linear(lin2_size, lin3_size),
            nn.BatchNorm1d(lin3_size),
            nn.LeakyReLU(),
            nn.Linear(lin3_size, lin3_size),
            nn.BatchNorm1d(lin3_size),
            nn.LeakyReLU(),
            nn.Linear(lin3_size, lin3_size),
            nn.BatchNorm1d(lin3_size),
            nn.LeakyReLU(),
            nn.Linear(lin3_size, output_size),
        )

    def forward(self, x):

        return self.model(x)

class SimpleDenseNet_LR_5H_gauss(nn.Module):
    def __init__(
        self,
        input_size: int = 784,
        lin1_size: int = 256,
        lin2_size: int = 256,
        lin3_size: int = 256,
        lin4_size: int = 256,
        lin5_size: int = 256,
        output_size: int = 10,
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, lin1_size),
            nn.BatchNorm1d(lin1_size),
            nn.LeakyReLU(),
            nn.Linear(lin1_size, lin2_size),
            nn.BatchNorm1d(lin2_size),
            nn.LeakyReLU(),
            nn.Linear(lin2_size, lin3_size),
            nn.BatchNorm1d(lin3_size),
            nn.LeakyReLU(),
            nn.Linear(lin3_size, lin4_size),
            nn.BatchNorm1d(lin4_size),
            nn.LeakyReLU(),
            nn.Linear(lin4_size, lin5_size),
            nn.BatchNorm1d(lin5_size),
            nn.LeakyReLU(),
            nn.Linear(lin5_size, output_size),
        )

    def forward(self, x):
        
        return self.model(x)

class SimpleDenseNet_SiRu_10H_gauss(nn.Module):
    def __init__(
        self,
        input_size: int = 100,
        lin1_size: int = 256,
        lin2_size: int = 256,
        lin3_size: int = 256,
        lin4_size: int = 256,
        lin5_size: int = 256,
        output_size: int = 10,
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, lin1_size),
            nn.SiLU(),
            nn.Linear(lin1_size, lin2_size),
            nn.SiLU(),
            nn.Linear(lin2_size, lin2_size),
            nn.SiLU(),
            nn.Linear(lin2_size, lin2_size),
            nn.SiLU(),
            nn.Linear(lin2_size, lin2_size),
            nn.SiLU(),
            nn.Linear(lin2_size, lin2_size),
            nn.SiLU(),
            nn.Linear(lin2_size, lin2_size),
            nn.SiLU(),
            nn.Linear(lin2_size, lin2_size),
            nn.SiLU(),
            nn.Linear(lin2_size, lin2_size),
            nn.SiLU(),
            nn.Linear(lin2_size, lin2_size),
            nn.SiLU(),
            nn.Linear(lin2_size, output_size),
        )

    def forward(self, x):
        
        return self.model(x)

class SimpleDenseNet_ReLU_10H_gauss(nn.Module):
    def __init__(
        self,
        input_size: int = 100,
        lin1_size: int = 256,
        lin2_size: int = 256,
        lin3_size: int = 256,
        lin4_size: int = 256,
        lin5_size: int = 256,
        output_size: int = 10,
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, lin1_size),
            nn.ReLU(),
            nn.Linear(lin1_size, lin2_size),
            nn.ReLU(),
            nn.Linear(lin2_size, lin2_size),
            nn.ReLU(),
            nn.Linear(lin2_size, lin2_size),
            nn.ReLU(),
            nn.Linear(lin2_size, lin2_size),
            nn.ReLU(),
            nn.Linear(lin2_size, lin2_size),
            nn.ReLU(),
            nn.Linear(lin2_size, lin2_size),
            nn.ReLU(),
            nn.Linear(lin2_size, lin2_size),
            nn.ReLU(),
            nn.Linear(lin2_size, lin2_size),
            nn.ReLU(),
            nn.Linear(lin2_size, lin2_size),
            nn.ReLU(),
            nn.Linear(lin2_size, output_size),
        )

    def forward(self, x):
        
        return self.model(x)
    
class SimpleDenseNet_ReLU_20H_gauss(nn.Module):
    def __init__(
        self,
        input_size: int = 100,
        lin1_size: int = 256,
        lin2_size: int = 256,
        lin3_size: int = 256,
        lin4_size: int = 256,
        lin5_size: int = 256,
        output_size: int = 10,
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, lin1_size),
            nn.ReLU(),
            nn.Linear(lin1_size, lin2_size),
            nn.ReLU(),
            nn.Linear(lin2_size, lin2_size),
            nn.ReLU(),
            nn.Linear(lin2_size, lin2_size),
            nn.ReLU(),
            nn.Linear(lin2_size, lin2_size),
            nn.ReLU(),
            nn.Linear(lin2_size, lin2_size),
            nn.ReLU(),
            nn.Linear(lin2_size, lin2_size),
            nn.ReLU(),
            nn.Linear(lin2_size, lin2_size),
            nn.ReLU(),
            nn.Linear(lin2_size, lin2_size),
            nn.ReLU(),
            nn.Linear(lin2_size, lin2_size),
            nn.ReLU(),
            nn.Linear(lin2_size, lin2_size),
            nn.ReLU(),
            nn.Linear(lin2_size, lin2_size),
            nn.ReLU(),
            nn.Linear(lin2_size, lin2_size),
            nn.ReLU(),
            nn.Linear(lin2_size, lin2_size),
            nn.ReLU(),
            nn.Linear(lin2_size, lin2_size),
            nn.ReLU(),
            nn.Linear(lin2_size, lin2_size),
            nn.ReLU(),
            nn.Linear(lin2_size, lin2_size),
            nn.ReLU(),
            nn.Linear(lin2_size, lin2_size),
            nn.ReLU(),
            nn.Linear(lin2_size, lin2_size),
            nn.ReLU(),
            nn.Linear(lin2_size, lin2_size),
            nn.ReLU(),
            nn.Linear(lin2_size, output_size),
        )

    def forward(self, x):
        
        return self.model(x)

class SimpleDenseNet_LeakyReLU_20H_gauss(nn.Module):
    def __init__(
        self,
        input_size: int = 100,
        lin1_size: int = 256,
        lin2_size: int = 256,
        lin3_size: int = 256,
        lin4_size: int = 256,
        lin5_size: int = 256,
        output_size: int = 10,
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, lin1_size),
            nn.BatchNorm1d(lin1_size),
            nn.LeakyReLU(),
            nn.Linear(lin1_size, lin2_size),
            nn.BatchNorm1d(lin2_size),
            nn.LeakyReLU(),
            nn.Linear(lin2_size, lin2_size),
            nn.BatchNorm1d(lin2_size),
            nn.LeakyReLU(),
            nn.Linear(lin2_size, lin2_size),
            nn.BatchNorm1d(lin2_size),
            nn.LeakyReLU(),
            nn.Linear(lin2_size, lin2_size),
            nn.BatchNorm1d(lin2_size),
            nn.LeakyReLU(),
            nn.Linear(lin2_size, lin2_size),
            nn.BatchNorm1d(lin2_size),
            nn.LeakyReLU(),
            nn.Linear(lin2_size, lin2_size),
            nn.BatchNorm1d(lin2_size),
            nn.LeakyReLU(),
            nn.Linear(lin2_size, lin2_size),
            nn.BatchNorm1d(lin2_size),
            nn.LeakyReLU(),
            nn.Linear(lin2_size, lin2_size),
            nn.BatchNorm1d(lin2_size),
            nn.LeakyReLU(),
            nn.Linear(lin2_size, lin2_size),
            nn.BatchNorm1d(lin2_size),
            nn.LeakyReLU(),
            nn.Linear(lin2_size, lin2_size),
            nn.BatchNorm1d(lin2_size),
            nn.LeakyReLU(),
            nn.Linear(lin2_size, lin2_size),
            nn.BatchNorm1d(lin2_size),
            nn.LeakyReLU(),
            nn.Linear(lin2_size, lin2_size),
            nn.BatchNorm1d(lin2_size),
            nn.LeakyReLU(),
            nn.Linear(lin2_size, lin2_size),
            nn.BatchNorm1d(lin2_size),
            nn.LeakyReLU(),
            nn.Linear(lin2_size, lin2_size),
            nn.BatchNorm1d(lin2_size),
            nn.LeakyReLU(),
            nn.Linear(lin2_size, lin2_size),
            nn.BatchNorm1d(lin2_size),
            nn.LeakyReLU(),
            nn.Linear(lin2_size, lin2_size),
            nn.BatchNorm1d(lin2_size),
            nn.LeakyReLU(),
            nn.Linear(lin2_size, lin2_size),
            nn.BatchNorm1d(lin2_size),
            nn.LeakyReLU(),
            nn.Linear(lin2_size, lin2_size),
            nn.BatchNorm1d(lin2_size),
            nn.LeakyReLU(),
            nn.Linear(lin2_size, lin2_size),
            nn.BatchNorm1d(lin2_size),
            nn.LeakyReLU(),
            nn.Linear(lin2_size, output_size),
        )

    def forward(self, x):
        
        return self.model(x)