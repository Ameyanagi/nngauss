from torch import nn

class H3_ReLU(nn.Module):
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
    
class H4_ReLU(nn.Module):
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
            nn.Linear(lin3_size, lin3_size),
            nn.ReLU(),
            nn.Linear(lin3_size, output_size),
        )

    def forward(self, x):
        
        return self.model(x)

class H5_ReLU(nn.Module):
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
            nn.Linear(lin3_size, lin3_size),
            nn.ReLU(),
            nn.Linear(lin3_size, lin3_size),
            nn.ReLU(),
            nn.Linear(lin3_size, output_size),
        )

    def forward(self, x):
        
        return self.model(x)


class H10_ReLU(nn.Module):
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
            nn.Linear(lin3_size, lin3_size),
            nn.ReLU(),
            nn.Linear(lin3_size, lin3_size),
            nn.ReLU(),
            nn.Linear(lin3_size, lin3_size),
            nn.ReLU(),
            nn.Linear(lin3_size, lin3_size),
            nn.ReLU(),
            nn.Linear(lin3_size, lin3_size),
            nn.ReLU(),
            nn.Linear(lin3_size, lin3_size),
            nn.ReLU(),
            nn.Linear(lin3_size, lin3_size),
            nn.ReLU(),
            nn.Linear(lin3_size, output_size),
        )

    def forward(self, x):
        
        return self.model(x)

class H15_ReLU(nn.Module):
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
            nn.Linear(lin3_size, lin3_size),
            nn.ReLU(),
            nn.Linear(lin3_size, lin3_size),
            nn.ReLU(),
            nn.Linear(lin3_size, lin3_size),
            nn.ReLU(),
            nn.Linear(lin3_size, lin3_size),
            nn.ReLU(),
            nn.Linear(lin3_size, lin3_size),
            nn.ReLU(),
            nn.Linear(lin3_size, lin3_size),
            nn.ReLU(),
            nn.Linear(lin3_size, lin3_size),
            nn.ReLU(),
            nn.Linear(lin3_size, lin3_size),
            nn.ReLU(),
            nn.Linear(lin3_size, lin3_size),
            nn.ReLU(),
            nn.Linear(lin3_size, lin3_size),
            nn.ReLU(),
            nn.Linear(lin3_size, lin3_size),
            nn.ReLU(),
            nn.Linear(lin3_size, lin3_size),
            nn.ReLU(),
            nn.Linear(lin3_size, output_size),
        )

    def forward(self, x):
        
        return self.model(x)


class H3_SiLU(nn.Module):
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
            nn.SiLU(),
            nn.Linear(lin1_size, lin2_size),
            nn.SiLU(),
            nn.Linear(lin2_size, lin3_size),
            nn.SiLU(),
            nn.Linear(lin3_size, output_size),
        )

    def forward(self, x):
        
        return self.model(x)
    
class H4_SiLU(nn.Module):
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
            nn.SiLU(),
            nn.Linear(lin1_size, lin2_size),
            nn.SiLU(),
            nn.Linear(lin2_size, lin3_size),
            nn.SiLU(),
            nn.Linear(lin3_size, lin3_size),
            nn.SiLU(),
            nn.Linear(lin3_size, output_size),
        )

    def forward(self, x):
        
        return self.model(x)

class H5_SiLU(nn.Module):
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
            nn.SiLU(),
            nn.Linear(lin1_size, lin2_size),
            nn.SiLU(),
            nn.Linear(lin2_size, lin3_size),
            nn.SiLU(),
            nn.Linear(lin3_size, lin3_size),
            nn.SiLU(),
            nn.Linear(lin3_size, lin3_size),
            nn.SiLU(),
            nn.Linear(lin3_size, output_size),
        )

    def forward(self, x):
        
        return self.model(x)


class H10_SiLU(nn.Module):
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
            nn.SiLU(),
            nn.Linear(lin1_size, lin2_size),
            nn.SiLU(),
            nn.Linear(lin2_size, lin3_size),
            nn.SiLU(),
            nn.Linear(lin3_size, lin3_size),
            nn.SiLU(),
            nn.Linear(lin3_size, lin3_size),
            nn.SiLU(),
            nn.Linear(lin3_size, lin3_size),
            nn.SiLU(),
            nn.Linear(lin3_size, lin3_size),
            nn.SiLU(),
            nn.Linear(lin3_size, lin3_size),
            nn.SiLU(),
            nn.Linear(lin3_size, lin3_size),
            nn.SiLU(),
            nn.Linear(lin3_size, lin3_size),
            nn.SiLU(),
            nn.Linear(lin3_size, output_size),
        )

    def forward(self, x):
        
        return self.model(x)

class H15_SiLU(nn.Module):
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
            nn.SiLU(),
            nn.Linear(lin1_size, lin2_size),
            nn.SiLU(),
            nn.Linear(lin2_size, lin3_size),
            nn.SiLU(),
            nn.Linear(lin3_size, lin3_size),
            nn.SiLU(),
            nn.Linear(lin3_size, lin3_size),
            nn.SiLU(),
            nn.Linear(lin3_size, lin3_size),
            nn.SiLU(),
            nn.Linear(lin3_size, lin3_size),
            nn.SiLU(),
            nn.Linear(lin3_size, lin3_size),
            nn.SiLU(),
            nn.Linear(lin3_size, lin3_size),
            nn.SiLU(),
            nn.Linear(lin3_size, lin3_size),
            nn.SiLU(),
            nn.Linear(lin3_size, lin3_size),
            nn.SiLU(),
            nn.Linear(lin3_size, lin3_size),
            nn.SiLU(),
            nn.Linear(lin3_size, lin3_size),
            nn.SiLU(),
            nn.Linear(lin3_size, lin3_size),
            nn.SiLU(),
            nn.Linear(lin3_size, lin3_size),
            nn.SiLU(),
            nn.Linear(lin3_size, output_size),
        )

    def forward(self, x):
        
        return self.model(x)


class H3_LeakyReLU(nn.Module):
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
            nn.LeakyReLU(),
            nn.Linear(lin1_size, lin2_size),
            nn.LeakyReLU(),
            nn.Linear(lin2_size, lin3_size),
            nn.LeakyReLU(),
            nn.Linear(lin3_size, output_size),
        )

    def forward(self, x):
        
        return self.model(x)
    
class H4_LeakyReLU(nn.Module):
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
            nn.LeakyReLU(),
            nn.Linear(lin1_size, lin2_size),
            nn.LeakyReLU(),
            nn.Linear(lin2_size, lin3_size),
            nn.LeakyReLU(),
            nn.Linear(lin3_size, lin3_size),
            nn.LeakyReLU(),
            nn.Linear(lin3_size, output_size),
        )

    def forward(self, x):
        
        return self.model(x)

class H5_LeakyReLU(nn.Module):
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
            nn.LeakyReLU(),
            nn.Linear(lin1_size, lin2_size),
            nn.LeakyReLU(),
            nn.Linear(lin2_size, lin3_size),
            nn.LeakyReLU(),
            nn.Linear(lin3_size, lin3_size),
            nn.LeakyReLU(),
            nn.Linear(lin3_size, lin3_size),
            nn.LeakyReLU(),
            nn.Linear(lin3_size, output_size),
        )

    def forward(self, x):
        
        return self.model(x)


class H10_LeakyReLU(nn.Module):
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
            nn.LeakyReLU(),
            nn.Linear(lin1_size, lin2_size),
            nn.LeakyReLU(),
            nn.Linear(lin2_size, lin3_size),
            nn.LeakyReLU(),
            nn.Linear(lin3_size, lin3_size),
            nn.LeakyReLU(),
            nn.Linear(lin3_size, lin3_size),
            nn.LeakyReLU(),
            nn.Linear(lin3_size, lin3_size),
            nn.LeakyReLU(),
            nn.Linear(lin3_size, lin3_size),
            nn.LeakyReLU(),
            nn.Linear(lin3_size, lin3_size),
            nn.LeakyReLU(),
            nn.Linear(lin3_size, lin3_size),
            nn.LeakyReLU(),
            nn.Linear(lin3_size, lin3_size),
            nn.LeakyReLU(),
            nn.Linear(lin3_size, output_size),
        )

    def forward(self, x):
        
        return self.model(x)

class H15_LeakyReLU(nn.Module):
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
            nn.LeakyReLU(),
            nn.Linear(lin1_size, lin2_size),
            nn.LeakyReLU(),
            nn.Linear(lin2_size, lin3_size),
            nn.LeakyReLU(),
            nn.Linear(lin3_size, lin3_size),
            nn.LeakyReLU(),
            nn.Linear(lin3_size, lin3_size),
            nn.LeakyReLU(),
            nn.Linear(lin3_size, lin3_size),
            nn.LeakyReLU(),
            nn.Linear(lin3_size, lin3_size),
            nn.LeakyReLU(),
            nn.Linear(lin3_size, lin3_size),
            nn.LeakyReLU(),
            nn.Linear(lin3_size, lin3_size),
            nn.LeakyReLU(),
            nn.Linear(lin3_size, lin3_size),
            nn.LeakyReLU(),
            nn.Linear(lin3_size, lin3_size),
            nn.LeakyReLU(),
            nn.Linear(lin3_size, lin3_size),
            nn.LeakyReLU(),
            nn.Linear(lin3_size, lin3_size),
            nn.LeakyReLU(),
            nn.Linear(lin3_size, lin3_size),
            nn.LeakyReLU(),
            nn.Linear(lin3_size, lin3_size),
            nn.LeakyReLU(),
            nn.Linear(lin3_size, output_size),
        )

    def forward(self, x):
        
        return self.model(x)


class H3_ELU(nn.Module):
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
            nn.ELU(),
            nn.Linear(lin1_size, lin2_size),
            nn.ELU(),
            nn.Linear(lin2_size, lin3_size),
            nn.ELU(),
            nn.Linear(lin3_size, output_size),
        )

    def forward(self, x):
        
        return self.model(x)
    
class H4_ELU(nn.Module):
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
            nn.ELU(),
            nn.Linear(lin1_size, lin2_size),
            nn.ELU(),
            nn.Linear(lin2_size, lin3_size),
            nn.ELU(),
            nn.Linear(lin3_size, lin3_size),
            nn.ELU(),
            nn.Linear(lin3_size, output_size),
        )

    def forward(self, x):
        
        return self.model(x)

class H5_ELU(nn.Module):
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
            nn.ELU(),
            nn.Linear(lin1_size, lin2_size),
            nn.ELU(),
            nn.Linear(lin2_size, lin3_size),
            nn.ELU(),
            nn.Linear(lin3_size, lin3_size),
            nn.ELU(),
            nn.Linear(lin3_size, lin3_size),
            nn.ELU(),
            nn.Linear(lin3_size, output_size),
        )

    def forward(self, x):
        
        return self.model(x)


class H10_ELU(nn.Module):
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
            nn.ELU(),
            nn.Linear(lin1_size, lin2_size),
            nn.ELU(),
            nn.Linear(lin2_size, lin3_size),
            nn.ELU(),
            nn.Linear(lin3_size, lin3_size),
            nn.ELU(),
            nn.Linear(lin3_size, lin3_size),
            nn.ELU(),
            nn.Linear(lin3_size, lin3_size),
            nn.ELU(),
            nn.Linear(lin3_size, lin3_size),
            nn.ELU(),
            nn.Linear(lin3_size, lin3_size),
            nn.ELU(),
            nn.Linear(lin3_size, lin3_size),
            nn.ELU(),
            nn.Linear(lin3_size, lin3_size),
            nn.ELU(),
            nn.Linear(lin3_size, output_size),
        )

    def forward(self, x):
        
        return self.model(x)

class H15_ELU(nn.Module):
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
            nn.ELU(),
            nn.Linear(lin1_size, lin2_size),
            nn.ELU(),
            nn.Linear(lin2_size, lin3_size),
            nn.ELU(),
            nn.Linear(lin3_size, lin3_size),
            nn.ELU(),
            nn.Linear(lin3_size, lin3_size),
            nn.ELU(),
            nn.Linear(lin3_size, lin3_size),
            nn.ELU(),
            nn.Linear(lin3_size, lin3_size),
            nn.ELU(),
            nn.Linear(lin3_size, lin3_size),
            nn.ELU(),
            nn.Linear(lin3_size, lin3_size),
            nn.ELU(),
            nn.Linear(lin3_size, lin3_size),
            nn.ELU(),
            nn.Linear(lin3_size, lin3_size),
            nn.ELU(),
            nn.Linear(lin3_size, lin3_size),
            nn.ELU(),
            nn.Linear(lin3_size, lin3_size),
            nn.ELU(),
            nn.Linear(lin3_size, lin3_size),
            nn.ELU(),
            nn.Linear(lin3_size, lin3_size),
            nn.ELU(),
            nn.Linear(lin3_size, output_size),
        )

    def forward(self, x):
        
        return self.model(x)




class H3_Softshrink(nn.Module):
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
            nn.Softshrink(),
            nn.Linear(lin1_size, lin2_size),
            nn.Softshrink(),
            nn.Linear(lin2_size, lin3_size),
            nn.Softshrink(),
            nn.Linear(lin3_size, output_size),
        )

    def forward(self, x):
        
        return self.model(x)
    
class H4_Softshrink(nn.Module):
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
            nn.Softshrink(),
            nn.Linear(lin1_size, lin2_size),
            nn.Softshrink(),
            nn.Linear(lin2_size, lin3_size),
            nn.Softshrink(),
            nn.Linear(lin3_size, lin3_size),
            nn.Softshrink(),
            nn.Linear(lin3_size, output_size),
        )

    def forward(self, x):
        
        return self.model(x)

class H5_Softshrink(nn.Module):
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
            nn.Softshrink(),
            nn.Linear(lin1_size, lin2_size),
            nn.Softshrink(),
            nn.Linear(lin2_size, lin3_size),
            nn.Softshrink(),
            nn.Linear(lin3_size, lin3_size),
            nn.Softshrink(),
            nn.Linear(lin3_size, lin3_size),
            nn.Softshrink(),
            nn.Linear(lin3_size, output_size),
        )

    def forward(self, x):
        
        return self.model(x)


class H10_Softshrink(nn.Module):
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
            nn.Softshrink(),
            nn.Linear(lin1_size, lin2_size),
            nn.Softshrink(),
            nn.Linear(lin2_size, lin3_size),
            nn.Softshrink(),
            nn.Linear(lin3_size, lin3_size),
            nn.Softshrink(),
            nn.Linear(lin3_size, lin3_size),
            nn.Softshrink(),
            nn.Linear(lin3_size, lin3_size),
            nn.Softshrink(),
            nn.Linear(lin3_size, lin3_size),
            nn.Softshrink(),
            nn.Linear(lin3_size, lin3_size),
            nn.Softshrink(),
            nn.Linear(lin3_size, lin3_size),
            nn.Softshrink(),
            nn.Linear(lin3_size, lin3_size),
            nn.Softshrink(),
            nn.Linear(lin3_size, output_size),
        )

    def forward(self, x):
        
        return self.model(x)

class H15_Softshrink(nn.Module):
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
            nn.Softshrink(),
            nn.Linear(lin1_size, lin2_size),
            nn.Softshrink(),
            nn.Linear(lin2_size, lin3_size),
            nn.Softshrink(),
            nn.Linear(lin3_size, lin3_size),
            nn.Softshrink(),
            nn.Linear(lin3_size, lin3_size),
            nn.Softshrink(),
            nn.Linear(lin3_size, lin3_size),
            nn.Softshrink(),
            nn.Linear(lin3_size, lin3_size),
            nn.Softshrink(),
            nn.Linear(lin3_size, lin3_size),
            nn.Softshrink(),
            nn.Linear(lin3_size, lin3_size),
            nn.Softshrink(),
            nn.Linear(lin3_size, lin3_size),
            nn.Softshrink(),
            nn.Linear(lin3_size, lin3_size),
            nn.Softshrink(),
            nn.Linear(lin3_size, lin3_size),
            nn.Softshrink(),
            nn.Linear(lin3_size, lin3_size),
            nn.Softshrink(),
            nn.Linear(lin3_size, lin3_size),
            nn.Softshrink(),
            nn.Linear(lin3_size, lin3_size),
            nn.Softshrink(),
            nn.Linear(lin3_size, output_size),
        )

    def forward(self, x):
        
        return self.model(x)



