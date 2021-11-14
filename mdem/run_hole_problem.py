from mdem import define_hole as des
from utilities import utility as util
from torch.autograd import grad
from utilities.integration_loss import *
import numpy as np
import torch
import torch.nn as nn


dev = torch.device('cpu')
if torch.cuda.is_available():
    print("CUDA is available, running on GPU")
    dev = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    print("CUDA not available, running on CPU")


E = 1000
nu = 0.3
mu = E / (2 * (1 + nu))
lam = (E * nu) / ((1 + nu) * (1 - 2 * nu))



class MultiLayerNet(torch.nn.Module):
    def __init__(self, inp, out, activation, num_hidden_units=60, num_layers=6):
        super(MultiLayerNet, self).__init__()
        self.fc1 = nn.Linear(inp, num_hidden_units, bias=True)
        self.fc2 = nn.ModuleList()
        for i in range(num_layers):
            self.fc2.append(nn.Linear(num_hidden_units, num_hidden_units, bias=True))
        self.fc3 = nn.Linear(num_hidden_units, out,bias=True)
        self.activation = activation


    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        for fc in self.fc2:
            x = fc(x)
            x = self.activation(x)
        x = self.fc3(x)
        return x


def NeoHookean2D( u, x):

    mu = E / (2 * (1 + nu))
    lam = (E * nu) / ((1 + nu) * (1 - 2 * nu))
    duxdxy = grad(u[:, 0].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True,
                  retain_graph=True)[0]
    duydxy = grad(u[:, 1].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True,
                  retain_graph=True)[0]
    Fxx1 = duxdxy[:, 0].unsqueeze(1) + 1
    Fxy1 = duxdxy[:, 1].unsqueeze(1) + 0
    Fyx1 = duydxy[:, 0].unsqueeze(1) + 0
    Fyy1 = duydxy[:, 1].unsqueeze(1) + 1

    detF = Fxx1 * Fyy1 - Fxy1 * Fyx1
    trC = Fxx1 ** 2 + Fxy1 ** 2 + Fyx1 ** 2 + Fyy1 ** 2


    strainEnergy = 0.5 * lam * (torch.log(detF) * torch.log(detF)) - mu * torch.log(detF) + 0.5 * mu * (trC - 2)

    sum = torch.sum(strainEnergy)
    if torch.isnan(sum):
        print('insna')

    return strainEnergy



class DeepEnergyMethod:
    # Instance attributes
    def __init__(self,inp, out, activation, num_hidden_units, num_layers, dim):
        # self.data = data
        self.model = MultiLayerNet( inp, out, activation, num_hidden_units, num_layers)
        self.model = self.model.to(dev)
        self.intLoss = IntegrationLoss('simpson', 2)
        self.dim = dim
        self.lossArray = []



    def train_modelLFBGS(self, data, neumannBC, dirichletBC,delaunayIntegration,tractionFreeCircle, iteration, learning_rate,dataPlot):


        x = torch.from_numpy(data).float()
        x = x.to(dev)
        x.requires_grad_(True)


        dirBC_coordinates_left = torch.from_numpy(dirichletBC['dirichlet_1']['coord_left']).float().to(dev)
        dirBC_values_left = torch.from_numpy(dirichletBC['dirichlet_1']['known_value_left']).float().to(dev)

        dirBC_coordinates_bottom = torch.from_numpy(dirichletBC['dirichlet_1']['coord_bottom']).float().to(dev)
        dirBC_values_bottom = torch.from_numpy(dirichletBC['dirichlet_1']['known_value_bottom']).float().to(dev)


        neuBC_coordinates = {}
        neuBC_values = {}

        for i, keyi in enumerate(neumannBC):
            neuBC_coordinates[i] = torch.from_numpy(neumannBC[keyi]['coord']).float().to(dev)
            neuBC_coordinates[i].requires_grad_(True)
            neuBC_values[i] = torch.from_numpy(neumannBC[keyi]['known_value']).float().to(dev)



        loss_fct = torch.nn.MSELoss()

        pointsP11 = torch.from_numpy(neumannBC['neumann_1']['bcpPoints']['coordsP11']).float().to(dev)
        valP11 = torch.from_numpy(neumannBC['neumann_1']['bcpPoints']['valP11']).float().to(dev)
        pointsP11.requires_grad_(True)

        pointsP12 = torch.from_numpy(neumannBC['neumann_1']['bcpPoints']['coordsP12']).float().to(dev)
        valP12 = torch.from_numpy(neumannBC['neumann_1']['bcpPoints']['valP12']).float().to(dev)
        pointsP12.requires_grad_(True)


        pointsP21 = torch.from_numpy(neumannBC['neumann_1']['bcpPoints']['coordsP21']).float().to(dev)
        valP21 = torch.from_numpy(neumannBC['neumann_1']['bcpPoints']['valP21']).float().to(dev)
        pointsP21.requires_grad_(True)


        pointsP22 = torch.from_numpy(neumannBC['neumann_1']['bcpPoints']['coordsP22']).float().to(dev)
        valP22 = torch.from_numpy(neumannBC['neumann_1']['bcpPoints']['valP22']).float().to(dev)
        pointsP22.requires_grad_(True)


        del1 = delaunayIntegration['del1']


        circlePoints = torch.from_numpy(tractionFreeCircle['coord']).float()
        circleNormals = torch.from_numpy(tractionFreeCircle['norm']).float()

        circlePoints = circlePoints.to(dev)
        circleNormals = circleNormals.to(dev)

        circlePoints.requires_grad_(True)
        circleNormals.requires_grad_(True)



        optimizer = torch.optim.LBFGS(self.model.parameters(), lr=learning_rate, max_iter=20)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)

        dataPlot = torch.from_numpy(dataPlot).to(dev)
        dataPlot = dataPlot.float()
        for t in range(iteration):
            def closure():
                u_pred = self.getU(x)
                u_pred.double()

                storedEnergy = NeoHookean2D(u_pred, x)

                internal = del1.getVolume(storedEnergy)
                external = torch.zeros(len(neuBC_coordinates))
                for i, vali in enumerate(neuBC_coordinates):
                    neu_u_pred = self.getU(neuBC_coordinates[i])[:, 0:2]
                    fext = torch.bmm((neu_u_pred + neuBC_coordinates[i]).unsqueeze(1), neuBC_values[i].unsqueeze(2))
                    DX = torch.abs(neuBC_coordinates[0][1, 0] - neuBC_coordinates[0][0, 0])
                    external[i] = self.intLoss.lossExternalEnergy(fext, dx=DX)

                energy_loss = internal - torch.sum(external)

                dir_u_pred_left = self.getU(dirBC_coordinates_left)[:, 0:1]
                bc_u_crit_left = self.loss_squared_sum(dir_u_pred_left, dirBC_values_left[:, 0:1])

                dir_u_pred_bottom = self.getU(dirBC_coordinates_bottom)[:, 0:2]
                bc_u_crit_bottom = self.loss_squared_sum(dir_u_pred_bottom, dirBC_values_bottom[:, 0:2])

                duxdxy = grad(u_pred[:, 0].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True,
                              retain_graph=True)[0]
                duydxy = grad(u_pred[:, 1].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True,
                              retain_graph=True)[0]
                Pxx_out = u_pred[:, 2]
                Pxy_out = u_pred[:, 3]
                Pyx_out = u_pred[:, 4]
                Pyy_out = u_pred[:, 5]

                dP11dxy = grad(Pxx_out.unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev),
                               create_graph=True, retain_graph=True)[0]
                dP12dxy = grad(Pxy_out.unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev),
                               create_graph=True, retain_graph=True)[0]

                dP21dxy = grad(Pyx_out.unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev),
                               create_graph=True, retain_graph=True)[0]
                dP22dxy = grad(Pyy_out.unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev),
                               create_graph=True, retain_graph=True)[0]

                out1 = dP11dxy[:, 0] + dP12dxy[:, 1]
                out2 = dP21dxy[:, 0] + dP22dxy[:, 1]

                F11 = duxdxy[:, 0].unsqueeze(1) + 1
                F12 = duxdxy[:, 1].unsqueeze(1) + 0
                F21 = duydxy[:, 0].unsqueeze(1) + 0
                F22 = duydxy[:, 1].unsqueeze(1) + 1

                detF = F11 * F22 - F12 * F21
                invF11 = F22 / detF
                invF22 = F11 / detF
                invF12 = -F12 / detF
                invF21 = -F21 / detF

                P11 = mu * F11 + (lam * torch.log(detF) - mu) * invF11
                P12 = mu * F12 + (lam * torch.log(detF) - mu) * invF21
                P21 = mu * F21 + (lam * torch.log(detF) - mu) * invF12
                P22 = mu * F22 + (lam * torch.log(detF) - mu) * invF22

                dP11indxy = grad(P11, x, torch.ones(x.size()[0], 1, device=dev),
                                 create_graph=True, retain_graph=True)[0]
                dP12indxy = grad(P12, x, torch.ones(x.size()[0], 1, device=dev),
                                 create_graph=True, retain_graph=True)[0]

                dP21indxy = grad(P21, x, torch.ones(x.size()[0], 1, device=dev),
                                 create_graph=True, retain_graph=True)[0]
                dP22indxy = grad(P22, x, torch.ones(x.size()[0], 1, device=dev),
                                 create_graph=True, retain_graph=True)[0]

                outin1 = dP11indxy[:, 0] + dP12indxy[:, 1]
                outin2 = dP21indxy[:, 0] + dP22indxy[:, 1]

                Pxx_loss = loss_fct(Pxx_out, P11[:, 0])
                Pxy_loss = loss_fct(Pxy_out, P12[:, 0])
                Pyx_loss = loss_fct(Pyx_out, P21[:, 0])
                Pyy_loss = loss_fct(Pyy_out, P22[:, 0])

                fcirc = self.getU(circlePoints)

                t1 = torch.multiply(fcirc[:, 2], circleNormals[:, 0]) + torch.multiply(fcirc[:, 3], circleNormals[:, 1])
                t2 = torch.multiply(fcirc[:, 4], circleNormals[:, 0]) + torch.multiply(fcirc[:, 5], circleNormals[:, 1])

                t1_loss = self.loss_sum(torch.pow(t1, 2))
                t2_loss = self.loss_sum(torch.pow(t2, 2))

                loss_sec1BCP11 = self.getLossStressesBoundary(pointsP11, valP11, 'P11')
                loss_sec1BCP12 = self.getLossStressesBoundary(pointsP12, valP12, 'P12')
                loss_sec1BCP21 = self.getLossStressesBoundary(pointsP21, valP21, 'P21')
                loss_sec1BCP22 = self.getLossStressesBoundary(pointsP22, valP22, 'P22')

                loscBC = loss_sec1BCP11 + loss_sec1BCP12 + loss_sec1BCP21 + loss_sec1BCP22

                boundary_loss = torch.sum(bc_u_crit_left + bc_u_crit_bottom)
                loss = energy_loss + boundary_loss + (
                            Pxx_loss + Pxy_loss + Pyx_loss + Pyy_loss) + loscBC + t1_loss + t2_loss + self.loss_sum(
                    torch.pow(out1, 2)) + self.loss_sum(
                    torch.pow(out2, 2)) + self.loss_sum(torch.pow(outin1, 2)) + self.loss_sum(
                    torch.pow(outin2, 2))
                optimizer.zero_grad()
                loss.backward()
                if t%10==0:
                    print('Iter: %d Loss: %.9e Energy: %.9e Boundary: %.9e'
                        % (t + 1, loss.item(), energy_loss.item(), boundary_loss.item()))

                return loss
            optimizer.step(closure)

            if t % 5 == 0:
                filename_out = './output/vtk_files/LFBGSHole' + str(t)
                z = np.array([0])
                U, S11, S12, S13, S22, S23, S33, E11, E12, E13, E22, E23, E33, SVonMises, F11, F12, F21, F22, P11, P12, P13, P21, P22, P23, P31, P32, P33 = self.evaluate_model(dataPlot)
                des.save_WithHole(dataPlot.cpu().detach().numpy(), filename_out, U, S11, S12, S13, S22, S23, S33, E11, E12, E13, E22, E23,
                                  E33, SVonMises,
                                  F11, F12, F21, F22, P11, P12, P13, P21, P22, P23, P31, P32, P33)




    def getLossStressesBoundary(self, points, val, whichP):
        u = self.getU(points)
        loss_fct = torch.nn.MSELoss()
        if whichP == 'P11':
            P = u[:, 2:3]
        elif whichP == 'P21':
            P = u[:, 4:5]

        elif whichP== 'P12':
            P = u[:, 3:4]

        elif whichP == 'P22':
            P = u[:, 5:6]

        loss = loss_fct(P, val)
        return loss


    def train_modelADAM(self, data, neumannBC, dirichletBC,delaunayIntegration,tractionFreeCircle, iteration, learning_rate,dataPlot):


        x = torch.from_numpy(data).float()
        x = x.to(dev)
        x.requires_grad_(True)


        dirBC_coordinates_left = torch.from_numpy(dirichletBC['dirichlet_1']['coord_left']).float().to(dev)
        dirBC_values_left = torch.from_numpy(dirichletBC['dirichlet_1']['known_value_left']).float().to(dev)

        dirBC_coordinates_bottom = torch.from_numpy(dirichletBC['dirichlet_1']['coord_bottom']).float().to(dev)
        dirBC_values_bottom = torch.from_numpy(dirichletBC['dirichlet_1']['known_value_bottom']).float().to(dev)



        neuBC_coordinates = {}
        neuBC_values = {}

        for i, keyi in enumerate(neumannBC):
            neuBC_coordinates[i] = torch.from_numpy(neumannBC[keyi]['coord']).float().to(dev)
            neuBC_coordinates[i].requires_grad_(True)
            neuBC_values[i] = torch.from_numpy(neumannBC[keyi]['known_value']).float().to(dev)


        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50000, gamma=0.8)


        loss_fct = torch.nn.MSELoss()

        pointsP11 = torch.from_numpy(neumannBC['neumann_1']['bcpPoints']['coordsP11']).float().to(dev)
        valP11 = torch.from_numpy(neumannBC['neumann_1']['bcpPoints']['valP11']).float().to(dev)
        pointsP11.requires_grad_(True)

        pointsP12 = torch.from_numpy(neumannBC['neumann_1']['bcpPoints']['coordsP12']).float().to(dev)
        valP12 = torch.from_numpy(neumannBC['neumann_1']['bcpPoints']['valP12']).float().to(dev)
        pointsP12.requires_grad_(True)


        pointsP21 = torch.from_numpy(neumannBC['neumann_1']['bcpPoints']['coordsP21']).float().to(dev)
        valP21 = torch.from_numpy(neumannBC['neumann_1']['bcpPoints']['valP21']).float().to(dev)
        pointsP21.requires_grad_(True)


        pointsP22 = torch.from_numpy(neumannBC['neumann_1']['bcpPoints']['coordsP22']).float().to(dev)
        valP22 = torch.from_numpy(neumannBC['neumann_1']['bcpPoints']['valP22']).float().to(dev)
        pointsP22.requires_grad_(True)


        del1 = delaunayIntegration['del1']


        circlePoints = torch.from_numpy(tractionFreeCircle['coord']).float()
        circleNormals = torch.from_numpy(tractionFreeCircle['norm']).float()

        circlePoints = circlePoints.to(dev)
        circleNormals = circleNormals.to(dev)

        circlePoints.requires_grad_(True)
        circleNormals.requires_grad_(True)

        dataPlot = torch.from_numpy(dataPlot).to(dev)
        dataPlot = dataPlot.float()
        for t in range(iteration):
            u_pred = self.getU(x)
            u_pred.double()

            storedEnergy = NeoHookean2D(u_pred, x)

            internal = del1.getVolume(storedEnergy)
            external = torch.zeros(len(neuBC_coordinates))
            for i, vali in enumerate(neuBC_coordinates):
                neu_u_pred = self.getU(neuBC_coordinates[i])[:, 0:2]
                fext = torch.bmm((neu_u_pred + neuBC_coordinates[i]).unsqueeze(1), neuBC_values[i].unsqueeze(2))
                DX = torch.abs(neuBC_coordinates[0][1,0] - neuBC_coordinates[0][0,0])
                external[i] = self.intLoss.lossExternalEnergy(fext, dx=DX)

            energy_loss = internal - torch.sum(external)

            dir_u_pred_left = self.getU(dirBC_coordinates_left)[:, 0:1]
            bc_u_crit_left = self.loss_squared_sum(dir_u_pred_left, dirBC_values_left[:, 0:1])

            dir_u_pred_bottom = self.getU(dirBC_coordinates_bottom)[:, 0:2]
            bc_u_crit_bottom = self.loss_squared_sum(dir_u_pred_bottom, dirBC_values_bottom[:, 0:2])

            duxdxy = grad(u_pred[:, 0].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True,
                          retain_graph=True)[0]
            duydxy = grad(u_pred[:, 1].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True,
                          retain_graph=True)[0]
            Pxx_out = u_pred[:, 2]
            Pxy_out = u_pred[:, 3]
            Pyx_out = u_pred[:, 4]
            Pyy_out = u_pred[:, 5]


            dP11dxy = grad(Pxx_out.unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev),
                           create_graph=True, retain_graph=True)[0]
            dP12dxy = grad(Pxy_out.unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev),
                           create_graph=True, retain_graph=True)[0]

            dP21dxy = grad(Pyx_out.unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev),
                           create_graph=True, retain_graph=True)[0]
            dP22dxy = grad(Pyy_out.unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev),
                           create_graph=True, retain_graph=True)[0]

            out1 = dP11dxy[:, 0] + dP12dxy[:, 1]
            out2 = dP21dxy[:, 0] + dP22dxy[:, 1]





            F11 = duxdxy[:, 0].unsqueeze(1) + 1
            F12 = duxdxy[:, 1].unsqueeze(1) + 0
            F21 = duydxy[:, 0].unsqueeze(1) + 0
            F22 = duydxy[:, 1].unsqueeze(1) + 1

            detF = F11 * F22 - F12 * F21
            invF11 = F22 / detF
            invF22 = F11 / detF
            invF12 = -F12 / detF
            invF21 = -F21 / detF

            P11 = mu * F11 + (lam * torch.log(detF) - mu) * invF11
            P12 = mu * F12 + (lam * torch.log(detF) - mu) * invF21
            P21 = mu * F21 + (lam * torch.log(detF) - mu) * invF12
            P22 = mu * F22 + (lam * torch.log(detF) - mu) * invF22


            dP11indxy = grad(P11, x, torch.ones(x.size()[0], 1, device=dev),
                           create_graph=True, retain_graph=True)[0]
            dP12indxy = grad(P12, x, torch.ones(x.size()[0], 1, device=dev),
                           create_graph=True, retain_graph=True)[0]

            dP21indxy = grad(P21, x, torch.ones(x.size()[0], 1, device=dev),
                           create_graph=True, retain_graph=True)[0]
            dP22indxy = grad(P22, x, torch.ones(x.size()[0], 1, device=dev),
                           create_graph=True, retain_graph=True)[0]

            outin1 = dP11indxy[:, 0] + dP12indxy[:, 1]
            outin2 = dP21indxy[:, 0] + dP22indxy[:, 1]




            Pxx_loss = loss_fct(Pxx_out, P11[:, 0])
            Pxy_loss = loss_fct(Pxy_out, P12[:, 0])
            Pyx_loss = loss_fct(Pyx_out, P21[:, 0])
            Pyy_loss = loss_fct(Pyy_out, P22[:, 0])


            fcirc = self.getU(circlePoints)

            t1 = torch.multiply(fcirc[:,2], circleNormals[:,0]) + torch.multiply(fcirc[:,3], circleNormals[:,1])
            t2 = torch.multiply(fcirc[:,4], circleNormals[:,0]) + torch.multiply(fcirc[:,5], circleNormals[:,1])

            t1_loss = self.loss_sum(torch.pow(t1, 2))
            t2_loss = self.loss_sum(torch.pow(t2, 2))


            loss_sec1BCP11 = self.getLossStressesBoundary(pointsP11, valP11, 'P11')
            loss_sec1BCP12 = self.getLossStressesBoundary(pointsP12, valP12, 'P12')
            loss_sec1BCP21 = self.getLossStressesBoundary(pointsP21, valP21, 'P21')
            loss_sec1BCP22 = self.getLossStressesBoundary(pointsP22, valP22, 'P22')

            loscBC = loss_sec1BCP11+ loss_sec1BCP12+ loss_sec1BCP21+loss_sec1BCP22


            boundary_loss = torch.sum(bc_u_crit_left + bc_u_crit_bottom)
            loss = energy_loss + boundary_loss +  (Pxx_loss + Pxy_loss + Pyx_loss + Pyy_loss)+loscBC +t1_loss + t2_loss + self.loss_sum(torch.pow(out1, 2)) + self.loss_sum(
                torch.pow(out2, 2)) + self.loss_sum(torch.pow(outin1, 2)) + self.loss_sum(
                torch.pow(outin2, 2))


            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            scheduler.step()
            if t % 100 == 0:
                print(
                    'Iter: %d Loss: %.9e Energy: %.9e Boundary: %.9e'
                    % (t + 1, loss.item(), energy_loss.item(), boundary_loss.item()))

            if t % 2000 == 0:
                filename_out = './output/vtk_files/Hole' + str(t)
                z = np.array([0])
                U, S11, S12, S13, S22, S23, S33, E11, E12, E13, E22, E23, E33, SVonMises, F11, F12, F21, F22, P11, P12, P13, P21, P22, P23, P31, P32, P33 = self.evaluate_model(dataPlot)
                des.save_WithHole(dataPlot.cpu().detach().numpy(), filename_out, U, S11, S12, S13, S22, S23, S33, E11, E12, E13, E22, E23,
                                  E33, SVonMises,
                                  F11, F12, F21, F22, P11, P12, P13, P21, P22, P23, P31, P32, P33)





    def getU(self, x):
        out = self.model(x)

        Ux = x[:, 0].to(dev) * x[:, 1].to(dev) * out[:, 0].to(dev)
        Uy = x[:, 1] * out[:, 1]

        P11 = out[:, 2]

        P12 = out[:, 3]

        P21 =  out[:, 4]
        P22 = out[:, 5]


        Ux = Ux.reshape(Ux.shape[0], 1)
        Uy = Uy.reshape(Uy.shape[0], 1)
        P11 = P11.reshape(Uy.shape[0], 1)
        P12 = P12.reshape(Uy.shape[0], 1)
        P21 = P21.reshape(Uy.shape[0], 1)
        P22 = P22.reshape(Uy.shape[0], 1)
        u_pred = torch.cat((Ux, Uy, P11, P12, P21, P22), -1)
        return u_pred



    def evaluate_model(self, dataPlot):
        dataPlot = dataPlot.to(dev)
        dim = self.dim
        if dim == 2:


            dataPlot.requires_grad_(True)
            u_pred_torch = self.getU(dataPlot)

            duxdxy = grad(u_pred_torch[:, 0].unsqueeze(1), dataPlot, torch.ones(dataPlot.size()[0], 1, device=dev),
                           retain_graph=True)[0]
            duydxy = grad(u_pred_torch[:, 1].unsqueeze(1), dataPlot, torch.ones(dataPlot.size()[0], 1, device=dev),
                           retain_graph=True)[0]

            F11 = duxdxy[:, 0].unsqueeze(1) + 1
            F12 = duxdxy[:, 1].unsqueeze(1) + 0
            F21 = duydxy[:, 0].unsqueeze(1) + 0
            F22 = duydxy[:, 1].unsqueeze(1) + 1
            detF = F11 * F22 - F12 * F21
            invF11 = F22 / detF
            invF22 = F11 / detF
            invF12 = -F12 / detF
            invF21 = -F21 / detF
            C11 = F11 ** 2 + F21 ** 2
            C12 = F11 * F12 + F21 * F22
            C21 = F12 * F11 + F22 * F21
            C22 = F12 ** 2 + F22 ** 2
            E11 = 0.5 * (C11 - 1)
            E12 = 0.5 * C12
            E21 = 0.5 * C21
            E22 = 0.5 * (C22 - 1)
            P11 = u_pred_torch[:, 2:3]
            P12 = u_pred_torch[:, 3:4]
            P21 = u_pred_torch[:, 4:5]
            P22 = u_pred_torch[:, 5:6]
            S11 = invF11 * P11 + invF12 * P21
            S12 = invF11 * P12 + invF12 * P22
            S21 = invF21 * P11 + invF22 * P21
            S22 = invF21 * P12 + invF22 * P22
            u_pred = u_pred_torch.detach().cpu().numpy()

            P11_pred = P11.detach().cpu().numpy()
            P12_pred = P12.detach().cpu().numpy()
            P13_pred = np.zeros(P12_pred.shape)
            P21_pred = P21.detach().cpu().numpy()
            P22_pred = P22.detach().cpu().numpy()
            P23_pred = np.zeros(P12_pred.shape)
            P31_pred = np.zeros(P12_pred.shape)
            P32_pred = np.zeros(P12_pred.shape)
            P33_pred = np.zeros(P12_pred.shape)

            F11_pred = F11.detach().cpu().numpy()
            F12_pred = F12.detach().cpu().numpy()
            F13_pred = np.zeros(F11_pred.shape)
            F21_pred = F21.detach().cpu().numpy()
            F22_pred = F22.detach().cpu().numpy()
            F23_pred = np.zeros(F11_pred.shape)
            F31_pred = np.zeros(F11_pred.shape)
            F32_pred = np.zeros(F11_pred.shape)
            F33_pred = np.zeros(F11_pred.shape)


            E11_pred = E11.detach().cpu().numpy()
            E12_pred = E12.detach().cpu().numpy()
            E13_pred = np.zeros(E12_pred.shape)
            E21_pred = E21.detach().cpu().numpy()
            E22_pred = E22.detach().cpu().numpy()
            E23_pred = np.zeros(E12_pred.shape)
            E33_pred = np.zeros(E12_pred.shape)


            S11_pred = S11.detach().cpu().numpy()
            S12_pred = S12.detach().cpu().numpy()
            S13_pred = np.zeros(S12_pred.shape)
            S21_pred = S21.detach().cpu().numpy()
            S22_pred = S22.detach().cpu().numpy()
            S23_pred = np.zeros(S12_pred.shape)
            S33_pred = np.zeros(S12_pred.shape)


            Ux_pred = u_pred[:,0]
            Uy_pred = u_pred[:,1]
            Uz_pred = np.zeros(Ux_pred.shape)

            SVonMises = np.float64(
                np.sqrt(0.5 * ((S11_pred - S22_pred) ** 2 + (S22_pred) ** 2 + (-S11_pred) ** 2 + 6 * (S12_pred ** 2))))
            U = (np.float64(u_pred[:,0]), np.float64(u_pred[:,1]), np.float64(Uz_pred))
            return U, np.float64(S11_pred), np.float64(S12_pred), np.float64(S13_pred), np.float64(S22_pred), np.float64(
                S23_pred), \
                   np.float64(S33_pred), np.float64(E11_pred), np.float64(E12_pred), \
                   np.float64(E13_pred), np.float64(E22_pred), np.float64(E23_pred), np.float64(E33_pred), np.float64(
                SVonMises), \
                   np.float64(F11_pred), np.float64(F12_pred), np.float64(F21_pred), np.float64(F22_pred), np.float64(
                P11_pred), np.float64(P12_pred), \
                   np.float64(P13_pred), np.float64(P21_pred), np.float64(P22_pred), np.float64(P23_pred), np.float64(
                P31_pred), np.float64(P32_pred), np.float64(P33_pred),
    # --------------------------------------------------------------------------------
    # method: loss sum for the energy part
    # --------------------------------------------------------------------------------
    @staticmethod
    def loss_sum(tinput):
        return torch.sum(tinput) / tinput.data.nelement()

    # --------------------------------------------------------------------------------
    # purpose: loss square sum for the boundary part
    # --------------------------------------------------------------------------------
    @staticmethod
    def loss_squared_sum(tinput, target):
        row, column = tinput.shape
        loss = 0
        for j in range(column):
            loss += torch.sum((tinput[:, j] - target[:, j]) ** 2) / tinput[:, j].data.nelement()
        return loss


def run():
    # ----------------------------------------------------------------------
    #                   STEP 1: SETUP DOMAIN - COLLECT CLEAN DATABASE
    # ----------------------------------------------------------------------
    dom, boundary_neumann, boundary_dirichlet,delaunayIntegration,tractionFreeCircle,distances, inputValues, outputValues = des.setup_domain()
    datatest = des.get_datatest()
    # ----------------------------------------------------------------------
    #                   STEP 2: SETUP MODEL
    # ----------------------------------------------------------------------
    dem = DeepEnergyMethod(2, 6, nn.Tanh(),60,6, 2)

    # ----------------------------------------------------------------------
    #                   STEP 3: TRAINING MODEL
    # ----------------------------------------------------------------------
    iterationADAM =1000000
    lrADAM = 1e-3
    dem.train_modelADAM(dom, boundary_neumann, boundary_dirichlet,delaunayIntegration,tractionFreeCircle, iterationADAM, lrADAM,datatest)

    iterationLFBGS =100
    lrLFBGS= 1e-3
    dem.train_modelLFBGS(dom, boundary_neumann, boundary_dirichlet,delaunayIntegration,tractionFreeCircle, iterationLFBGS, lrLFBGS,datatest)



