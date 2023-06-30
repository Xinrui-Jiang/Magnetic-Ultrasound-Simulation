import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import math
"""
其他材料:
二维波动方程模拟: https://zhuanlan.zhihu.com/p/500450976
基于单个二阶偏微分形式波动方程的模拟: https://wuli.wiki/online/W1dNum.html
"""
"""
#最初手动模拟的版本，看着图一乐
class OneDimensionalFluctuation:
    def __init__(self,nx=150,nt=290,dx=1e-3,boundary1=40,boundary2=80,
                 tao_t_1=1.073,tao_r_1=0.073,tao_t_2=0.927,tao_r_2=-0.073):
        self.nx=nx
        self.nt=nt
        self.dx=dx
        self.x=np.linspace(0,self.nx*self.dx,self.nx)
        np.random.seed(6)

        self.c=3e8
        self.dt = self.nt/self.nx*self.dx
        self.bd1 = boundary1
        self.bd2 = boundary2
        self.rho = np.empty((self.nx,1))
        self.rho[:boundary1] = np.random.uniform(low=0.99, high=1.01, size=(boundary1,1))
        self.rho[boundary1:boundary2] = np.random.uniform(low=0.99, high=1.01, size=(boundary2-boundary1,1))
        self.rho[boundary2:] = np.random.uniform(low=0.99, high=1.01, size=(nx-boundary2,1))
        self.k_fat_1 = 1
        self.k_muscle = 1
        self.k_fat_2 = 1

        self.data_bank = np.empty((6,nt,nx,1))

        self.decay_factor = np.ones((nx,1))
        for i in range(boundary1):
            self.decay_factor[i] = math.exp(-0.0002*i)
        for i in range(boundary1,boundary2):
            self.decay_factor[i] = math.exp(-0.0003*(i-boundary1))
        for i in range(boundary2,nx):
            self.decay_factor[i] = math.exp(-0.0002*(i-boundary2))

        self.decay_factor_reverse = np.ones((nx,1))
        for i in range(boundary1):
            self.decay_factor_reverse[i] = math.exp(-0.0002*(boundary1-i))
        for i in range(boundary1,boundary2):
            self.decay_factor_reverse[i] = math.exp(-0.0003*(boundary2-i))
        for i in range(boundary2,nx):
            self.decay_factor_reverse[i] = math.exp(-0.0002*(nx-i))

        self.p = np.zeros((nx,1))
        self.u = np.zeros((nx,1))
        self.p_reflect1 = np.zeros((nx,1))
        self.u_reflect1 = np.zeros((nx,1))
        self.p_reflect2 = np.zeros((nx,1))
        self.u_reflect2 = np.zeros((nx,1))
        self.p_trans3 = np.zeros((nx,1))
        self.u_trans3 = np.zeros((nx,1))

        self.tao_t_1 = tao_t_1
        self.tao_r_1 = tao_r_1
        self.tao_t_2 = tao_t_2
        self.tao_r_2 = tao_r_2

    def run(self):
        for i in np.arange(0,self.nt):
            #发信号
            if i<41:
                self.p[0]=10*np.sin(np.pi*0.1*i)
            else:
                self.p[0] = 0

            #正常传播
            self.p[1:self.bd1] = self.p[1:self.bd1]/self.rho[1:self.bd1]
            self.u[0:self.bd1-1] = self.u[0:self.bd1-1] - \
                        (self.p[1:self.bd1]-self.p[0:self.bd1-1])/self.rho[0:self.bd1-1]
            self.p[1:self.bd1] = self.p[1:self.bd1] - (self.u[1:self.bd1]-self.u[0:self.bd1-1])/self.k_fat_1
            self.p[1:self.bd1] *= self.decay_factor[1:self.bd1]
            self.u[self.bd1 - 1] = self.u[self.bd1 - 2]


            #透射1
            self.p[self.bd1] = self.tao_t_1 * self.p[self.bd1-1]
            self.u[self.bd1:self.bd2-1] = self.u[self.bd1:self.bd2-1] - \
                (self.p[self.bd1+1:self.bd2]-self.p[self.bd1:self.bd2-1])/self.rho[self.bd1:self.bd2-1]
            self.p[self.bd1+1:self.bd2] = self.p[self.bd1+1:self.bd2] - \
                                          (self.u[self.bd1+1:self.bd2]-self.u[self.bd1:self.bd2-1])/self.k_muscle
            self.p[self.bd1 + 1:self.bd2] *= self.decay_factor[self.bd1+1:self.bd2]
            self.u[self.bd2-1] = self.u[self.bd2-2]
            #self.u[self.bd1-1] = 2*self.u[self.bd1-2] - self.u[self.bd1-1]

            #透射2
            self.p[self.bd2] = self.tao_t_2 * self.p[self.bd2-1]

            self.u[self.bd2:self.nx-1] = self.u[self.bd2:self.nx-1] - \
                                         (self.p[self.bd2+1:self.nx] - self.p[self.bd2:self.nx-1])/self.rho[self.bd2:self.nx-1]
            self.p[self.bd2+1:self.nx] = self.p[self.bd2+1:self.nx] - \
                                         (self.u[self.bd2+1:self.nx] - self.u[self.bd2:self.nx-1])/self.k_fat_2
            self.p[self.bd2+1:self.nx] *= self.decay_factor[self.bd2 + 1:self.nx]
            self.u[self.nx-1] = self.u[self.nx-2]

            #反射1
            self.p_reflect1[self.bd1-1] = self.tao_r_1 * self.p[self.bd1-1]
            self.u_reflect1[1:self.bd1] = self.u_reflect1[1:self.bd1] - \
                                (self.p_reflect1[1:self.bd1] - self.p_reflect1[0:self.bd1 - 1]) / self.rho[1:self.bd1]
            self.p_reflect1[0:self.bd1-1] = self.p_reflect1[0:self.bd1-1] - \
                                (self.u_reflect1[1:self.bd1] - self.u_reflect1[0:self.bd1 - 1])/self.k_fat_1
            self.p_reflect1[0:self.bd1 - 1] *= self.decay_factor_reverse[0:self.bd1-1]
            self.u_reflect1[0] = self.u_reflect1[1]

            #反射2
            self.p_reflect2[self.bd2-1] = self.tao_r_2 * self.p[self.bd2-1]
            self.u_reflect2[self.bd1+1:self.bd2] = self.u_reflect2[self.bd1+1:self.bd2] - \
            (self.p_reflect2[self.bd1+1:self.bd2] - self.p_reflect2[self.bd1:self.bd2 - 1])/self.rho[self.bd1+1:self.bd2]
            self.p_reflect2[self.bd1:self.bd2-1] = self.p_reflect2[self.bd1:self.bd2-1] - \
                        (self.u_reflect2[self.bd1+1:self.bd2] - self.u_reflect2[self.bd1:self.bd2 - 1])/self.k_muscle
            self.p_reflect2[self.bd1:self.bd2 - 1] *= self.decay_factor_reverse[self.bd1:self.bd2-1]
            self.u_reflect2[self.bd1] = self.u_reflect2[self.bd1+1]

            #透射3
            self.p_reflect2[self.bd1-1] = self.tao_t_2 * self.p_reflect2[self.bd1]

            self.u_reflect2[1:self.bd1] = self.u_reflect2[1:self.bd1] - \
                                (self.p_reflect2[1:self.bd1] - self.p_reflect2[0:self.bd1 - 1])/self.rho[1:self.bd1]
            self.p_reflect2[0:self.bd1-1] = self.p_reflect2[0:self.bd1-1] - \
                                        (self.u_reflect2[1:self.bd1] - self.u_reflect2[0:self.bd1 - 1])/self.k_fat_2
            self.p_reflect2[0:self.bd1-1] *= self.decay_factor_reverse[0:self.bd1-1]
            self.u_reflect2[0] = self.u_reflect2[1]

            data_lists = [self.p,self.p_reflect1,self.p_reflect2,self.u,self.u_reflect1,self.u_reflect2]
            for j in range(len(data_lists)):
                self.data_bank[j,i] = data_lists[j]

    def draw(self):
        plt.figure()
        plt.ion()
        for i in range(self.nt):
            plt.clf()
            plt.subplot(2,1,1)
            plt.title('p')
            plt.ylim([-10.1, 10.1])
            plt.plot(self.x,self.data_bank[0,i])
            plt.plot(self.x,self.data_bank[1,i])
            plt.plot(self.x,self.data_bank[2,i])
            rho_sub = np.empty((self.nx,1))
            rho_sub[:self.bd1] = self.rho[:self.bd1]+5
            rho_sub[self.bd1:self.bd2] = self.rho[self.bd1:self.bd2]-5
            rho_sub[self.bd2:] = self.rho[self.bd2:]+5
            plt.plot(self.x, rho_sub)
            plt.subplot(2,1,2)
            plt.title('u')
            plt.ylim([-10.1, 10.1])
            plt.plot(self.x,self.data_bank[3,i])
            plt.plot(self.x,self.data_bank[4,i])
            plt.plot(self.x,self.data_bank[5,i])
            plt.plot(self.x,rho_sub)
            plt.pause(self.dt)

        plt.ioff()
        plt.show()
"""

class OneDimensionalFluctuation:
    def __init__(self,nx=250,nt=350,dx=1e-4,boundary1=50,boundary2=100,
                 tao_t_1=1.073,tao_r_1=0.073,tao_t_2=0.927,tao_r_2=-0.073,CFL=0.79):
        self.nx=nx
        self.nt=nt
        self.dx=dx
        self.x=np.linspace(0,self.nx*self.dx,self.nx)
        np.random.seed(6)

        #设定环境值
        self.c_muscle = 1547
        self.c_fat = 1478
        self.rho_muscle = 1050
        self.rho_fat = 950
        self.k_muscle = 1 / (self.rho_muscle * (self.c_muscle**2))
        self.k_fat = 1 / (self.rho_fat * (self.c_fat**2))
        self.k = np.empty((self.nx, 1))
        self.k[:boundary1] = self.k_fat
        self.k[boundary1:boundary2] = self.k_muscle
        self.k[boundary2:] = self.k_fat

        self.dt = np.empty((nx,1))
        #根据CFL条件设置dt的长度， delta t= delta x / c * CFL.具体解释见报告部分
        self.dt[:boundary1] = (self.dx / self.c_fat) * CFL
        self.dt[boundary1:boundary2] = (self.dx / self.c_muscle) * CFL
        self.dt[boundary2:] = (self.dx / self.c_fat) * CFL

        #设定介质边界
        self.bd1 = boundary1
        self.bd2 = boundary2
        self.rho = np.empty((self.nx,1))

        #加入轻微的波动，模拟实际介质的不均匀性
        self.rho[:boundary1] = np.random.uniform(low=self.rho_fat-200, high=self.rho_fat+200, size=(boundary1,1))
        self.rho[boundary1:boundary2] = np.random.uniform(low=self.rho_muscle-300, high=self.rho_muscle+300
                                                          , size=(boundary2-boundary1,1))
        self.rho[boundary2:] = np.random.uniform(low=self.rho_fat-200, high=self.rho_fat+200, size=(nx-boundary2,1))

        self.rho_temp = np.empty((self.nx, 1))
        self.rho_temp[:self.bd1] = self.rho[:self.bd1] / 1e3
        self.rho_temp[self.bd1:self.bd2] = self.rho[self.bd1:self.bd2] / 1e3 + 5
        self.rho_temp[self.bd2:] = self.rho[self.bd2:] / 1e3

        self.data_bank = np.empty((2,nt,nx,1))      #存数据用的，用来画图

        #考虑衰减因素
        self.decay_factor = np.ones((nx,1))

        for i in range(boundary1):
            self.decay_factor[i] = math.exp(-0.0006/2*i)
        for i in range(boundary1,boundary2):
            self.decay_factor[i] = math.exp(-0.0013/2*(i-boundary1))
        for i in range(boundary2,nx):
            self.decay_factor[i] = math.exp(-0.0006/2*(i-boundary2))

        self.p = np.zeros((nx,1))
        self.u = np.zeros((nx,1))


    def run(self):
        for i in np.arange(0,self.nt):
            #发信号
            if i<43:
                self.p[0]=10 * np.sin(np.pi*0.05*i)

            #正常传播
            og = self.p[self.bd2-1]
            self.u[0:self.nx-1] = self.u[0:self.nx-1] - (self.dt[1:self.nx]/(self.dx*self.rho[1:self.nx]))*(self.p[1:self.nx]-self.p[0:self.nx-1])
            self.p[1:self.nx] = self.p[1:self.nx] - (self.dt[0:self.nx-1] / (self.dx * self.k[0:self.nx-1])) * (self.u[1:self.nx] - self.u[0:self.nx - 1])
            self.p[1:self.nx] *= self.decay_factor[1:self.nx]

            self.p[0] =self.p[1]
            self.u[self.nx - 1] = self.u[self.nx - 2]
            self.u[0] = self.u[1]

            data_lists = [self.p,self.u]
            for j in range(len(data_lists)):
                self.data_bank[j,i] = data_lists[j]

    def draw(self):
        #画图展示,不会生成动图,生成的图片也不能保存
        plt.figure()
        plt.ion()
        p_max, p_min = np.max(self.data_bank[0]), np.min(self.data_bank[0])
        u_max, u_min = np.max(self.data_bank[1]), np.min(self.data_bank[1])

        for i in range(self.nt):
            plt.clf()
            plt.subplot(2,1,1)
            plt.title('p')
            plt.ylim([p_min, p_max])
            plt.plot(self.x,self.data_bank[0,i])
            plt.plot(self.x,self.rho_temp)

            plt.subplot(2,1,2)
            plt.title('u')
            plt.ylim([u_min, u_max])
            plt.plot(self.x,self.data_bank[1,i])
            plt.plot(self.x,self.rho/1e3)
            plt.pause(self.dt[0]/3e20)

        plt.ioff()
        plt.show()

    def animation(self,save=False):
        #画动图，要保存就选True
        fig, (ax1,ax2) = plt.subplots(2,1)
        p_max, p_min = np.max(self.data_bank[0]), np.min(self.data_bank[0])
        u_max, u_min = np.max(self.data_bank[1]), np.min(self.data_bank[1])
        ax1.set_ylim(p_min, p_max)
        ax2.set_ylim(u_min, u_max)

        ax1.plot(self.x, self.rho_temp)
        ax1.set_title('p')
        line1, = ax1.plot(self.x,self.data_bank[0][0])
        ax2.plot(self.x,self.rho_temp/1e3)
        ax2.set_title('u')
        line2, = ax2.plot(self.x,self.data_bank[1][0])

        def update(frame):
            line1.set_ydata(self.data_bank[0,frame])  # 更新曲线数据
            line2.set_ydata(self.data_bank[1,frame])
            return line1,line2

        ani = FuncAnimation(fig, update, frames=self.nt, blit=True,interval=25)
        if save==True:
            ani.save('{}.gif'.format('Simulation5'))
        plt.show()

if __name__ == '__main__':
    u = OneDimensionalFluctuation()
    u.run()
    u.animation()

