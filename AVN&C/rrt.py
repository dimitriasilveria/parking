import networkx as nx
import numpy as np

import random
import math
import matplotlib.pyplot as plt

import matplotlib.animation as animation
from matplotlib import style

class Path_Generator():
    def __init__(self, car_length, obst_tol,target_tol, radius) -> None:
        self.G = nx.DiGraph()
        self.q_target = np.zeros(2)
        self.q_init = np.zeros(2)
        self.obst_tol = obst_tol
        self.target_tol = target_tol

        self.first = True
        self.L = car_length
        self.radius = radius
        self.v_max = 0.1
        self.phi_max = 0.6
    def cost(self,x_last,t,dt):
        c_min = 100000
        for v in np.arange(-self.v_max,self.v_max,self.v_max):
            for phi in np.arange(-self.phi_max,self.phi_max+0.01,0.01):
                X_new, Y_new, Psi_new,Phi_new = self.kinematics(dt,t,x_last[0:2],x_last[2],x_last[3],v,phi)
                if any(X_new):
                    c = 20*self.dist([X_new[-1],Y_new[-1]],self.q_init) + self.dist([X_new[-1],Y_new[-1]],self.q_target) #- 0.15*(1-m**2)
                    if c < c_min and X_new[-1] != x_last[0] and Y_new[-1] !=x_last[1] and Psi_new[-1] !=x_last[2] and X_new[-1] != self.q_init[0]:
                        c_min = c
                        x_better = X_new
                        y_better = Y_new
                        psi_b = Psi_new
                        phi_b = Phi_new
                        print(x_better)


        return x_better,y_better, psi_b, phi_b

    def random_points(self, center):
        # radius of the circle

        # center of the circle (x, y)

        # if self.first:
        #     x = self.q_target[0]-self.q_init[0] #front parking
        #     y = self.q_target[1]-self.q_init[1]
        #     #x = -self.q_target[0]+self.q_init[0]
        #     #y = -self.q_target[1]+self.q_init[1]

            
        #     if(x<0 and y<0):
        #         alpha = math.pi + random.random() * math.pi
        #         #print(" belongs to 3rd Quadrant.")
        #     elif(x<0 and y>0):
        #         alpha = random.random()* math.pi
        #         #print(" belongs to 2nd Quadrant.")
        #     elif(x>0 and y>0):
        #         alpha = random.random() * math.pi
        #         #print(" belongs to 1st Quadrant.",alpha)
        #     else:
        #         alpha = math.pi + random.random() * math.pi
        #         #print(" belongs to 4th Quadrant.")
        #     #alpha = np.pi/2
        # else:
        #     # random angle
        #     alpha = 2*math.pi * random.random()
        # # random radius
        # r = self.radius * random.random() #0.3*self.radius + 0.7*self.radius * math.sqrt(random.random())
        # # calculating coordinates
        # x = r * math.cos(alpha) + center[0]
        # y = r * math.sin(alpha) + center[1]
        phi = -self.phi_max + random.random()*2 * self.phi_max
        v = -self.v_lim + 2*self.v_lim*random.random()
        #print("Random point", (x, y))
        return v,phi
    
    def outside_box(self,p ):
        # radius of the circle

        # center of the circle (x, y)
        x0 = self.q_init[0] - 4
        x1 = self.q_init[0] + 2
        y0 = self.q_init[1] -2
        y1 = self.q_init[1] +4

        if (p[0] < x0 or p[0] > x1 or p[1] < y0 or p[1] > y1) :
            return True
        else :
            return False


    def solve(self, bl, tr, p) :
        if (p[0] > bl[0] and p[0] < tr[0] and p[1] > bl[1] and p[1] < tr[1]) :
            return True
        else :
            return False

    def obstacle(self,x):
        self.d_obstacle = 0.5
        obstacle = self.q_target[:2]
        #obstacle_1 = self.box([obstacle[0]-self.d_obstacle,obstacle[1]+self.d_obstacle],[obstacle[0]+self.d_obstacle,obstacle[1]+2*self.d_obstacle])
        obs_1 = self.solve([obstacle[0]-self.d_obstacle-self.obst_tol,obstacle[1]+self.d_obstacle-self.obst_tol],
                           [obstacle[0]+self.d_obstacle+self.obst_tol,obstacle[1]+self.d_obstacle+self.obst_tol],x)
        #obstacle_2  = self.box([obstacle[0]-self.d_obstacle,obstacle[1]-self.d_obstacle],
        #                       [obstacle[0]+self.d_obstacle,obstacle[1]-2*self.d_obstacle])
        obs_2 = self.solve( [self.q_target[0]-self.d_obstacle-self.obst_tol,self.q_target[1]-self.d_obstacle-self.obst_tol],
                           [self.q_target[0]+self.d_obstacle+self.obst_tol,self.q_target[1]-self.d_obstacle+self.obst_tol],
                           x)
            # [obstacle[0]-self.d_obstacle-self.obst_tol,
            #                 obstacle[1]-self.d_obstacle+self.obst_tol],
            #                 [obstacle[0]+self.d_obstacle+self.obst_tol,
            #                 obstacle[1]-2*self.d_obstacle-self.obst_tol]
            #                 ,x)

        return False#obs_1 or obs_2
    def dist(self,x1,x2):
        #print(x1,x2)
        d = np.sqrt((x1[0]-x2[0])**2+(x1[1]-x2[1])**2)
        return d

    def search_nearest(self, x_new):
        d_min = 10000
        n_min = 0
        for node in self.G.nodes:
            #print(self.G.nodes[node])
            #print(self.G.nodes[node]['qx'],'qx')
            d = self.dist([self.G.nodes[node]['qx'][-1],self.G.nodes[node]['qy'][-1]],x_new)
            if d < d_min:
                d_min = d
                #print(self.G.nodes[node]['q'])
                n_min = node
        return n_min

    def box(self,v_min,v_max):
        ver_x = np.array([v_min[0],v_max[0],v_max[0],v_min[0],v_min[0]])
        ver_y = np.array([v_min[1],v_min[1],v_max[1],v_max[1],v_min[1]])

        return ver_x, ver_y
    def kinematics(self,dt,T,x0,psi,phi,v,phi_d):

        x = x0[0]
        y = x0[1]
        psi = psi

        phi = phi_d-phi
        X = []
        Y = []
        Psi = []
        Phi = []
        #h = 0.01
        
        if phi < -self.phi_max:
            phi = -self.phi_max
        elif phi > self.phi_max:
            phi = self.phi_max 
        t=0
        while t<T : #and self.obstacle([x,y]) == False:

        
            x= x + v*dt*np.cos(psi)#*np.cos(phi)
            y = y +v*dt*np.sin(psi)#*np.cos(phi)
            psi = psi + v*dt*np.tan(phi)/self.L
            if self.obstacle([x,y]) or self.outside_box([x,y]) == True:
                print('obstacle')
                return [],[],[],[]
            X.append(x)
            Y.append(y)
            Psi.append(psi)
            Phi.append(phi)
            #i = i +1
            t=t+dt
        return X, Y, Psi,Phi
    
    def path_smoothing(self,G_path):
        X = []
        Y = []
        ind = 0

        Psi = []#self.q_init[2]
        Phi = []#self.q_init[3]
        ind = list(G_path.nodes)[-1]
        #print(G_path.nodes,'nodes')
        for _ in range(len(list(G_path.nodes))-1):
            #print(self.G.nodes[node])
            
            #next = list(G_path.predecessors(ind))[0]
            #print(next,'next')
            x = G_path.nodes[ind]['qx']
            y = G_path.nodes[ind]['qy']
            psi = G_path.nodes[ind]['qpsi']
            phi = G_path.nodes[ind]['qphi']
            #xf = G_path.nodes[next]['q']
            #print(x0,xf,'x')
            ind = list(G_path.predecessors(ind))[0]
            #print(ind,'ind')
           
            X = X + x
            Y = Y + y
            Psi = Psi + psi
            Phi = Phi + phi
            #print('lists in path smoothing',X,Y)
        return X, Y, Psi, Phi

    def generate_path(self,q_init,q_target,N,v,w):
        self.q_target = q_target
        self.q_init = q_init
        self.v = v
        self.w = w
        t = 10
        dt = 0.05

        ind = 0

        for j in range(1):

            X = []
            Y = []
            Phi = []
            Psi = []
            #print(j)
            #print(list(self.G.nodes[0]['q']))
            x_last = self.q_init
            Psi = [self.q_init[2]]
            Phi = [self.q_init[3]]
            radius = 4
            self.first = True
            for i in range(1,2,1):
                print(i)
                #v,phi = self.random_points(x_last)
                X_new, Y_new, Psi_new,Phi_new = self.cost(x_last,t,dt)

                #x_new = self.cost(x_rand,x_last)
                #print(x_new,'x_new')

                if any(X_new) == False:
                    break
                x_last = [X_new[-1],Y_new[-1],Psi[-1],Phi[-1]]
                
                self.G.add_nodes_from([
                (i, {'qx':X_new,
                        'qy': Y_new,
                        'qpsi': Psi,
                        'qphi': Phi})
                ])
                X = X + X_new
                Y = Y + Y_new
                Psi = Psi +Psi_new
                Phi = Phi + Phi_new

            if abs(x_last[0] -self.q_target[0])  <= self.target_tol  and abs(x_last[1] -self.q_target[1])  <= self.target_tol and abs(x_last[2] -self.q_target[2])  <= self.target_tol:
                #print(x_new,'x_new')
                print(j)
                break

                

        return X,Y, Psi, Phi


    def trim_path(self):
        G_path = nx.DiGraph()
        dim = len(self.G.nodes)

        ind = list(self.G.nodes)[-1]
        G_path.add_nodes_from([
        (0, self.G.nodes[ind])
        ])

        i = 1
        while 1:

            #print(self.G.nodes[ind]['q'],'edges')
            G_path.add_nodes_from([
            (i, self.G.nodes[ind])
            ])
            G_path.add_edge(i-1,i)
            if ind == 0:
                break
            ind = list(self.G.predecessors(ind))[0]

            #print(ind,'predecessor')
            
            
            i+=1
        return G_path


    def plot_path(self,X,Y):#,X,Y):
        #pos = {node: G_path.nodes[node]['q'] for node in G_path.nodes}
        #print(pos)
        # options = {
        #     "font_size": 20,
        #     "node_size": 500,
        #     "node_color": "white",
        #     "edgecolors": "black",
        #     "linewidths": 5,
        #     "width": 5,
        # }
        #nx.draw_networkx(G_path, pos, **options)

        # Set margins for the axes so that nodes aren't clipped
        #ax = plt.gca()
        #ax.margins(0.20)

        #print(([self.q_target[0]-self.d_obstacle,self.q_target[1]+self.d_obstacle],[self.q_target[0]+self.d_obstacle,self.q_target[1]+2*self.d_obstacle]))
        x1,y1 = self.box([self.q_target[0]-self.d_obstacle,self.q_target[1]+self.d_obstacle],[self.q_target[0]+self.d_obstacle,self.q_target[1]+self.d_obstacle])
        x2,y2  = self.box([self.q_target[0]-self.d_obstacle,self.q_target[1]-self.d_obstacle],[self.q_target[0]+self.d_obstacle,self.q_target[1]-self.d_obstacle])
        plt.plot(x1,y1)
        plt.plot(x2,y2)
        plt.plot(X,Y)
        plt.plot(0,0, markersize=14)

        #plt.axis("off")
        plt.show()

if __name__ == '__main__':
    L = 0.26
    W = 0.17
    path_gen = Path_Generator(car_length=L,obst_tol=0,target_tol=0.5,radius=1)
    q_init = np.array([1,-2,np.pi/2,0])
    q_target = np.array([0,0,3*np.pi/4,0])
    #angles_i = np.array([]) #psi, phi
    #angles_f = np.array([])
    X1,Y1, Psi1, Phi1 = path_gen.generate_path(q_init,q_target,20,v=0.01,w=0)

    print(X1[-1],Y1[-1])
    if any(X1):
        path_gen.plot_path(X1,Y1)