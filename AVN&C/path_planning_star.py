import networkx as nx
import numpy as np

import random
import math
import matplotlib.pyplot as plt

class Path_Generator():
    def __init__(self, car_length, obst_tol,target_tol, radius) -> None:
        self.G = nx.DiGraph()
        self.q_target = np.zeros(2)
        self.q_init = np.zeros(2)
        self.obst_tol = obst_tol
        self.target_tol = target_tol
        self.y_prev = 0
        self.x_prev = 0
        self.first = True
        self.L = car_length
        self.radius = radius
        self.v = 5
        self.W = 1
        self.obstacles = []
    def best(self,x_new,x_nearest):
        d = 2
        gamma = 2
        c_min = 100000
        n_best = -1
        n = len(self.G.nodes)
        r = gamma*((np.log(n+1))/(n+1))**(1/d)
        for node in self.G.nodes:
            x_node = [self.G.nodes[node]['qx'][-1],self.G.nodes[node]['qy'][-1]]
            d_node = self.dist(x_node,x_new)
            m = (x_node[1]-x_new[1])/(x_node[0]-x_new[0])
            #print('d',d,node)
            #print('x new',x_new,self.G.nodes[node]['qx'][-1],self.G.nodes[node]['qy'][-1])
            if d <= r and d >0.01:
                cost = self.G.nodes[node]['cost'] + d_node + abs(m)
                if cost < c_min:
                    c_min = cost
                #print(self.G.nodes[node]['q'])
                    n_best = node

        return n_best

    def random_points_box(self, center):
        # radius of the circle

        # center of the circle (x, y)
        obs_1 = self.obstacles[0]
        x0 = obs_1[0][0] 
        x1 = self.q_init[0] 
        y0 = self.q_init[1]
        y1 = obs_1[1][1]
        #print(x0,x1,y0,y1)
        x = x0 + random.random()*(x1-x0) -0.1
        y = y0 + random.random() *(y1-y0) + 0.1
        return [x,y]

    def solve(self, bl, tr, p) :
        if (p[0] > bl[0] and p[0] < tr[0] and p[1] > bl[1] and p[1] < tr[1]) :
            return True
        else :
            return False

    def set_obstacles(self,corners):
        bl = corners[0]
        tr = corners[1]
        bl[0] = bl[0]-self.obst_tol
        bl[1] = bl[1]-self.obst_tol
        tr[0] = tr[0] + self.obst_tol
        tr[1] = tr[1] + self.obst_tol
        self.obstacles.append([bl,tr])


    def obstacle(self,x):
        is_obstacle = []
        for obstacle in self.obstacles:
            is_obs = self.solve(obstacle[0],obstacle[1],x)
            is_obstacle.append(is_obs)
        return any(is_obstacle)
    
    def dist(self,x1,x2):
        #print(x1,x2)
        d = np.sqrt((x1[0]-x2[0])**2+(x1[1]-x2[1])**2)
        return d
    
    def outside_box(self,p ):
        # radius of the circle

        # center of the circle (x, y)
        obs_1 = self.obstacles[0]
        x0 = obs_1[0][0] - 0.1
        x1 = self.q_init[0] 
        y0 = self.q_init[1]
        y1 = obs_1[1][1] + 0.1


        if (p[0] < x0 or p[0] > x1 or p[1] < y0 or p[1] > y1) :
            return True
        else :
            return False


    def search_nearest(self, x_new):
        d_min = 10000
        n_min = -1
        print('nodes',len(self.G.nodes))
        for node in self.G.nodes:

            d = self.dist([self.G.nodes[node]['qx'][-1],self.G.nodes[node]['qy'][-1]],x_new)

            if d <= d_min:
                d_min = d
                n_min = node

        if d_min < 0.01:
            return -1
        else:
            return n_min

    def box(self,v_min,v_max):
        ver_x = np.array([v_min[0],v_min[0],v_max[0],v_max[0],v_min[0]])
        ver_y = np.array([v_min[1],v_max[1],v_max[1],v_min[1],v_min[1]])

        return ver_x, ver_y
    def kinematics(self,dt,x0,xf,psi,phi,v):
        r = 0.5
        tol = 0.02 
            
        x = x0[0]
        y = x0[1]
        psi = psi
        phi = phi
        X = []
        Y = []
        Psi = []
        Phi = []

        #h = 0.01
        t = 0
        cost = 0
        t_max = 0.75*random.randint(1,3)
        while (abs(x - xf[0]) > tol or abs(y-xf[1])> tol ) and t<0.5: #and self.obstacle([x,y]) == False:
            #print(i)
            m = np.arctan2(xf[1]-y,xf[0]-x)
            
            phi = m-psi #front parking
            if phi < -0.6:
                phi = -0.6
            elif phi > 0.6:
               phi = 0.6
            cost += np.sqrt(v*dt*np.cos(psi)**2+v*dt*np.sin(psi)**2)
            x= x + v*dt*np.cos(psi)#*np.cos(phi)
            y = y +v*dt*np.sin(psi)#*np.cos(phi)
            psi = psi + v*dt*np.tan(phi)/self.L
            if self.obstacle([x,y]) or self.outside_box([x,y]) == True:
                return [],[],[],[],[]
            #print('x,y',x,y)
            X.append(x)
            Y.append(y)
            Psi.append(psi)
            Phi.append(phi)
            #i = i +1
            t = t+dt
        return X, Y, Psi,Phi,cost
    
    def path_smoothing(self,G_path):
        X = []
        Y = []
        ind = 0

        Psi = []#self.q_init[2]
        Phi = []#self.q_init[3]
        ind = list(G_path.nodes)[-1]
        #print(G_path.nodes,'nodes')
        for _ in range(len(list(G_path.nodes))-1):

            x = G_path.nodes[ind]['qx']
            y = G_path.nodes[ind]['qy']
            psi = G_path.nodes[ind]['qpsi']
            phi = G_path.nodes[ind]['qphi']
            ind = list(G_path.predecessors(ind))[0]
            
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
        self.W = w
        dt = 1/50
        X = []
        Y = []
        while 1:

            self.G.add_nodes_from([
                (0, {'qx':[self.q_init[0]],
                     'qy': [self.q_init[1]],
                    'qpsi':[self.q_init[2]],
                     'qphi': [self.q_init[3]],
                     'cost' : 0 })
            ])

            #print(list(self.G.nodes[0]['q']))
            x_last = self.q_init[0:2]
            Psi = [self.q_init[2]]
            Phi = [self.q_init[3]]
            radius = 4
            self.first = True

            plt.clf()
            # obs_1 = self.obstacles[0]
            # x0 = obs_1[0][0]# - 0.25
            # x1 = self.q_init[0] 
            # y0 = self.q_init[1]
            # y1 = obs_1[1][1] + 0.1
            # print('0',x0,y0,'1',x1,y1)
            # print(self.q_target)
            # x,y = self.box([x0,y0],[x1,y1])
            # plt.plot(x,y)
            plt.scatter(q_init[0],q_init[1],100)
            plt.scatter(q_target[0],q_target[1],100)
            for obstacle in self.obstacles:
                x,y=self.box(obstacle[0],obstacle[1])
                plt.plot(x,y)
            i=1
            while i <40:
                
                print(i)
                x_new = self.random_points_box(x_last)               
                                #print(x_new,'x_new')
                #print(self.obstacle(self.q_target,x_new))
                if self.obstacle(x_new) != False and self.outside_box(x_new) != False:
                    continue
                nearest = self.search_nearest(x_new)
                if nearest == -1:
                    continue
                x_nearest = [self.G.nodes[nearest]['qx'][-1],
                                self.G.nodes[nearest]['qy'][-1]]#,self.G.nodes[nearest]['cost']]
                best = self.best(x_new,x_nearest)
                if best == -1:
                    best = nearest
                x_best = [self.G.nodes[best]['qx'][-1],
                                self.G.nodes[best]['qy'][-1],self.G.nodes[best]['qpsi'][-1],self.G.nodes[best]['qphi'][-1]]
                X_new, Y_new, Psi,Phi,cost = self.kinematics(dt,x_best[0:2],x_new,x_best[2],x_best[3],self.v)
                #print(Phi[-1],'phi')
                if any(X_new) != True:
                    continue
                x_last = [X_new[-1],Y_new[-1],Psi[-1]]
                
                self.G.add_nodes_from([
                (i, {'qx':X_new,
                        'qy': Y_new,
                        'qpsi': Psi,
                        'qphi': Phi,
                        'cost': cost})
                ])
                
                self.G.add_edge(best,i)
                pos = {node: [self.G.nodes[node]['qx'][-1],self.G.nodes[node]['qy'][-1]] for node in self.G.nodes}
                i+=1
                # options = {
                #     "font_size": 20,
                #     "node_size": 500,
                #     "node_color": "white",
                #     "edgecolors": "black",
                #     "linewidths": 5,
                #     "width": 5,
                # }

                #plt.plot(x_box,y_box)
                nx.draw_networkx(self.G, pos)
                #nx.draw(self.G)
                plt.draw()
                plt.pause(0.01) 
                self.first = False

                if abs(x_last[0] -self.q_target[0])  <= self.target_tol  and abs(x_last[1] -self.q_target[1])  <= self.target_tol:# and abs(x_last[2] -self.q_target[2])  <= 2*self.target_tol:
                    print(i)
                    break
                    #return self.G
            if abs(x_last[0] -self.q_target[0])  <= self.target_tol  and abs(x_last[1] -self.q_target[1])  <= self.target_tol:# and abs(x_last[2] -self.q_target[2])  <= 2*self.target_tol:
                #print(x_new,'x_new')

                break
            else:
                self.G.clear()
                
        if len(self.G.nodes):
            
            G_new = self.trim_path()
            X, Y, Psi, Phi = self.path_smoothing(G_new)
            return X,Y, Psi, Phi
        else:
            print('could not find path')    
            return [],[],[],[]

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
        obstacle = self.q_target

        #print(([self.q_target[0]-self.d_obstacle,self.q_target[1]+self.d_obstacle],[self.q_target[0]+self.d_obstacle,self.q_target[1]+2*self.d_obstacle]))
        for obstacle in self.obstacles:
            x,y=self.box(obstacle[0],obstacle[1])
            print(x,y)
            #plt.plot(x,y)
        plt.plot(X,Y,color='red')
        plt.plot(0,0, markersize=14)

        #plt.axis("off")
        plt.show()

if __name__ == '__main__':
    L = 0.26
    W = 0.17
    path_gen = Path_Generator(car_length=L,obst_tol=0,target_tol=0.06,radius=1)
    obs_1 = [[-0.04,0.3],[0.04,0.45]]
    obs_2 = [[-0.04,-0.4],[0.04,-0.25]]
    obs_3 = [[obs_1[1][0]+W/2,obs_1[0][1]-L],[obs_1[1][0]+3/2*W,obs_1[0][1]]]
    path_gen.set_obstacles(obs_1)
    path_gen.set_obstacles(obs_2)
    #path_gen.set_obstacles(obs_3)
    q_init = np.array([0.2,obs_2[0][1]-0.35,np.pi/2,0])
    q_target = np.array([(obs_1[0][0]+obs_1[1][0])/2,(obs_1[0][1]+obs_2[1][1])/2,np.pi/2,0])
    #angles_i = np.array([]) #psi, phi
    #angles_f = np.array([])
    X1,Y1, Psi1, Phi1 = path_gen.generate_path(q_init,q_target,50,v=0.11,w=W)
   
    # options = {
    #     "font_size": 20,
    #     "node_size": 500,
    #     "node_color": "white",
    #     "edgecolors": "black",
    #     "linewidths": 5,
    #     "width": 5,
    # }
    #x_box,y_box = self.box([x0,y0],[x1,y1])
    #plt.plot(x_box,y_box)
    #nx.draw_networkx(G_new, pos)
    #nx.draw(self.G)
    #plt.draw()
    #plt.pause(0.01) 

    print(X1[-1],Y1[-1])
    if any(X1):
        path_gen.plot_path(X1,Y1)