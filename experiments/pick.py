import pprint,pickle
import matplotlib
#matplotlib.rcParams['backend'] = 'SVG'###！！！矢量图格式svg
import matplotlib.pyplot as plt
import numpy as np

from openpyxl import Workbook

plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

ReCat_dir='D:/MyWorkPlacePy/mtsr-maddpg-master/experiments/learning_curves_tt/'
####D:/ProjectData/maddpg-master/experiments/learning_curves_716ZxL/
p=open(ReCat_dir+'expname_rewards.pkl','rb')
d=pickle.load(p)
# pprint.pprint(d)
#plt.plot(d,label="total reward")###总回报

p1=open(ReCat_dir+'expname_agrewards.pkl','rb')
d1=pickle.load(p1)
#pprint.pprint(d1)


Ntr=len(d) #训练次数
Nag=int(len(d1)/len(d)) # number of agents
Q=np.zeros((Nag, Ntr))
for j in range(Ntr):
    for i in range(Nag):
        Q[i, j] = d1[j*Nag+i]
#print(Q)
# plt.plot(Q[0],label="AdversaryAgent Reward")
# plt.plot(Q[-5],label="BlueAgent1 Reward")
# plt.plot(Q[-4],label="BlueAgent2 Reward")
# plt.plot(Q[-3],label="GreenAgent1 Reward")###BlueAgent1 Reward###
# plt.plot(Q[-2],label="GreenAgent2 Reward")###BlueAgent2 Reward ###
# plt.plot(Q[Nag-1],label="GreenAgent3 Reward")
plt.plot(Q[0],label="attacker I")##  #守方回报值
plt.plot(Q[1],label="attacker II")##  #攻击者1回报值
plt.plot(Q[2],label="attacker III")##  #攻击者1回报值
plt.plot(Q[3],label="attacker IV")##  #攻击者1回报值
plt.plot(Q[4],label="attacker V")##  #攻击者2回报值
plt.plot(Q[5],label="attacker VI")##  #诱骗者1回报值

#plt.plot(Q[Nag-1],label="诱骗者3回报值")

plt.legend(loc='lower right',ncol=1)####center##lower right
plt.xlabel('length of training / 1000 episodes')##  #训练次数/千回合
plt.ylabel('reward value')##  #回报值
# plt.savefig('REWARD.svg',format='svg')###！！！保存矢量图在…/experiments文件夹下
plt.show()
# p2=open('D:/SicSoftware/PycharmProjects/maddpg-master/experiments/benchmark_files_tt/expname.pkl','rb')
# d2=pickle.load(p2)
# pprint.pprint(d2)

TQ=0
for i in range(Nag):
   TQ = TQ+Q[i]
TQmax=max(TQ)
print(TQ)
print('%.2f' % TQmax)

'''
p3=open(ReCat_dir+'Number_AgentsCatched.pkl','rb')
d3=pickle.load(p3)
#pprint.pprint(d3)
Nsp=len(d3) ## number of save point ## print(len(d3))
sp=[]
for i in range(Nsp):
    sp.append(sum(d3[i]))
print(sp)
print(np.max(np.max(d3)))
#plt.plot(d3[0])
# plt.plot(d3[1])###第1000-1999回合
# plt.plot(d3[-1])###最后1000回合
# plt.show()
plt.plot(sp)###抓捕次数
plt.xlabel('length of training / 1000 episodes')##  #训练次数/千回合
plt.ylabel('Number of times attacking agents are intercepted')##  #攻方智能体被截击次数
plt.savefig('Tt.svg',format='svg')###！！！保存矢量图在…/experiments文件夹下
plt.show()


p3b=open(ReCat_dir+'Number_BuleAgentsCatched.pkl','rb')
d3b=pickle.load(p3b)
#pprint.pprint(d3)
Nspb=len(d3b) ## number of save point ## print(len(d3))
spb=[]
for i in range(Nspb):
    spb.append(sum(d3b[i]))
print(spb)
print(np.max(np.max(d3b)))
#plt.plot(d3[0])
# plt.plot(d3[1])###第1000-1999回合
# plt.plot(d3[-1])###最后1000回合
# plt.show()
plt.plot(spb)###抓捕蓝球次数
plt.xlabel('length of training / 1000 episodes')##  #训练次数/千回合
plt.ylabel('Number of times attackers are intercepted')##  #攻击者被截击次数
plt.savefig('Ta.svg',format='svg')###！！！保存矢量图在…/experiments文件夹下
plt.show()
'''

p4=open(ReCat_dir+'Number_FoodsReached.pkl','rb')
d4=pickle.load(p4)
# pprint.pprint(d4)
Nusp=len(d4) ## number of save point ## print(len(d3))
num_targets=len(d4[0][0])
sap=[]
for i in range(Nusp):
    Num_ij = [0] * num_targets
    for j in range(num_targets):
        Num_ij[j] = sum([x[j] for x in d4[i]])
        # print('Num to target i = ', Num_ij)
    sap.append(Num_ij)
print('sap= ', sap)

#plt.plot(d4[0])
# plt.plot(d4[1])###第1000-1999回合
# plt.plot(d4[-1])###最后1000回合
# plt.show()
plt.figure()
plt.plot(sap)###到达目标次数
plt.xlabel('length of training / 1000 episodes')##  #训练次数/千回合
plt.ylabel('Number of times attackers reach targets')##  #攻击者到达目标次数
plt.savefig('Tr.svg',format='svg')###！！！保存矢量图在…/experiments文件夹下
plt.show()

wb = Workbook()#######创建并保存Excel文件
ws = wb.active
ws.append(range(Ntr)) ##训练回合/1000
for i in range(Nag):
    lst = list(Q[i])
    print('Q[%d] = ' %i, lst)
    ws.append(lst)
ws.append(d)  ##总reward
# for j in range(num_targets)

for j in range(num_targets):
    lst = [x[j] for x in sap]
    ws.append(lst)  ##NumToTargets
Num_AT = [sum(t) for t in sap]
# print('Number of agents to targets: ', Num_AT)
ws.append(Num_AT) ###智能体到达所有目标总次数
wb.save(ReCat_dir+'QiAndNumToTargets.xlsx')