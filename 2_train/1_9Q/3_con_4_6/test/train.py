import os,sys
import torch
import numpy as np
import random
import datetime
import pennylane as qml
from torch import optim
from Global import *
from Read_Data.read_data import *

dev = qml.device("default.qubit", wires=N_Qubit)

def Para():
  par1=torch.rand((N_Lay1,N_par1),device=dev2,dtype=Dtype,requires_grad=True)
  par2=torch.rand((N_par2),device=dev2,dtype=Dtype,requires_grad=True)
  return par1,par2

def Train_Pyt(bfe,img):
  sam=bfe.shape[0]
  n_batch=int(np.ceil(sam/Batch))
  num=np.arange(sam)
  random.shuffle(num)
  batch=Batch
  for i in range (n_batch): # n_batch
    beg_frm=i*Batch
    end_frm=(i+1)*Batch
    if end_frm>sam:
      end_frm=sam
      batch=end_frm-beg_frm

    # 01 2 34 5678
    # 01 34 2 56 78
    # 01 56 234 78
    # 0 1 78 234 56
    # 02 34 1 56 78
    # 02 56 134 78
    # 0 2 78 1 3456
    # 0123 45678

    bfe2=bfe[num[beg_frm:end_frm]]
    img1=img[num[beg_frm:end_frm]].reshape(batch,4,2,4,16).permute(1,3,2,4,0).reshape(16,32*batch) # 0134 25678
    pred=torch.matmul(Con_Unitary(Par1[0],S_Ker1),img1).reshape(4,4,2,4,4,batch).permute(0,3,2,1,4,5).reshape(16,32*batch) # 0156 23478
    pred=torch.matmul(Con_Unitary(Par1[1],S_Ker1),pred).reshape(4,4,8,4,batch).permute(0,3,2,1,4).reshape(16,32*batch) # 0178 23456
    pred=torch.matmul(Con_Unitary(Par1[2],S_Ker1),pred).reshape(2,2,4,8,4,batch).permute(0,3,1,4,2,5).reshape(16,32*batch) # 0234 15678
    pred=torch.matmul(Con_Unitary(Par1[3],S_Ker1),pred).reshape(4,4,2,4,4,batch).permute(0,3,2,1,4,5).reshape(16,32*batch) # 0256 13478
    pred=torch.matmul(Con_Unitary(Par1[4],S_Ker1),pred).reshape(4,4,8,4,batch).permute(0,3,2,1,4).reshape(16,32*batch) # 0278 13456
    pred=torch.matmul(Con_Unitary(Par1[5],S_Ker1),pred).reshape(2,2,4,2,16,batch).permute(5,0,3,1,4,2).reshape(batch,8,64) # 0123 45678
    pred=pred.pow(2).sum(dim=2)
    err=torch.matmul(pred,Par2)-bfe2
    loss=torch.square(err).sum()
    Opt.zero_grad()
    loss.backward()
    Opt.step()

def Pred_Pyt(img,par1,par2):
  E=[]
  sam=img.shape[0]
  n_batch=int(np.ceil(sam/Batch))
  num=np.arange(sam)
  batch=Batch
  cm0=Con_Unitary(par1[0],S_Ker1)
  cm1=Con_Unitary(par1[1],S_Ker1)
  cm2=Con_Unitary(par1[2],S_Ker1)
  cm3=Con_Unitary(par1[3],S_Ker1)
  cm4=Con_Unitary(par1[4],S_Ker1)
  cm5=Con_Unitary(par1[5],S_Ker1)
  for i in range (n_batch):
    beg_frm=i*Batch
    end_frm=(i+1)*Batch
    if end_frm>sam:
      end_frm=sam
      batch=end_frm-beg_frm
    img1=img[num[beg_frm:end_frm]].reshape(batch,4,2,4,16).permute(1,3,2,4,0).reshape(16,32*batch) # 0134 25678
    pred=torch.matmul(cm0,img1).reshape(4,4,2,4,4,batch).permute(0,3,2,1,4,5).reshape(16,32*batch) # 0156 23478
    pred=torch.matmul(cm1,pred).reshape(4,4,8,4,batch).permute(0,3,2,1,4).reshape(16,32*batch) # 0178 23456
    pred=torch.matmul(cm2,pred).reshape(2,2,4,8,4,batch).permute(0,3,1,4,2,5).reshape(16,32*batch) # 0234 15678
    pred=torch.matmul(cm3,pred).reshape(4,4,2,4,4,batch).permute(0,3,2,1,4,5).reshape(16,32*batch) # 0256 13478
    pred=torch.matmul(cm4,pred).reshape(4,4,8,4,batch).permute(0,3,2,1,4).reshape(16,32*batch) # 0278 13456
    pred=torch.matmul(cm5,pred).reshape(2,2,4,2,16,batch).permute(5,0,3,1,4,2).reshape(batch,8,64) # 0123 45678
    pred=pred.pow(2).sum(dim=2)
    pred=torch.matmul(pred,par2).to('cpu').tolist()
    E=E+pred
  del cm0,cm1,cm2,cm3,cm4,cm5
  return E

@qml.qnode(dev,interface="torch")
def quantum_circuit(state,cm0,cm1,cm2,cm3,cm4,cm5):
  qml.QubitStateVector(state,wires=list(range(N_Qubit)))
  qml.QubitUnitary(cm0,wires=[0,1,3,4])
  qml.QubitUnitary(cm1,wires=[0,1,5,6])
  qml.QubitUnitary(cm2,wires=[0,1,7,8])
  qml.QubitUnitary(cm3,wires=[0,2,3,4])
  qml.QubitUnitary(cm4,wires=[0,2,5,6])
  qml.QubitUnitary(cm5,wires=[0,2,7,8])
  return qml.probs(wires=[0,1,2])

def Pred_Pen(img,par1,par2):
  E=[]
  cm0=Con_Unitary(par1[0],S_Ker1).to('cpu').numpy()
  cm1=Con_Unitary(par1[1],S_Ker1).to('cpu').numpy()
  cm2=Con_Unitary(par1[2],S_Ker1).to('cpu').numpy()
  cm3=Con_Unitary(par1[3],S_Ker1).to('cpu').numpy()
  cm4=Con_Unitary(par1[4],S_Ker1).to('cpu').numpy()
  cm5=Con_Unitary(par1[5],S_Ker1).to('cpu').numpy()
  for i in range(img.shape[0]):
    pred=quantum_circuit(img[i].to('cpu').numpy(),cm0,cm1,cm2,cm3,cm4,cm5)
    pred=pred.clone()
    pred=torch.matmul(pred,par2).to('cpu')
    E.append(pred)
  del cm0,cm1,cm2,cm3,cm4,cm5
  return E

def Main():
  global Par1,Par2,Opt
  Par1,Par2=Para()
  Pdb_gen,Pdb_ref,Bfe_gen,Bfe_ref=Read_PDB()
  Img_gen=Read_Img(Pdb_gen)
  Img_ref=Read_Img(Pdb_ref)
  Opt=optim.SGD([Par1,Par2],lr=learning_rate,momentum=0.9)
  fw=open("Result/rmsd","w", buffering=1)
  for i in range (N_Ite):
    Train_Pyt(Bfe_gen,Img_gen)
    if i%10==9:
      E_gen_pyt=Pred_Pyt(Img_gen,Par1.detach(),Par2.detach())
      Out_E_diff(fw,i,Bfe_gen,E_gen_pyt)
  E_gen_pyt=Pred_Pyt(Img_gen,Par1.detach(),Par2.detach())
  E_gen_pen=Pred_Pen(Img_gen,Par1.detach(),Par2.detach().to('cpu'))
  E_ref_pyt=Pred_Pyt(Img_ref,Par1.detach(),Par2.detach())
  E_ref_pen=Pred_Pen(Img_ref,Par1.detach(),Par2.detach().to('cpu'))
  Out_Energy('E_train',Bfe_gen,E_gen_pyt,E_gen_pen)
  Out_Energy('E_test',Bfe_ref,E_ref_pyt,E_ref_pen)
  Out_E_diff(fw,1000,Bfe_gen.to('cpu'),E_gen_pen)
  Out_E_diff(fw,1000,Bfe_ref.to('cpu'),E_ref_pen)
  fw.close()

start=datetime.datetime.now()
Main()
end=datetime.datetime.now()
print("執行時間：",end-start)

