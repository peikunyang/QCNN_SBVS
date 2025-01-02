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
  par1=torch.rand((N_par1),device=dev2,dtype=Dtype,requires_grad=True)
  return par1

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
    bfe2=bfe[num[beg_frm:end_frm]]
    pred=img[num[beg_frm:end_frm]].reshape(batch,8,64)
    pred=pred.pow(2).sum(dim=2)
    err=torch.matmul(pred,Par1)-bfe2
    loss=torch.square(err).sum()
    Opt.zero_grad()
    loss.backward()
    Opt.step()

def Pred_Pyt(img,par1):
  E=[]
  sam=img.shape[0]
  n_batch=int(np.ceil(sam/Batch))
  num=np.arange(sam)
  batch=Batch
  for i in range (n_batch):
    beg_frm=i*Batch
    end_frm=(i+1)*Batch
    if end_frm>sam:
      end_frm=sam
      batch=end_frm-beg_frm
    pred=img[num[beg_frm:end_frm]].reshape(batch,8,64)
    pred=pred.pow(2).sum(dim=2)
    pred=torch.matmul(pred,par1).to('cpu').tolist()
    E=E+pred
  return E

@qml.qnode(dev,interface="torch")
def quantum_circuit(state):
  qml.QubitStateVector(state,wires=list(range(N_Qubit)))
  return qml.probs(wires=[0,1,2])

def Pred_Pen(img,par1):
  E=[]
  for i in range(img.shape[0]):
    pred=quantum_circuit(img[i].to('cpu').numpy())
    pred=pred.clone()
    pred=torch.matmul(pred,par1).to('cpu')
    E.append(pred)
  return E

def Main():
  global Par1,Opt
  Par1=Para()
  Pdb_gen,Pdb_ref,Bfe_gen,Bfe_ref=Read_PDB()
  Img_gen=Read_Img(Pdb_gen)
  Img_ref=Read_Img(Pdb_ref)
  Opt=optim.SGD([Par1],lr=learning_rate,momentum=0.9)
  fw=open("Result/rmsd","w", buffering=1)
  for i in range (N_Ite):
    Train_Pyt(Bfe_gen,Img_gen)
    if i%10==9:
      E_gen_pyt=Pred_Pyt(Img_gen,Par1.detach())
      Out_E_diff(fw,i,Bfe_gen,E_gen_pyt)
  E_gen_pyt=Pred_Pyt(Img_gen,Par1.detach())
  E_gen_pen=Pred_Pen(Img_gen,Par1.detach().to('cpu'))
  E_ref_pyt=Pred_Pyt(Img_ref,Par1.detach())
  E_ref_pen=Pred_Pen(Img_ref,Par1.detach().to('cpu'))
  Out_Energy('E_train',Bfe_gen,E_gen_pyt,E_gen_pen)
  Out_Energy('E_test',Bfe_ref,E_ref_pyt,E_ref_pen)
  Out_E_diff(fw,1000,Bfe_gen.to('cpu'),E_gen_pen)
  Out_E_diff(fw,1000,Bfe_ref.to('cpu'),E_ref_pen)
  OutPara(Par1.detach())
  fw.close()

start=datetime.datetime.now()
Main()
end=datetime.datetime.now()
print("執行時間：",end-start)

