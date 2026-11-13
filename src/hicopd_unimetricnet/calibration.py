from __future__ import annotations
import numpy as np


def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-40,40)))

def logit(p):
    p=np.clip(np.asarray(p,float),1e-7,1-1e-7)
    return np.log(p/(1-p))

def _golden_search(func,a,b,tol=1e-6,max_iter=120):
    gr=(5**0.5-1)/2
    c=b-gr*(b-a); d=a+gr*(b-a); fc=func(c); fd=func(d)
    for _ in range(max_iter):
        if abs(b-a)<tol: break
        if fc<fd:
            b,d,fd=d,c,fc; c=b-gr*(b-a); fc=func(c)
        else:
            a,c,fc=c,d,fd; d=a+gr*(b-a); fd=func(d)
    return (a+b)/2

def fit_temperature(y,p):
    y=np.asarray(y,float); z=logit(p)
    if len(np.unique(y))<2: raise ValueError('Temperature calibration requires both endpoint classes')
    def obj(logt):
        t=np.exp(logt); q=sigmoid(z/t); return float(-np.mean(y*np.log(q+1e-8)+(1-y)*np.log(1-q+1e-8)))
    return float(np.exp(_golden_search(obj,np.log(.05),np.log(20.0))))

def apply_temperature(p,t): return sigmoid(logit(p)/float(t))

def fit_robust_temperature(frame, domain_col='logical_dataset', grid_points=121):
    """Choose one source-only temperature minimizing worst-domain NLL.

    The target domain is never consulted. Domains without both classes still
    contribute NLL but cannot dominate via undefined discrimination metrics.
    """
    y=np.asarray(frame['y'],int); p=np.asarray(frame['probability'],float); d=np.asarray(frame[domain_col],str); z=logit(p)
    if len(np.unique(y))<2: raise ValueError('Robust calibration requires both endpoint classes in source calibration data')
    temps=np.exp(np.linspace(np.log(.15),np.log(8.0),int(grid_points)))
    best=(float('inf'),1.0)
    for t in temps:
        q=sigmoid(z/t); vals=[]
        for dom in np.unique(d):
            m=d==dom; yy=y[m]; qq=np.clip(q[m],1e-7,1-1e-7); vals.append(float(-np.mean(yy*np.log(qq)+(1-yy)*np.log(1-qq))))
        score=max(vals)+0.10*float(np.mean(vals))
        if score<best[0]: best=(score,float(t))
    return best[1]
