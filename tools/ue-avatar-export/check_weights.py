import json,struct,sys
import numpy as np
def load(p):
    f=open(p,'rb'); f.read(12); clen,=struct.unpack('<I',f.read(4)); f.read(4); j=json.loads(f.read(clen)); blen,=struct.unpack('<I',f.read(4)); f.read(4); b=f.read(blen); return j,b
def acc(j,b,ai):
    a=j['accessors'][ai]; bv=j['bufferViews'][a['bufferView']]; off=bv.get('byteOffset',0)+a.get('byteOffset',0)
    n={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4}[a['type']]; dt={5121:np.uint8,5123:np.uint16,5125:np.uint32,5126:np.float32}[a['componentType']]
    return np.frombuffer(b,dtype=dt,count=a['count']*n,offset=off).reshape(a['count'],n)
for p in sys.argv[1:]:
    j,b=load(p); print('==',p.split('/')[-1])
    for m in j['meshes']:
        for pi,pr in enumerate(m['primitives']):
            A=pr['attributes']
            if 'WEIGHTS_0' not in A: continue
            ws=[]; 
            for k in ('WEIGHTS_0','WEIGHTS_1','WEIGHTS_2'):
                if k in A:
                    w=acc(j,b,A[k]).astype(np.float32); 
                    if j['accessors'][A[k]]['componentType']==5121: w/=255
                    elif j['accessors'][A[k]]['componentType']==5123: w/=65535
                    ws.append(w)
            tot=sum(w.sum(1) for w in ws); zero0=(ws[0].sum(1)==0).sum()
            print(f"  {m['name'][:26]:26} prim{pi} verts={len(tot):6} sum min/med/max={tot.min():.2f}/{np.median(tot):.2f}/{tot.max():.2f} bad(|sum-1|>0.05)={(abs(tot-1)>0.05).sum()} W0all0={zero0}")
