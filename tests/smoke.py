#!/usr/bin/env python3
"""Small end-to-end float/uint8/signed-int8 build, save, reload and search check."""
import argparse,json,os,struct,subprocess,tempfile
from pathlib import Path
import numpy as np

def write(p,x):
 with p.open('wb') as f:f.write(struct.pack('<II',*x.shape));x.tofile(f)
def main():
 a=argparse.ArgumentParser();a.add_argument('--build-dir',type=Path,required=True);args=a.parse_args();root=Path(__file__).resolve().parents[1];cli=root/'scripts/run.py';rng=np.random.default_rng(12345)
 if 'avx512f' not in Path('/proc/cpuinfo').read_text():print('SKIP: AVX-512 CPU required');return
 for dtype,dim in [('float32',96),('uint8',128),('int8',100)]:
  with tempfile.TemporaryDirectory(prefix='lowmem-pipnn-test-') as temp:
   f=Path(temp);n=2048;nq=32
   if dtype=='float32':x=rng.normal(size=(n,dim)).astype('float32');q=x[:nq].copy()
   elif dtype=='uint8':x=rng.integers(0,256,size=(n,dim),dtype=np.uint8);q=x[:nq].copy()
   else:x=rng.integers(-128,128,size=(n,dim),dtype=np.int8);q=x[:nq].copy()
   xf=x.astype('float64');qf=q.astype('float64');ds=((qf[:,None,:]-xf[None,:,:])**2).sum(2);gt=np.argsort(ds,axis=1,kind='stable')[:,:100].astype('<u4');write(f/'base.bin',x);write(f/'query.bin',q);write(f/'gt.ibin',gt)
   common=['python3',str(cli),'--build-dir',str(args.build_dir.resolve())];build=subprocess.run(common+['build','--base',str(f/'base.bin'),'--index',str(f/'idx'),'--threads','4','--dtype',dtype],capture_output=True,text=True)
   if build.returncode:raise RuntimeError(build.stdout+'\n'+build.stderr)
   command=common+['search','--index',str(f/'idx'),'--query',str(f/'query.bin'),'--gt',str(f/'gt.ibin'),'--dtype',dtype,'--threads','4','--L','20,100','--rounds','2'];search=subprocess.run(command,capture_output=True,text=True)
   if search.returncode:raise RuntimeError(search.stdout+'\n'+search.stderr)
   rows=[dict(part.split('=',1) for part in line.split()[1:]) for line in search.stdout.splitlines() if line.startswith('SEARCH ')];assert len(rows)==4
   # A small beam has lower recall on random Gaussian data; require high recall at L100.
   for L in ['20','100']:
    recall=[float(r['recall']) for r in rows if r['ef']==L];assert len(set(recall))==1 and recall[0]>=(.8 if L=='100' else .4),(dtype,L,recall)
   again=subprocess.run(command,capture_output=True,text=True,check=True);rr=[line.split('recall=')[1].split()[0] for line in again.stdout.splitlines() if line.startswith('SEARCH ')];assert rr==[r['recall'] for r in rows]
   print(json.dumps(dict(dtype=dtype,d=dim,n=n,nq=nq,save_load=True,repeat_recall=True,recall_L100=rows[-1]['recall'])),flush=True)
if __name__=='__main__':main()
