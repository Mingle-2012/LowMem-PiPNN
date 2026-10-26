#!/usr/bin/env python3
"""Build, load, or benchmark the released R32 LowMem-PiPNN method."""
import argparse,os,subprocess,sys
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build-dir',type=Path,default=Path(__file__).resolve().parents[1]/'build');sub=p.add_subparsers(dest='command',required=True)
 for name in ['build','search','benchmark']:
  a=sub.add_parser(name);a.add_argument('--index',type=Path,required=name!='benchmark',default=Path('unused-index'));a.add_argument('--threads',type=int,default=128);a.add_argument('--dtype',choices=['float32','uint8','int8'],default='float32');a.add_argument('--query-dtype',choices=['float32','uint8','int8'],default=None)
  if name in ['build','benchmark']:a.add_argument('--base',type=Path,required=True)
  if name in ['search','benchmark']:
   a.add_argument('--query',type=Path,required=True);a.add_argument('--gt',type=Path,required=True);a.add_argument('--L',default='20,40,60,80,100,150,200,300,400');a.add_argument('--rounds',type=int,default=5)
 args=p.parse_args()
 if not 1<=args.threads<=128:p.error('--threads must be1..128')
 flags=Path('/proc/cpuinfo').read_text().split('flags',1)[-1].split('\n',1)[0].split()
 if not {'avx512f','avx512bw','avx512dq','avx512vl','f16c'}.issubset(flags):p.error('AVX-512 F/BW/DQ/VL and F16C are required')
 binary=args.build_dir/('lowmem-pipnn-int8' if args.dtype=='int8' else 'lowmem-pipnn-float')
 if not binary.is_file():p.error(f'{binary} is missing; build the CMake project first')
 env=dict(os.environ)
 for k in list(env):
  if k.startswith('LOWMEM_'):env.pop(k)
 env.update(LOWMEM_THREADS=str(args.threads),OMP_NUM_THREADS=str(args.threads),OMP_THREAD_LIMIT=str(args.threads),OMP_PROC_BIND='false',OMP_WAIT_POLICY='PASSIVE',PARLAY_NUM_THREADS=str(args.threads),PARLAY_ELASTIC_PARALLELISM='true',PIPNN_BUILD_SEED='12345',PIPNN_ROOT_SUBTREE_BATCH='32',PIPNN_AEA_EVIDENCE_FLOOR='64',PIPNN_AEA_OVERFLOW_MIB='0',LOWMEM_DEGREE='32',LOWMEM_BUILD_EF='400',LOWMEM_TWOHOP_NEW_TOP='64',LOWMEM_PRE_MODE='1',LOWMEM_POST_MODE='2',LOWMEM_G0_REVERSE_DEPTH='1',LOWMEM_PRE_OPT='1',LOWMEM_INPUT_DTYPE=args.dtype,LOWMEM_QUERY_DTYPE=args.query_dtype or args.dtype,LOWMEM_METRIC='L2',LOWMEM_B64_ALPHA='1.2',LOWMEM_UNIQUE_VECTOR='1',LOWMEM_UNIQUE_FROZEN='0',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',BLIS_NUM_THREADS='1',OMP_MAX_ACTIVE_LEVELS='1')
 if args.command!='build':
  try:grid=[int(s) for s in args.L.split(',')]
  except ValueError:p.error('--L requires comma-separated integers')
  if not grid or any(n<10 for n in grid) or args.rounds<1:p.error('L must be>=10 and rounds positive')
  env.update(LOWMEM_QUERY=str(args.query.resolve()),LOWMEM_GT=str(args.gt.resolve()),LOWMEM_SEARCH_ROUNDS=str(args.rounds),LOWMEM_SEARCH_GRID=args.L)
 args.index.parent.mkdir(parents=True,exist_ok=True)
 if args.command=='search':cmd=[str(binary.resolve()),'search',str(args.index.resolve())]
 else:
  if args.command=='benchmark':env['LOWMEM_NO_SAVE']='1'
  elif any(Path(str(args.index)+ext).exists() for ext in ['.meta','.vec','.blocks']):p.error('index prefix already exists; choose a new prefix')
  cmd=[str(binary.resolve()),'build',str(args.base.resolve()),str(args.index.resolve())]
 return subprocess.run(cmd,env=env).returncode
if __name__=='__main__':sys.exit(main())
