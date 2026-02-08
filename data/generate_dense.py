
#!/usr/bin/env python3
import numpy as np, argparse
p=argparse.ArgumentParser()
p.add_argument('--n', type=int, default=512)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--A', default='A.csv')
p.add_argument('--x', default='x.csv')
args=p.parse_args()
rng=np.random.default_rng(args.seed)
A=rng.random((args.n,args.n))
x=rng.random(args.n)
np.savetxt(args.A, A, delimiter=',')
np.savetxt(args.x, x, delimiter=',')
print('Wrote', args.A, 'and', args.x)
