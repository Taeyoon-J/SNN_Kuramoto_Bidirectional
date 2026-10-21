"""Resource-only wrapper; frozen scientific worker and fingerprint stay intact."""
import argparse,math

POLICY='shared_8GiB_admission_no_foreign_yield_v3'
OBSERVED_RESIDENT_MIB=2234

def required_free_mib(total,observed=OBSERVED_RESIDENT_MIB):
    if total<=0 or observed<=0:raise ValueError('positive memory sizes required')
    # Conservative floor exceeds 3x observed post-spike-preflight residency.
    # This is admission headroom, not an asserted upper bound on allocation.
    return max(8192,math.ceil(3*observed)+1024)

def main(model,seed):
    from . import queue as original
    original.scope.require(model,seed)
    original.policy.required_free_mib=required_free_mib
    original.policy.NAME=POLICY
    original.supervise(model,seed)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--model',choices=('240','237','244','245'),required=True)
    parser.add_argument('--seed',type=int,choices=(0,1,2),required=True);args=parser.parse_args()
    main(args.model,args.seed)
