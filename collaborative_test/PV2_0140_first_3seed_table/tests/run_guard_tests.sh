#!/bin/bash
# The guards in agg3.py are the only thing standing between an unattended run and a report
# that claims a 3-seed mean it does not have. They are therefore tested, not assumed.
#
# Fixtures: ok_s{0,1,2}.json are three distinct runs; same_{a,b,c}.json are one run copied
# three times; other_ds.json belongs to a different dataset.
cd "$(dirname "$0")"
A=../code/agg3.py
fail=0
check () {  # check <description> <expect pass|refuse> <args...>
  d=$1; want=$2; shift 2
  out=$(python3 $A "$@" 2>&1); got=refuse
  grep -q "^wrote" <<<"$out" && got=pass
  if [ "$got" = "$want" ]; then echo "ok    $d"; else
    echo "FAIL  $d (wanted $want, got $got)"; echo "$out" | tail -2; fail=1; fi
}
check "three distinct runs are accepted" pass \
  --results ok_s0.json ok_s1.json ok_s2.json --label t --key head_recipe --expect-seeds 3 --out /tmp/g1.json
check "one run copied three times is refused" refuse \
  --results same_a.json same_b.json same_c.json --label t --key head_recipe --expect-seeds 3 --out /tmp/g2.json
check "two results when three expected is refused" refuse \
  --results ok_s0.json ok_s1.json --label t --key head_recipe --expect-seeds 3 --out /tmp/g3.json
check "results spanning two datasets are refused" refuse \
  --results ok_s0.json ok_s1.json other_ds.json --label t --key head_recipe --expect-seeds 3 --out /tmp/g4.json
exit $fail
