# Goal runtime and usage stop condition

- User requested on 2026-10-04 that goal-mode pause at the earlier of:
  1. 2026-10-04 04:44:22 America/New_York (two hours from request handling), or
  2. 7 percentage points of weekly token usage from the request.
- Goal accounting observed immediately before setting the deadline: 95,406 tokens.
- The agent cannot inspect the account's weekly usage denominator or percentage.
  Therefore condition 2 needs an exact token allowance from the user to become
  machine-checkable. Do not infer one.
- Until then, condition 1 is the enforceable hard upper bound. At the first safe
  goal turn boundary at or after the deadline, call `update_goal(status="paused")`,
  report the pause, and perform no further substantive work.
- Before starting each new experiment, check the current time and remaining goal
  accounting. Avoid starting work that cannot be cleanly recorded before the deadline.
