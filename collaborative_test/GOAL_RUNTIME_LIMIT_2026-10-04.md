# Goal runtime and usage stop condition

- User replaced the earlier mixed time/token condition with one limit:
  run for 1 hour 30 minutes from 2026-10-04 02:45:42 America/New_York.
- Hard deadline: 2026-10-04 04:15:42 America/New_York.
- The prior two-hour / weekly-7-percent condition is superseded.
- At the first safe goal turn boundary at or after the deadline, call
  `update_goal(status="paused")`,
  report the pause, and perform no further substantive work.
- Before starting each new experiment, check the current time and remaining goal
  accounting. Avoid starting work that cannot be cleanly recorded before the deadline.
