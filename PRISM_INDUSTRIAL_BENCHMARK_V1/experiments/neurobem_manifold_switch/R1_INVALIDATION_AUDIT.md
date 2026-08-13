# R1 invalidation audit

R1 is retained but is not scientific evidence. After its formal test, SHA256
comparison showed that 11 of 12 files in the released `valid/` directory were
byte-identical to test trajectories under different names. One released train
file was also byte-identical to `random_points.csv`. Consequently R1 monitor
calibration accessed duplicated test content before the formal test.

Status: `INVALID_TEST_DUPLICATED_IN_CALIBRATION_SET`.

R2 makes no numerical threshold change. It refits the same PRISM route family
and W family using the earliest 75% of released train parent flights, calibrates
the unchanged monitor on the latest 25% of train parent flights, and excludes
every train file whose SHA256 matches a test file. Parent-flight chronology and
entity isolation are preserved.

