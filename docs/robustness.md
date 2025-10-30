# Robustness quick check (px/frame)

| clip       | avg unique_ids / 5s | parked / 5s | note                         |
|------------|----------------------|-------------|------------------------------|
| dark.mp4  |  30.83              |   1.00       | slight drop; conf=0.15 ok |
| blur.mp4  |  31.58              |   1.00       | recall ↓; conf=0.12 helps |
| lowres.mp4 |  31.04              |   1.00       | small objects weaker |