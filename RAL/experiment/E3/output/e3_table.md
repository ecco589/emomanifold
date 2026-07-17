# E3 VAD axis orthogonality and cluster cohesion (pole_n=50, Qwen2.5-0.5B)

Cluster cohesion is the mean intra-cluster pairwise cosine over five predefined multilingual emotion sets (five languages each). The random baseline is shared across rows (100 trials per cluster, no scaler); value shown in parentheses.

| Pole lexicon | \|cos(V,A)\| | \|cos(A,D)\| | \|cos(V,D)\| | Cluster cos. (rand.) |
| --- | --- | --- | --- | --- |
| EN-only | 0.151 | 0.194 | 0.153 | 0.913 (0.410) |
| ZH-only | 0.220 | 0.116 | 0.186 | 0.898 (0.410) |
