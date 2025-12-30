Perfect. Here is a clean, complete, no-BS summary of HRP, with every main step and the exact math used at each step, in the correct order.

Read this as your mental checklist for HRP.

✅ HIERARCHICAL RISK PARITY (HRP): COMPLETE STEP-BY-STEP SUMMARY
🔹 INPUTS (what you start with)

For N assets, you need only:

Volatility vector

𝜎
=
(
𝜎
1
,
…
,
𝜎
𝑁
)
σ=(σ
1
	​

,…,σ
N
	​

)

Correlation matrix

𝜌
∈
𝑅
𝑁
×
𝑁
ρ∈R
N×N

That’s it.
❌ No expected returns
❌ No optimization
❌ No matrix inversion (yet)

STEP 1️⃣ — Correlation → Distance (make clustering possible)

Clustering algorithms need distances, not correlations.

Formula (used for every pair 
𝑖
,
𝑗
i,j):
𝑑
𝑖
𝑗
=
1
−
𝜌
𝑖
𝑗
2
d
ij
	​

=
2
1−ρ
ij
	​

	​

	​

	​

Output:

Distance matrix 
𝐷
∈
𝑅
𝑁
×
𝑁
D∈R
N×N

Properties:

𝑑
𝑖
𝑗
≥
0
d
ij
	​

≥0

𝑑
𝑖
𝑗
=
𝑑
𝑗
𝑖
d
ij
	​

=d
ji
	​


𝑑
𝑖
𝑖
=
0
d
ii
	​

=0

📌 Interpretation:

High correlation → small distance

Low / negative correlation → large distance

STEP 2️⃣ — Hierarchical Clustering (build the risk tree)

Using the distance matrix, perform agglomerative hierarchical clustering.

Algorithm:

Start with N single-asset clusters

Repeatedly:

Find the closest pair of clusters

Merge them

Continue until 1 cluster (root) remains

🔗 Cluster distance (Average Linkage — HRP standard)

For two clusters 
𝐶
1
,
𝐶
2
C
1
	​

,C
2
	​

:

𝑑
(
𝐶
1
,
𝐶
2
)
=
1
∣
𝐶
1
∣
 
∣
𝐶
2
∣
∑
𝑖
∈
𝐶
1
∑
𝑗
∈
𝐶
2
𝑑
𝑖
𝑗
d(C
1
	​

,C
2
	​

)=
∣C
1
	​

∣∣C
2
	​

∣
1
	​

i∈C
1
	​

∑
	​

j∈C
2
	​

∑
	​

d
ij
	​

	​


📌 This step uses only distances, nothing else.

Output:

A binary tree (dendrogram)

Leaves = assets

Internal nodes = merged clusters

Clustering stops only when the root is formed.

STEP 3️⃣ — Quasi-Diagonalization (ordering, not math-heavy)

Reorder assets according to the tree leaf order, so that:

Correlated assets are adjacent

Covariance matrix becomes block-structured

📌 This is for numerical stability and recursion order.

(No new math here — just reordering.)

STEP 4️⃣ — Compute Covariance Matrix (needed for risk)

Before allocating, compute covariance:

Σ
𝑖
𝑗
=
𝜌
𝑖
𝑗
 
𝜎
𝑖
 
𝜎
𝑗
Σ
ij
	​

=ρ
ij
	​

σ
i
	​

σ
j
	​

	​


This matrix is used only for risk computation, not optimization.

STEP 5️⃣ — Compute Cluster Volatility (core HRP math)

For any cluster 
𝐶
C:

1️⃣ Internal weights (inverse-vol rule)
𝑤
𝑖
=
1
/
𝜎
𝑖
∑
𝑗
∈
𝐶
1
/
𝜎
𝑗
w
i
	​

=
∑
j∈C
	​

1/σ
j
	​

1/σ
i
	​

	​

	​

2️⃣ Cluster variance
𝜎
𝐶
2
=
𝑤
⊤
Σ
𝐶
𝑤
σ
C
2
	​

=w
⊤
Σ
C
	​

w
	​

3️⃣ Cluster volatility
𝜎
𝐶
=
𝜎
𝐶
2
σ
C
	​

=
σ
C
2
	​

	​

	​


📌 This answers:

“How risky is this cluster if capital is spread evenly by risk inside it?”

STEP 6️⃣ — Recursive Bisection (allocate capital)

Now HRP walks the tree from top to bottom.

At each split:

Parent cluster
├── Left child
└── Right child

Allocation rule (THIS IS THE KEY FORMULA)

If left and right clusters have volatilities 
𝜎
𝐿
,
𝜎
𝑅
σ
L
	​

,σ
R
	​

:

𝑤
𝐿
=
𝜎
𝑅
𝜎
𝐿
+
𝜎
𝑅
,
𝑤
𝑅
=
𝜎
𝐿
𝜎
𝐿
+
𝜎
𝑅
w
L
	​

=
σ
L
	​

+σ
R
	​

σ
R
	​

	​

,w
R
	​

=
σ
L
	​

+σ
R
	​

σ
L
	​

	​

	​


📌 Riskier cluster → less capital
📌 Safer cluster → more capital

Apply recursively:

Root → 2 clusters

Each cluster → its children

Continue until individual assets

Final asset weight = product of weights along its path

STEP 7️⃣ — Final Output

You now have:

𝑤
=
(
𝑤
1
,
…
,
𝑤
𝑁
)
w=(w
1
	​

,…,w
N
	​

)
	​


Properties:

Fully invested

Risk-diversified by structure

No expected returns

Stable under estimation error

🧠 ONE-PAGE MENTAL MODEL (MEMORIZE THIS)
Correlation
   ↓
Distance
   ↓
Hierarchical clustering (build tree)
   ↓
Compute cluster risk (portfolio volatility)
   ↓
Recursive bisection (inverse-risk allocation)
   ↓
Final weights