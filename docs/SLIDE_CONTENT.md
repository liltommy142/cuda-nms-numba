
# Real-Time NMS on GPU

[slide-google](https://docs.google.com/presentation/d/10enq7N5OOawB4k1h8jwsGlImZjNQYfnySDcQV4pCLOE/edit?usp=sharing)

CUDA (Numba) · Topic A4 · Group 11
Lê Quang Tân · Phùng Quốc Tuấn

[Visual: single object photo with 5+ overlapping detection boxes around it]

---

# The Problem

Thousands of boxes → 1 correct box
Serial NMS = O(n²) bottleneck

[Visual: before/after image — messy overlapping boxes → clean single box]

---

# Why GPU

IoU pairs: independent → parallel
Suppression: sequential → the real challenge

[Visual: grid of parallel arrows (IoU) next to a single chained arrow (suppression)]

---

# Roadmap

CPU → V1 → V2 → V3
1× → 5-10× → 15× → 30-80×

[Visual: horizontal roadmap/arrow with 4 milestones and speedup badges]

---

# Real Results (Colab T4)

N=10,000 → **9.7× faster**

[Visual: simple bar chart, CPU bar vs GPU V1 bar, 3 groups for N=100/1,000/10,000]

---

# Where We Are

V1 ✅ done — V2 ⏳ — V3 ⏳

[Visual: 3-step progress tracker, step 1 checked, steps 2-3 pending]

---

# Goals

75% · 100% · 125%

[Visual: 3-tier ladder/podium graphic labeled V1 / V2 / V3]

---

# Team

Tuấn — CPU + tests
Tân — GPU kernel

[Visual: two simple avatar/icon cards side by side]

---

# Thank You
