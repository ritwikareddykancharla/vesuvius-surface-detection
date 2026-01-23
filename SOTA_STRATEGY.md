# 🏆 Strategy: Reaching 0.65+ (The "Holy Grail" Plan)

The Host Baseline (0.562) is limited by its focus on local pixel accuracy and basic skeletonization. To hit **0.65+**, we need to solve the "Global Continuity" problem. Here is the blueprint.

## 1. Architecture: Swin-UNetR (3D Transformers)
Standard CNNs (nnU-Net) have a limited receptive field. In a tangled scroll, a sheet might disappear for a few voxels and reappear elsewhere. 
- **Upgrade**: Move from Res-UNet to **Swin-UNetR**.
- **Why**: The self-attention mechanism allows the model to "attend" to the continuation of a sheet even if it's distorted by noise or compression.

## 2. Loss: Combined "Topo-Hybrid" Objective
The current baseline uses `MedialSurfaceRecall`. We should use a tri-fusion loss:
$$ Loss = w_1 \cdot BCE + w_2 \cdot clDice + w_3 \cdot \text{Soft-Betti} $$
- **`clDice`**: Good for overall skeleton.
- **`Soft-Betti`**: Uses persistent homology to directly optimize Betti numbers (the actual TopoScore metric). 
- **Warmup Strategy**: Train with BCE for 10 epochs, then introduce `clDice`, then finalize with `Soft-Betti` for the last 5 epochs to "stitch" the topology together.

## 3. Data: The "Tangled" Augmentation Pipeline
The competition description notes that "tangled areas are where the real discoveries hide."
- **Deformation Augmentation**: Use `ElasticTransform3D` and simulated **"Cracks"** (random 3D voids) during training.
- **Hard Patch Mining**: Use a preliminary model to find patches with low **VOI scores** (where the model is confused by mergers) and over-sample those patches in the next training cycle.

## 4. Feature Engineering: Distance-to-Center (Umbilicus)
The recto surface faces the center. We can compute a **3D Radial Distance Map** from the center of the scroll.
- **Metadata Injection**: Pass this distance as a **4th channel** to the 3D model. This gives the model "directional awareness," helping it distinguish between "inner" and "outer" layers that look identical locally.

## 5. Post-Processing: Graph-Based Component Stitching
Instead of just a Frangi filter, implement a **Graph Refinement** stage:
1.  **Extract Components**: Find all 3D connected components (using 26-connectivity).
2.  **Predict Links**: For any two components that are close, use a small MLP to predict if they belong to the same sheet based on the **Vector Alignment** of their surface normals.
3.  **Stitch**: Bridge the components if the probability is high. This directly boosts the **VOI Split** and **TopoScore**.

## 6. Ensemble: Nelder-Mead Optimization
Don't just average models. Use **Nelder-Mead optimization** to find the weights for an ensemble of:
- 1x Swin-UNetR (Transformer-based)
- 1x Res-UNet (CNN-based)
- 1x nnU-Net (Baseline-optimized)

---

### 📅 Implementation Roadmap
1.  **Week 1**: Implement Swin-UNetR and the 4-channel metadata injection.
2.  **Week 2**: Integrate the `clDice` + `Soft-Betti` loss.
3.  **Week 3**: Build the Graph Stitching post-processor.
