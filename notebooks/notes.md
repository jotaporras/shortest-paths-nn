# Discussion: relationship between radius and network receptive field

Range: physical area
Receptive field: effective range of arch. 


1. Fix range and increase resolution (manifold convergence case)
    - $K$ decreases, epsilon decreases => receptive field also decreases.
2. Increase range and resolution, density is fixed.
    - Local weak convergence.
    - Receptive fields remains the same.
    - Application: wireless nets, Fred, Romina cases
3. Match range and receptive field: Fix range, increase resolution.
    - Then receptive field must increase with resolution.


The target function is radius dependent. Target function and setting (task) determine if we are in 1,2,3.

data, model => we can control (r,k).


# Next steps
1. Can SGT perform well in irregular data?
    - Can we perform well with fewer samples? 
    - Does it transfer?
    - Does the particular construction matter? 
    - Evaluate on cross terrain.

- Followup: looped transformer?
- Presentation (15-20 minutes).


# Paper outline
1. GT/SGT and manifold counterparts.
2. Sampling strategies.
3. Terrain experiments.