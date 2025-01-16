# Specialization-Based Coordination for Multi-Robot Teams

## Introduction
Coordinating multiple robots to accomplish a common objective in communication-constrained environments is a challenging task.
Coordination strategies are typically classified as either centralized or decentralized.
Centralized approaches can provide globally-optimal solutions, but require a strong connection with all robots and struggle with scalability as the problem size increases.
Decentralized methods address these issues by enabling each robot to make their own decisons using control rules that guide their actions given local observations.
It is, however, challenging to design or learn control rules that result in action selections that are aligned with the global optimal as deployed robots do not have access to global information.

## Approach
Here we consider how a Coordinator robot could be used to enhance the actions of a team of Worker robots operating in a decentralized manner.
We define a Worker's actions as contributing to multiple modalities (i.e. a Worker my move one step to the right, which impacts an "approach nearest task" modality and an "approach nearest robot" modality).
On their own, Workers may struggle to identify which modality to optimize for at a given time step.
Our approach aggregates Workers' observations at a Coordinator robot to assemble a global state.
The Coordinator then processes its global world map to assign modality weights, or "specializations," to each Worker such that the combined specialized team actions are aligned with the global optimal.
