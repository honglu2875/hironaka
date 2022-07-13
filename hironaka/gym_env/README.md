# hironaka.gym_env
This is a gym wrapper of hironaka host/agent environment.

Note:
 - The naming is indeed confusing, and I do not see a better way of naming them. **Please remember**
   - HironakaAgentEnv: a gym environment that takes an `Agent` as an initialization parameter. The `Agent` object is fixed throughout the game, and receives actions from an unknown `Host`.
   - HironakaHostEnv: a gym environment that takes an `Host` as an initialization parameter. The `Host` object is fixed throughout the game, and receives actions from an unknown `Agent`.

