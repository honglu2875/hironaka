from typing import Dict, List, Tuple, Type, Union

import torch


class ReplayBuffer:
    """
    Replay buffer. Most of the codes are inspired by stable-baselines3, but the central data type is Tensor instead
        of np.ndarray.
    This is the base class that does things in basic fashions.

    An experience is the following tuple (order matters!!):
        observations (self.dtype),
        actions (torch.int32),
        rewards (torch.float32),
        dones (torch.bool),
        next_observations (self.dtype)
    """

    def __init__(
        self,
        input_shape: Union[Dict, Tuple],
        output_dim: int,
        buffer_size: int,
        device: torch.device,
        dtype: Union[Type, torch.dtype] = torch.float32,
        **kwargs,
    ):
        """
        Parameters:
            input_shape: Either a tuple or a dict of tuples. The shape of the input.
            output_dim: int. The dimension of the output.
            buffer_size: int. The maximal size (on the 0-th axis, the batch-dimension).
            device: torch.device. Specifies which device to save.
        """
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.buffer_size = buffer_size
        self.device = device
        self.dtype = dtype

        if isinstance(self.input_shape, dict):
            self.observations = {}
            self.next_observations = {}
            for key in self.input_shape:
                self.observations[key] = torch.zeros(
                    (self.buffer_size, *self.input_shape[key]), device=self.device, dtype=self.dtype
                )
                self.next_observations[key] = torch.zeros(
                    (self.buffer_size, *self.input_shape[key]), device=self.device, dtype=self.dtype
                )
        else:
            self.observations = torch.zeros((self.buffer_size, *self.input_shape), device=self.device, dtype=self.dtype)
            self.next_observations = torch.zeros((self.buffer_size, *self.input_shape), device=self.device, dtype=self.dtype)

        self.actions = torch.zeros((self.buffer_size, 1), device=self.device, dtype=torch.int32)
        self.rewards = torch.zeros((self.buffer_size, 1), device=self.device, dtype=torch.float32)
        self.dones = torch.zeros((self.buffer_size, 1), device=self.device, dtype=torch.bool)

        self.pos = 0
        self.full = False

    def add(
        self,
        obs: Union[torch.Tensor, Dict],
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        next_obs: Union[torch.Tensor, Dict],
        clone=True,
    ):
        """
        Add a batch of experiences.
        Parameters:
            obs: observations
            action: actions
            reward: rewards
            done: whether the corresponding game has finished
            next_obs: the next observation
            clone: (Optional) whether to clone the input data before putting into replay buffer

        (Note that the types of the inputs will be forced into:
            obs: self.dtype
            action: torch.int32
            reward: torch.float32
            done: torch.bool
            next_obs: self.dtype)
        """
        # Shape checks
        assert action.shape[1:] == torch.Size([1])
        assert reward.shape[1:] == torch.Size([1])
        assert done.shape[1:] == torch.Size([1])
        length = action.shape[0]
        assert self.buffer_size > length, f"{length} samples are more than the buffer size."
        # Force the data type to be correct if mismatch
        self._make_types([obs, action, reward, done, next_obs])

        for storage, data in zip(
            [self.observations, self.actions, self.rewards, self.dones, self.next_observations],
            [obs, action, reward, done, next_obs],
        ):
            # The members are potentially dicts. In this case, one needs to update each Tensor inside.
            # Tensor is mutable. We copy them by reference and update them inline.
            each_storage = []
            each_data = []
            if isinstance(storage, dict):
                for key in storage:
                    each_storage.append(storage[key])
                    each_data.append(data[key])
            else:
                each_storage.append(storage)
                each_data.append(data)

            # Update each Tensor
            for target, source in zip(each_storage, each_data):
                if self.pos + length < self.buffer_size:
                    target[self.pos : self.pos + length] = self.set_value(source, clone=clone, device=self.device)
                else:  # If full, roll back.
                    target[self.pos : self.buffer_size] = self.set_value(
                        source[: self.buffer_size - self.pos], clone=clone, device=self.device
                    )
                    target[: length + self.pos - self.buffer_size] = self.set_value(
                        source[self.buffer_size - self.pos :], clone=clone, device=self.device
                    )

        self.full = self.full or (length + self.pos) >= self.buffer_size
        self.pos = (length + self.pos) % self.buffer_size

    def sample(self, batch_size: int, device: torch.device = None, clone: bool = True) -> Tuple:
        sample_index = self.pos if not self.full else self.buffer_size
        assert sample_index > 0

        # Generate the random indices and repeat to conform with the shapes
        rand_index = torch.randint(sample_index, (batch_size,), dtype=torch.int64)

        # Map the indices to the tensors
        replays = [self.observations, self.actions, self.rewards, self.dones, self.next_observations]

        experience = []
        for data in replays:
            if isinstance(data, dict):
                d = {}
                for key in data:
                    d[key] = self.set_value(data[key][rand_index], clone=clone, device=device)
            else:
                d = self.set_value(data[rand_index], clone=clone, device=device)
            experience.append(d)

        return tuple(experience)

    def reset(self):
        self.pos = 0
        self.full = False

    def _make_types(self, exp: List):
        exp[0] = self._check_and_fix_dtype(exp[0], self.dtype)  # observation
        exp[1] = self._check_and_fix_dtype(exp[1], torch.int32)  # action
        exp[2] = self._check_and_fix_dtype(exp[2], torch.float32)  # reward
        exp[3] = self._check_and_fix_dtype(exp[3], torch.bool)  # done
        exp[4] = self._check_and_fix_dtype(exp[4], self.dtype)  # next_observation

    @staticmethod
    def _check_and_fix_dtype(t: Union[torch.Tensor, dict], dtype: torch.dtype) -> Union[torch.Tensor, dict]:
        if isinstance(t, dict):
            # `dict` is mutable. Directly modify it instead of making a copy to save some efforts and spaces.
            for key, value in t.items():
                t[key] = value if value.dtype == dtype else value.type(dtype)
            return t
        elif isinstance(t, torch.Tensor):
            return t if t.dtype == dtype else t.type(dtype)
        else:
            raise TypeError(f"Unsupported type. Got {type(t)}.")

    @staticmethod
    def set_value(t: torch.Tensor, clone: bool, device: torch.device) -> torch.Tensor:
        return t.to(device).clone() if clone else t.to(device)
